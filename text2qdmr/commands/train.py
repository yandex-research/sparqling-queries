import argparse
import collections
import datetime
import json
import os
import shutil
import warnings
import subprocess

import _jsonnet
import attr
from contextlib import nullcontext
from packaging import version

import torch
from torch.utils.tensorboard import SummaryWriter

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from text2qdmr import datasets
# noinspection PyUnresolvedReferences
from text2qdmr import model

from text2qdmr.utils import registry
from text2qdmr.utils import random_state
from text2qdmr.utils import saver as saver_mod

# noinspection PyUnresolvedReferences
from text2qdmr.utils import vocab


@attr.s
class TrainConfig:
    eval_every_n = attr.ib(default=100)
    report_every_n = attr.ib(default=100)
    save_every_n = attr.ib(default=100)
    keep_every_n = attr.ib(default=1000)
    max_keep = attr.ib(default=100)

    batch_size = attr.ib(default=32)
    eval_batch_size = attr.ib(default=32)
    max_steps = attr.ib(default=100000)
    num_eval_items = attr.ib(default=None)
    eval_on_train = attr.ib(default=True)
    eval_on_val = attr.ib(default=True)

    # Seed for RNG used in shuffling the training data.
    data_seed = attr.ib(default=None)
    # Seed for RNG used in initializing the model.
    init_seed = attr.ib(default=None)
    # Seed for RNG used in computing the model's training loss.
    # Only relevant with internal randomness in the model, e.g. with dropout.
    model_seed = attr.ib(default=None)

    num_batch_accumulated = attr.ib(default=1)
    clip_grad = attr.ib(default=None)


class Logger:
    def __init__(self, log_path=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'a+')

    def log(self, msg):
        formatted = f'[{datetime.datetime.now().replace(microsecond=0).isoformat()}] {msg}'
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, 'a+')
            else:
                self.log_file.flush()

    def log_system_info(self):
        try:
            git_commit_id = subprocess.check_output(["git", "show", "-s", "--pretty=format:'%H'"])
            git_commit_id = git_commit_id.decode()
            self.log(f"Running on git commit {git_commit_id}")
        except Exception as error:
            self.log(f"Could not identity git commit: {type(error)}: {error}")

        try:
            conda_info = subprocess.check_output(["conda", "info"])
            conda_info = conda_info.decode()
            self.log(f"Result of conda info:")
            self.log(conda_info)
        except Exception as error:
            self.log(f"Could not run 'conda info': {type(error)}: {error}")

        try:
            import torch
            self.log(f"pytorch version: {torch.__version__}")
        except Exception as error:
            self.log(f"Could not determine pytorch version: {type(error)}: {error}")

        try:
            import transformers
            self.log(f"transformers version: {transformers.__version__}")
        except Exception as error:
            self.log(f"Could not determine transformers version: {type(error)}: {error}")


class Trainer:
    def __init__(self, logger, config, distributed=False):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.logger = logger
        self.train_config = registry.instantiate(TrainConfig, config['train'])

        # adding multi GPU support
        self.distributed = distributed
        if self.distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.num_gpus = int(os.environ["WORLD_SIZE"])
            assert self.train_config.num_batch_accumulated >= self.num_gpus, f"Need train_config.num_batch_accumulated >= {self.num_gpus} to run on {self.num_gpus} GPUs, but have only {self.train_config.num_batch_accumulated}"
            if self.train_config.num_batch_accumulated % self.num_gpus != 0:
                warnings.warn(f"Running train_config.num_batch_accumulated == {self.train_config.num_batch_accumulated} on {self.num_gpus} GPUs, GPUs will get different number of minibatches, which is inefficient. Consider changing num_batch_accumulated and batch_size.", RuntimeWarning)
        else:
            self.local_rank = 0
            self.num_gpus = 1

        # Each GPU has its own seed for data generation
        # Make extra seed a multiple of 42 (this has to help!) so that runs with successive outer seeds have different seeds here
        self.data_random = random_state.RandomContext(self.train_config.data_seed + self.local_rank * 42)

        # Make model stream different between devices to have dropout uncorrelated
        self.model_random = random_state.RandomContext(self.train_config.model_seed + self.local_rank * 42)

        # This stream should be identical across GPUs to have the same init - not critical, weight will be synced after eahc checkpoint
        self.init_random = random_state.RandomContext(self.train_config.init_seed)

        with self.init_random:
            # 0. Construct preprocessors

            if self.distributed:
                # this code helps to avoid deadlock/conflicts happening when at model construction there is a need to precompute some files
                # (e.g., preprocess Glove embeddings)
                # we do this first on the main thread and only afterwards on the others
                torch.distributed.barrier()
                if self.local_rank > 0:
                    torch.distributed.barrier()

            self.model_preproc = registry.instantiate(
                registry.lookup('model', config['model']).Preproc,
                config['model'],
                unused_keys=('name',))
            self.model_preproc.load()

            if self.distributed and self.local_rank == 0:
                torch.distributed.barrier()

            # 1. Construct model
            self.model = registry.construct('model', config['model'],
                                            unused_keys=('encoder_preproc', 'decoder_preproc'),
                                            preproc=self.model_preproc, device=self.device)
            self.model.to(self.device)

            # adding multi GPU support
            if self.distributed:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)


    def train(self, config, modeldir, tb_name):
        # slight difference here vs. unrefactored train: The init_random starts over here.
        # Could be fixed if it was important by saving random state at end of init
        with self.init_random:
            # We may be able to move optimizer and lr_scheduler to __init__ instead. Empirically it works fine. I think that's because saver.restore 
            # resets the state by calling optimizer.load_state_dict. 
            # But, if there is no saved file yet, I think this is not true, so might need to reset the optimizer manually?
            # For now, just creating it from scratch each time is safer and appears to be the same speed, but also means you have to pass in the config to train which is kind of ugly.

            # TODO: not nice
            if config["optimizer"].get("name", None) == 'bertAdamw':
                if not self.distributed:
                    bert_params = list(self.model.encoder.bert_model.parameters())
                else:
                    bert_params = list(self.model.module.encoder.bert_model.parameters())
                assert len(bert_params) > 0
                non_bert_params = []
                for name, _param in self.model.named_parameters():
                    if "bert" not in name:
                        non_bert_params.append(_param)
                assert len(non_bert_params) + len(bert_params) == len(list(self.model.parameters()))

                optimizer = registry.construct('optimizer', config['optimizer'], non_bert_params=non_bert_params,
                                               bert_params=bert_params)
                lr_scheduler = registry.construct('lr_scheduler',
                                                  config.get('lr_scheduler', {'name': 'noop'}),
                                                  param_groups=[optimizer.non_bert_param_group,
                                                                optimizer.bert_param_group])

            else:
                optimizer = registry.construct('optimizer', config['optimizer'], params=self.model.parameters())
                lr_scheduler = registry.construct('lr_scheduler',
                                                  config.get('lr_scheduler', {'name': 'noop'}),
                                                  param_groups=optimizer.param_groups)

        def sync_scheduler_optimizer(lr_scheduler, optimizer):
            lr_scheduler.param_groups = optimizer.param_groups

        # 2. Restore model parameters
        if self.distributed:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
            # when wrapped into DDP save the model itself to make it easy to load into a model without multi GPU
            model_local = self.model.module
        else:
            map_location = self.device
            model_local = self.model
        saver = saver_mod.Saver(
            {"model": model_local, "optimizer": optimizer}, 
            keep_every_n=self.train_config.keep_every_n, 
            max_keep=self.train_config.max_keep)
        last_step = saver.restore(modeldir, map_location=map_location)
        # hacky way to connect back optimizer and lr_schedular after sever.restore breaks the link
        sync_scheduler_optimizer(lr_scheduler, optimizer)

        if "pretrain" in config and last_step == 0:
            pretrain_config = config["pretrain"]
            _path = pretrain_config["pretrained_path"]
            _step = pretrain_config["checkpoint_step"]
            pretrain_step = saver.restore(_path, step=_step, map_location=map_location, item_keys=["model"])
            saver.save(modeldir, pretrain_step)  # for evaluating pretrained models
            last_step = pretrain_step

        if self.distributed:
            # give everyone time to load
            torch.distributed.barrier()

        # 3. Get training data somewhere
        with self.data_random:
            two_datasets = 'full_data' in config
            train_data = self.model_preproc.dataset('train', two_datasets=two_datasets, config=config)

            train_dataloader_params = {"batch_size" : self.train_config.batch_size,
                    "num_workers": config["num_dataloading_workers"] if "num_dataloading_workers" in config else 0, # by default load data in the main thread
                    "shuffle" : True,
                    "drop_last" : True,
                    "collate_fn" : lambda x: x}
            if version.parse(torch.__version__) >= version.parse("1.8") and train_dataloader_params["num_workers"] > 0:
                # add modern dataloader params that should improve speed
                train_dataloader_params["persistent_workers"] = True
                train_dataloader_params["prefetch_factor"] = 10

            train_data_loader = self._yield_batches_from_epochs(
                torch.utils.data.DataLoader(
                    train_data,
                    **train_dataloader_params
                    ))
        train_eval_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.train_config.eval_batch_size,
            collate_fn=lambda x: x)

        val_data = self.model_preproc.dataset('val')
        val_data_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.train_config.eval_batch_size,
            collate_fn=lambda x: x)
        if not self.distributed or self.local_rank == 0:
            if os.path.exists(tb_name):
                shutil.rmtree(tb_name)
            writer = SummaryWriter(tb_name)
        else:
            writer = None

        # log the system info
        if not self.distributed or self.local_rank == 0:
            self.logger.log_system_info()

        # 4. Start training loop
        with self.data_random:
            for batch in train_data_loader:
                # Quit if too long

                if last_step >= self.train_config.max_steps:
                    break

                # Evaluate model
                if last_step % self.train_config.eval_every_n == 0:
                    if not self.distributed or self.local_rank == 0:
                        
                        # eval only one 1 GPU because it is currently very fast
                        if self.distributed:
                            model_local = self.model.module
                        else:
                            model_local = self.model

                        if self.train_config.eval_on_train:
                            self._eval_model(self.logger, model_local, last_step, train_eval_data_loader, 'train',
                                            num_eval_items=self.train_config.num_eval_items, writer=writer)
                        if self.train_config.eval_on_val:
                            self._eval_model(self.logger, model_local, last_step, val_data_loader, 'val',
                                            num_eval_items=self.train_config.num_eval_items, writer=writer)

                    if self.distributed:
                        # wait until the evaluation is done
                        torch.distributed.barrier()


                # Compute and apply gradient
                # get the batches for processing on this device
                num_batches_to_process_this_device = 0
                for _i in range(self.train_config.num_batch_accumulated):
                    if not self.distributed or _i % self.num_gpus == self.local_rank:
                        num_batches_to_process_this_device += 1

                full_names = []
                for _i in range(num_batches_to_process_this_device):
                    is_last_item = (_i == num_batches_to_process_this_device - 1)
                    context_for_backward = self.model.no_sync if self.distributed and not is_last_item else nullcontext
                    # print(f"GPU {self.local_rank}:", [b[0]["full_name"] for b in batch])
                    full_names += [b[0]["full_name"] for b in batch]
                    with context_for_backward(), self.model_random:
                        # forward
                        loss = self.model(batch)

                        # add zero loss that touches all parameters to make reduce in DDP work better
                        # this bug might be fixed in https://github.com/pytorch/pytorch/pull/36054 which appeared in pytorch v1.6
                        dummy_loss = sum([m.view(-1)[0]*0.0 for m in self.model.parameters()])
                        loss = loss + dummy_loss.to(loss)

                        norm_loss = loss / num_batches_to_process_this_device

                        norm_loss.backward()

                    if not is_last_item:
                        batch = next(train_data_loader)


                def compute_grad_norm(param_group):
                    grad_norm = 0.0
                    for p in param_group["params"]:
                        if p.requires_grad and p.grad is not None:
                            grad_norm += p.grad.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    return grad_norm

                # Report metrics: do it before clipping grad to get actual gradients
                if last_step % self.train_config.report_every_n == 0:
                    grad_norms = [compute_grad_norm(param_group) for param_group in optimizer.param_groups]
                    grad_norms_str = ", ".join(f"{g:.4f}" for g in grad_norms)
                    self.logger.log(f'Step {last_step}: loss={loss.item():.4f}, grad norms: {grad_norms_str}')

                if self.train_config.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.bert_param_group["params"], \
                                                    self.train_config.clip_grad)
                else:
                    grad_norm = 0
                    
                optimizer.step()
                lr_scheduler.update_lr(last_step)
                optimizer.zero_grad()
                
                if writer:
                    writer.add_scalar('Loss/train', loss.item(), last_step)
                    writer.add_scalar('Norm/train', grad_norm, last_step)
                    text = '\n'.join(full_names)
                    writer.add_text('Batch', text, last_step)

                last_step += 1
                # Run saver
                if last_step == 1 or last_step % self.train_config.save_every_n == 0:
                    if not self.distributed or self.local_rank == 0:
                        saver.save(modeldir, last_step)

                    if self.distributed:
                        # wait until the model is saved
                        torch.distributed.barrier()
                        # load the model
                        last_step = saver.restore(modeldir, map_location=map_location)
                        # hack to connect back optimizer and lr_scheduler
                        sync_scheduler_optimizer(lr_scheduler, optimizer)
                        # wait until loaded
                        torch.distributed.barrier()


            # Save final model
            if not self.distributed or self.local_rank == 0:
                saver.save(modeldir, last_step)

    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch

    @staticmethod
    def _eval_model(logger, model, last_step, eval_data_loader, eval_section, num_eval_items=None, writer=None):
        stats = collections.defaultdict(float)
        model.eval()
        with torch.no_grad():
            for eval_batch in eval_data_loader:
                batch_res = model.eval_on_batch(eval_batch)
                for k, v in batch_res.items():
                    stats[k] += v
                if num_eval_items and stats['total'] > num_eval_items:
                    break
        model.train()

        # Divide each stat by 'total'
        for k in stats:
            if k != 'total':
                stats[k] /= stats['total']
        if 'total' in stats:
            del stats['total']

        kv_stats = ", ".join(f"{k} = {v}" for k, v in stats.items())
        logger.log(f"Step {last_step} stats, {eval_section}: {kv_stats}")

        if writer:
            writer.add_scalar(eval_section + '_loss', stats['loss'], last_step)



def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args


def main(args, distributed=False):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])

    # Initialize the logger
    reopen_to_flush = config.get('log', {}).get('reopen_to_flush')
    logger = Logger(os.path.join(args.logdir, 'log.txt'), reopen_to_flush)

    local_rank = int(os.environ["LOCAL_RANK"]) if distributed else 0
    if local_rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir,
                            f'config-{datetime.datetime.now().strftime("%Y%m%dT%H%M%S%Z")}.json'), 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

    logger.log(f'Logging to {args.logdir}')

    # Construct trainer and do training
    trainer = Trainer(logger, config, distributed=distributed)
    trainer.train(config, modeldir=args.logdir, tb_name=os.path.join('runs_train', args.name))


if __name__ == '__main__':
    args = add_parser()
    main(args)
