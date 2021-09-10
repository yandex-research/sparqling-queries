import argparse
import json
import os

import _jsonnet
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from text2qdmr.datasets.utils.metrics import Metrics
from text2qdmr.utils import registry
from text2qdmr.utils.serialization import ComplexDecoder


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    parser.add_argument('--section', required=True)
    parser.add_argument('--inferred', required=True)
    parser.add_argument('--output')
    parser.add_argument('--logdir')
    parser.add_argument('--virtuoso_server')
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config and args.logdir:
        args.logdir = os.path.join(args.logdir, config['model_name'])
    if args.logdir:
        args.inferred = args.inferred.replace('__LOGDIR__', args.logdir)

    inferred = open(args.inferred)
    writer = SummaryWriter(args.eval_tb_dir)

    if args.vis_dir:
        logdir_metrics = os.path.join(args.logdir, args.vis_dir)
    else:
        logdir_metrics = os.path.join(args.logdir, args.section)
    if not os.path.exists(logdir_metrics):
        os.mkdir(logdir_metrics)

    examples_with_name = {}
    for section in args.section:
        if config.get('full_data') is not None and args.part != 'spider':
            dataset = registry.construct('dataset', config['full_data'][section])
        else:
            dataset = args.data[section]
        for k, v in dataset.examples_with_name.items():
            examples_with_name[k] = v

    metrics = Metrics(writer, logdir_metrics, args.virtuoso_server)

    inferred_lines = list(inferred)
    for line in tqdm(inferred_lines):
        infer_results = json.loads(line, cls=ComplexDecoder)
        if infer_results['beams']:
            inferred_code = infer_results['beams'][0]['inferred_code']
        else:
            inferred_code = None
        assert 'name' in infer_results
        name = infer_results['name']
        section = infer_results['part']
        metrics.add(examples_with_name[name], inferred_code, section)

    metrics = metrics.finalize()

    if args.output:
        if args.logdir:
            output_path = args.output.replace('__LOGDIR__', args.logdir)
        else:
            output_path = args.output
        with open(output_path, 'w') as f:
            json.dump(metrics, f)
        print(f'Wrote eval results to {output_path}')
    else:
        print(metrics)
    return output_path


if __name__ == '__main__':
    args = add_parser()
    main(args)
