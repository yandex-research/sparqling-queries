import os
import collections
import json
import attr
import copy

import numpy as np

import torch
import torch.utils.data

from text2qdmr.model.modules import abstract_preproc
from text2qdmr.utils import registry
from text2qdmr.utils.serialization import ComplexEncoder, ComplexDecoder

from text2qdmr.datasets.qdmr import load_tables, BreakItem
from text2qdmr.datasets.utils.extract_values import ValueUnit
from qdmr2sparql.structures import prepare_dict_for_json, load_grounding_list_from_file, RdfGraph


class ZippedDataset(torch.utils.data.Dataset):
    def __init__(self, *components, process_func=None):
        assert len(components) >= 1
        lengths = [len(c) for c in components]

        self.process_func = process_func if process_func is not None else lambda x: x

        assert all(lengths[0] == other for other in lengths[1:]), f"Lengths don't match: {lengths}"
        self.components = components
        self.part = 'spider'
    
    def __getitem__(self, idx):
        item = tuple(c[idx] for c in self.components)
        item_processed = self.process_func(item)
        return item_processed
    
    def __len__(self):
        return len(self.components[0])


class TwoZippedDataset(torch.utils.data.Dataset):
    def __init__(self, *components, process_func=None):
        assert len(components) >= 1
        lengths = [len(c) for c in components]
        
        self.process_func = process_func if process_func is not None else lambda x: x

        assert all(lengths[0] == other for other in lengths[1:]), f"Lengths don't match: {lengths}"
        # components
        self.components = {'spider': [], 'full_break': []}
        self.names = list(self.components.keys())
        
        for items in zip(*components):
            if hasattr(items[0], "subset_name"):
                is_spider = items[0].subset_name == "SPIDER"
            else:
                is_spider = items[0]['subset_name'] == 'SPIDER'
            
            self.components['spider' if is_spider else 'full_break'].append(items)

        print('spider len', len(self.components['spider']))
        print('full_break len', len(self.components['full_break']))

        self.part = 'all'
    
    def __getitem__(self, idx):
        if self.part == 'all':
            # choice = np.random.choice(self.names)
            # use randomness from torch because it is taken care of autoatically when having multiple dataloading workers
            choice = torch.randint(0, len(self.names), (1, )).item()
            # choice = int(torch.bernoulli(torch.tensor([0.75])).item())
            choice = self.names[choice]
            item = self.components[choice][idx % len(self.components[choice])]
        elif self.part == 'spider':
            item = self.components['spider'][idx]
        elif self.part == 'full_break':
            item = self.components['full_break'][idx]

        item_processed = self.process_func(item)
        return item_processed
    
    def __len__(self):
        if self.part == 'all':
            return 2 * max(len(self.components['spider']), len(self.components['full_break']))
        elif self.part == 'spider':
            return len(self.components['spider'])
        elif self.part == 'full_break':
            return len(self.components['full_break'])


@registry.register('model', 'EncDec')
class EncDecModel(torch.nn.Module):
    class Preproc(abstract_preproc.AbstractPreproc):
        def __init__(
                self,
                encoder,
                decoder,
                encoder_preproc,
                decoder_preproc):
            super().__init__()

            self.enc_preproc = registry.lookup('encoder', encoder['name']).Preproc(**encoder_preproc)
            self.dec_preproc = registry.lookup('decoder', decoder['name']).Preproc(**decoder_preproc)
            
            self.items_raw = collections.defaultdict(list)
            self.data_dir = os.path.abspath(os.path.join(self.enc_preproc.data_dir, "..", "raw"))
        
        def validate_item(self, item, section):
            enc_result, enc_info = self.enc_preproc.validate_item(item, section)
            dec_result, dec_info = self.dec_preproc.validate_item(item, section, enc_info)
            return np.array(dec_result) * np.array(enc_result), (enc_info, dec_info)
        
        def add_item(self, item, section, idx_to_add, validation_info):
            enc_info, dec_info = validation_info
            self.enc_preproc.add_item(item, section, idx_to_add, enc_info)
            self.dec_preproc.add_item(item, section, idx_to_add, dec_info)
            self.add_raw_item(item, section)           

        def add_raw_item(self, item, section):
            raw_item = attr.evolve(item,
                                   schema=None, # delete schema
                                   text=item.orig_spider_entry['question'] if item.orig_spider_entry is not None else item.text, # give priority to whatever is written in orig_spider_entry
                                   orig_spider_entry=None,
                                   orig_schema=None,
                                   sql_code=None,
            )
            self.items_raw[section].append(raw_item)

        def clear_items(self):
            self.enc_preproc.clear_items()
            self.dec_preproc.clear_items()
            self.items_raw = collections.defaultdict(list)

        def save(self, partition=None):
            self.enc_preproc.save(partition=partition)
            self.dec_preproc.save(partition=partition)
            self.save_raw_items()

        def save_raw_items(self):
            os.makedirs(self.data_dir, exist_ok=True)
            for section, items in self.items_raw.items():
                with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                    for item in items:
                        item = attr.evolve(item, grounding=prepare_dict_for_json(item.grounding))
                        f.write(json.dumps(item, cls=ComplexEncoder) + '\n')

        def load_raw_dataset(self, section, paths=None):
            dataset = [BreakItem(**json.loads(line, cls=ComplexDecoder))
                        for line in open(os.path.join(self.data_dir, section + '.jsonl'))]
            # load groundings properly
            if paths:
                schemas, _ = load_tables(paths['tables_path'])
                # load spider data
                spider_data = json.load(open(paths['spider_path']))

            eval_graphs = {}
            for item in dataset:
                if item.grounding != 'None':
                    grnds = load_grounding_list_from_file(None, data={item.full_name: {"GROUNDINGS": item.grounding}})
                    grnds = grnds[item.full_name]["GROUNDINGS"]
                    item.grounding = grnds
                if paths:
                    spider_entry = spider_data[item.subset_idx]
                    item.sql_code = spider_entry['sql']
                    item.schema = schemas[item.db_id]

                    if item.db_id not in eval_graphs:
                        schema_with_data = item.schema.database_schema
                        schema_with_data.load_table_data(paths['db_path'])
                        rdf_graph = RdfGraph(schema_with_data)
                        eval_graphs[item.db_id] = (schema_with_data, rdf_graph)
                    item.eval_graphs = eval_graphs[item.db_id]

                    item.orig_spider_entry = spider_entry
                    item.orig_schema = schemas[item.db_id].orig

                values = item.values
                item.values = []
                for val_list in values:
                    val_list_new = [ValueUnit(**val_dict) for val_dict in val_list]
                    item.values.append(val_list_new)

            return dataset

        def load(self):
            self.enc_preproc.load()
            self.dec_preproc.load()

        def process_in_dataloader(self, item_tuple, section, schemas):
            item = copy.deepcopy(item_tuple[0])

            # load schema
            if item.db_id is not None:
                item = attr.evolve(item, schema=schemas[item.db_id])

            flag_shuffle_values = self.config_data["augment_at_iter_shuffle_values"]
            flag_shuffle_tables = self.config_data["augment_at_iter_shuffle_tables"]
            flag_shuffle_columns = self.config_data["augment_at_iter_shuffle_columns"]
            flag_shuffle_qdmr_ordering = self.config_data["augment_at_iter_shuffle_qdmr_ordering"]
            flag_shuffle_sort_dir = self.config_data["augment_at_iter_shuffle_sort_dir"]
            flag_shuffle_compsup_op = self.config_data["augment_at_iter_shuffle_compsup_op"]
            enc_result, enc_info = self.enc_preproc.validate_item(item, section, shuffle_values=flag_shuffle_values,
                                                                                 shuffle_tables=flag_shuffle_tables,
                                                                                 shuffle_columns=flag_shuffle_columns,
                                                                                 shuffle_sort_dir=flag_shuffle_sort_dir,
                                                                                 shuffle_compsup_op=flag_shuffle_compsup_op,
                                                                                 shuffle_qdmr_ordering=flag_shuffle_qdmr_ordering)
            dec_result, dec_info = self.dec_preproc.validate_item(item, section, enc_info)

            to_add = np.array(dec_result) * np.array(enc_result)
            if not to_add.any():
                print(f"WARNING: Item was too long at validation: will need to throw it away later")
                to_add[0] = True # hack to process items that are too long

            enc_data_new = self.enc_preproc.preprocess_item(item, to_add, enc_info)
            dec_data_new = self.dec_preproc.preprocess_item(item, to_add, dec_info)

            return (enc_data_new, dec_data_new)

        def dataset(self, section, two_datasets=False, config=None):  

            if config is not None and "use_online_data_processing" in config and config["use_online_data_processing"]:
                self.config_data = config['data'][section]
                schemas, _ = load_tables(self.config_data["paths"]["tables_path"])
                raw_dataset = self.load_raw_dataset(section)

                if two_datasets:
                    dataset = TwoZippedDataset(raw_dataset, process_func=lambda item : self.process_in_dataloader(item, section=section, schemas=schemas))  
                else:
                    dataset = ZippedDataset(raw_dataset, process_func=lambda item : self.process_in_dataloader(item, section=section, schemas=schemas)) 

                print("Loaded dataset size:", len(dataset))
                return dataset

            if two_datasets:
                return TwoZippedDataset(self.enc_preproc.dataset(section), self.dec_preproc.dataset(section))
            return ZippedDataset(self.enc_preproc.dataset(section), self.dec_preproc.dataset(section))
        
    def __init__(self, preproc, device, encoder, decoder):
        super().__init__()
        self.preproc = preproc
        self.encoder = registry.construct(
                'encoder', encoder, device=device, preproc=preproc.enc_preproc)
        self.decoder = registry.construct(
                'decoder', decoder, device=device, preproc=preproc.dec_preproc)
        
        if getattr(self.encoder, 'batched'):
            self.compute_loss = self._compute_loss_enc_batched
        else:
            self.compute_loss = self._compute_loss_unbatched

    # add regular forward to wrap into DistributedDataParallel
    def forward(self, batch):
        return self.compute_loss(batch)

    def _compute_loss_enc_batched(self, batch, debug=False):
        losses = []
        enc_states = self.encoder([enc_input for enc_input, _ in batch])

        # throw away batch elements that appeared to be too long and were not processed
        batch = [b for b, st_ in zip(batch, enc_states) if st_ is not None]
        enc_states = [st_ for st_ in enc_states if st_ is not None]
        enc_inputs = [b[0] for b in batch]
        dec_outputs = [b[1] for b in batch]

        if len(batch) == 0:
            # all batch elements were thrown away
            return torch.zeros([1])

        decoder_inputs = []
        for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
            decoder_input = self.decoder.compute_decoder_input(enc_input, dec_output, enc_state,
                                                               self.decoder.exclude_rules_loss,
                                                               self.decoder.preproc,
                                                               self.decoder.sup_att,
                                                               self.decoder.attn_type,
                                                               debug)
            decoder_inputs.append(decoder_input)

        batch_size = len(batch)
        self.decoder.state_update.set_dropout_masks(batch_size=batch_size) #(batch_size=1)

        losses_batched = self.decoder.compute_loss_batched(enc_inputs, dec_outputs, enc_states, decoder_inputs, debug)

        losses = losses_batched

        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def _compute_loss_enc_batched2(self, batch, debug=False):
        losses = []
        for enc_input, dec_output in batch:
            enc_state, = self.encoder([enc_input])
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def _compute_loss_unbatched(self, batch, debug=False):
        losses = []
        for enc_input, dec_output in batch:
            enc_state = self.encoder(enc_input)
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def eval_on_batch(self, batch):
        mean_loss = self.compute_loss(batch).item()
        batch_size = len(batch)
        result = {'loss': mean_loss * batch_size, 'total': batch_size}
        return result

    def begin_inference(self, preproc_item):
        enc_input, _ = preproc_item
        if getattr(self.encoder, 'batched'):
            enc_state, = self.encoder([enc_input])
        else:
            enc_state = self.encoder(enc_input)
        return self.decoder.begin_inference(enc_state)
