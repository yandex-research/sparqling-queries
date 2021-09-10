import argparse
import itertools
import json
import os
import sys
import random

import _jsonnet
import torch
import tqdm

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from text2qdmr.utils import registry
from text2qdmr.utils import saver as saver_mod
from text2qdmr.model.modules import decoder_utils
from text2qdmr.utils.serialization import ComplexEncoder


class Inferer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(1)

        # 0. Construct preprocessors
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])
        self.model_preproc.load()

    def load_model(self, logdir, step):
        '''Load a model (identified by the config used for construction) and return it'''
        # 1. Construct model
        model = registry.construct('model', self.config['model'], preproc=self.model_preproc, device=self.device)
        model.to(self.device)
        model.eval()

        # 2. Restore its parameters
        saver = saver_mod.Saver({"model": model})
        last_step = saver.restore(logdir, step=step, map_location=self.device, item_keys=["model"])
        if not last_step:
            raise Exception(f"Attempting to infer on untrained model in {logdir}, step={step}")
        return model

    def infer(self, model, output_path, args):
        output = open(output_path, 'w')
        
        with torch.no_grad():
            for section in args.section:
                if self.config.get('full_data') is not None and args.part != 'spider':
                    orig_data = registry.construct('dataset', self.config['full_data'][section])
                else:
                    orig_data = args.data[section]
                preproc_data = self.model_preproc.dataset(section, two_datasets=self.config.get('full_data'))
                preproc_data.part = args.part

                if args.shuffle:
                    idx_shuffle = list(range(len(orig_data)))
                    random.shuffle(idx_shuffle)
                    if args.limit:
                        idx_shuffle = idx_shuffle[:args.limit]
                    sliced_orig_data, sliced_preproc_data = [], []
                    for i, (orig_item, preproc_item) in enumerate(zip(orig_data, preproc_data)):
                        if i in idx_shuffle:
                            sliced_orig_data.append(orig_item)
                            sliced_preproc_data.append(preproc_item)
                else:
                    if args.limit:
                        sliced_orig_data = list(itertools.islice(orig_data, args.limit))
                        sliced_preproc_data = list(itertools.islice(preproc_data, args.limit))
                    else:
                        sliced_orig_data = orig_data
                        sliced_preproc_data = preproc_data
                self._inner_infer(model, args.beam_size, args.output_history, sliced_orig_data, sliced_preproc_data,
                                    output, args.strict_decoding, section)

    def _inner_infer(self, model, beam_size, output_history, sliced_orig_data, sliced_preproc_data, output, \
                    strict_decoding=False, section='val'):
        for orig_item, preproc_item in tqdm.tqdm(zip(sliced_orig_data, sliced_preproc_data), total=len(sliced_orig_data)):
            assert orig_item.full_name == preproc_item[0]['full_name'], (orig_item.full_name, preproc_item[0]['full_name'])
            decoded = self._infer_one(model, orig_item, preproc_item, beam_size, output_history, strict_decoding, section)
            output.write(
                json.dumps({
                    'name': orig_item.full_name,
                    'part': section,
                    'beams': decoded,
                }, cls=ComplexEncoder) + '\n')
            output.flush()

    def init_decoder_infer(self, model, data_item, section, strict_decoding):
        # schema
        model.decoder.schema = data_item.schema
        # grounding choices
        _, validation_info = model.preproc.validate_item(data_item, section)
        model.decoder.value_unit_dict = validation_info[0]
        model.decoder.ids_to_grounding_choices = model.decoder.preproc.grammar.get_ids_to_grounding_choices(data_item.schema, validation_info[0])

        for rule, idx in model.decoder.rules_index.items():
            if rule[1] == 'NextStepSelect':
                model.decoder.select_index = idx

        if strict_decoding:
            # column types
            assert len(data_item.column_data) == 1
            model.decoder.column_data = data_item.column_data[0]

            # info about value set
            model.decoder.no_vals = len(model.decoder.value_unit_dict) == 0

            model.decoder.required_column = False
            if not model.decoder.no_vals:
                model.decoder.no_column, model.decoder.required_column = True, True
                for val_units in model.decoder.value_unit_dict.values():
                    model.decoder.required_column = model.decoder.required_column and all(val_unit.column for val_unit in val_units)

            model.decoder.value_columns = set()
            model.decoder.val_types_wo_cols = set()
            for grnd_choice in model.decoder.ids_to_grounding_choices.values():
                if grnd_choice.choice_type == 'value':
                    for val_unit in grnd_choice.choice:
                        if val_unit.column:
                            model.decoder.value_columns.add((val_unit.table, val_unit.column))
                        else:
                            model.decoder.val_types_wo_cols.add(val_unit.value_type)
            for table in model.decoder.column_data.keys():
                for column, col_type in model.decoder.column_data[table].items():
                    if col_type in model.decoder.val_types_wo_cols:
                        model.decoder.value_columns.add((table, column))

            model.decoder.no_column = len(model.decoder.value_columns) == 0
        else:
            model.decoder.column_data = None

        return model

    def _infer_one(self, model, data_item, preproc_item, beam_size, output_history=False, strict_decoding=False, section='val'):
        model = self.init_decoder_infer(model, data_item, section, strict_decoding)
        
        beams = decoder_utils.beam_search(
                model, preproc_item, beam_size=beam_size, max_steps=1000, strict_decoding=strict_decoding)
        decoded = []

        for beam in beams:
            model_output, inferred_code = beam.inference_state.finalize()

            decoded.append({
                'orig_question': data_item.text,
                'model_output': model_output,
                'inferred_code': inferred_code,
                'score': beam.score,
                **({
                       'choice_history': beam.choice_history,
                       'score_history': beam.score_history,
                   } if output_history else {})})
        return decoded


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')

    parser.add_argument('--step', type=int)
    parser.add_argument('--section', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--beam-size', required=True, type=int)
    parser.add_argument('--output-history', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--mode', default='infer', choices=['infer', 'debug'])
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])

    output_path = args.output.replace('__LOGDIR__', args.logdir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f'Output file {output_path} already exists')
        sys.exit(1)

    inferer = Inferer(config)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)


if __name__ == '__main__':
    args = add_parser()
    main(args)
