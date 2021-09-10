import argparse
import json

import _jsonnet
from pyrsistent import v
import tqdm

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from text2qdmr import datasets
# noinspection PyUnresolvedReferences
from text2qdmr import model
# noinspection PyUnresolvedReferences
from text2qdmr.utils import registry
# noinspection PyUnresolvedReferences
from text2qdmr.utils import vocab


class Preprocessor:
    def __init__(self, config, partition=None):
        self.config = config
        self.partition = partition
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])

    def preprocess(self):
        self.model_preproc.clear_items()
        
        for section in ('train', 'val', 'test'):
            if self.partition and self.partition != section:
                continue

            data = registry.construct('dataset', self.config['data'][section])
            data.load_data()

            count_add = 0
            for i, item in tqdm.tqdm(enumerate(data), desc=f"{section} section", dynamic_ncols=True):
                to_add, validation_info = self.model_preproc.validate_item(item, section)
                if to_add.any():
                    self.model_preproc.add_item(item, section, to_add, validation_info)
                    count_add += 1

        print('break data', i)
        len_saved_data = {}
        for part in self.model_preproc.dec_preproc.items.keys():
            len_saved_data[part] = len(self.model_preproc.dec_preproc.items[part])
            print('saved {} data: {}'.format(part, len_saved_data[part]))

        if self.config.get('full_data') is not None:
            for section in self.config['full_data']:
                if self.partition and self.partition != section:
                    continue
                data = registry.construct('dataset', self.config['full_data'][section])
                data.load_data()

                count_add = 0
                for i, item in tqdm.tqdm(enumerate(data), desc=f"{section} section", dynamic_ncols=True):
                    to_add, validation_info = self.model_preproc.validate_item(item, section)
                    if to_add.any():
                        self.model_preproc.add_item(item, section, to_add, validation_info)
                        count_add += 1

            print('full break data', i)   
            for part in self.model_preproc.dec_preproc.items.keys():
                len_data = len(self.model_preproc.dec_preproc.items[part]) - len_saved_data[part]
                print('saved {} data: {}'.format(part, len_data))
        
        print()
        for part in self.model_preproc.dec_preproc.items.keys():
            print('saved {} data: {}'.format(part, len(self.model_preproc.dec_preproc.items[part])))
        self.model_preproc.save(partition=self.partition)


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args


def main(args, partition=None):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    preprocessor = Preprocessor(config, partition=partition)
    preprocessor.preprocess()


if __name__ == '__main__':
    args = add_parser()
    main(args)
