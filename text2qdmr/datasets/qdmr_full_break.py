import json
import re
import torch
from tqdm import tqdm
from text2qdmr.utils import registry
from text2qdmr.datasets.utils.spider_schema import load_tables
from text2qdmr.datasets.qdmr import BreakItem, create_qdmr_ast
from text2qdmr.datasets.qdmr import NotColumnGrndError
from text2qdmr.datasets.utils.extract_values import ValueExtractor

from qdmr2sparql.datasets import DatasetBreak
from qdmr2sparql.structures import load_grounding_list_from_file


@registry.register('dataset', 'qdmr-full')
class BreakFullDataset(torch.utils.data.Dataset):
    def __init__(self, paths, extract_value={}, exclude_names=None, partition=''):
        self.spider_path = paths['spider_path']
        self.tables_path = paths['tables_path']
        self.db_path = paths['db_path']      
        self.break_logic_form_path = paths['break_logic_form_path']
        self.grounding_path = paths['grounding_path']
        self.examples = []
        self.examples_with_name = {}
        self.exclude_names = exclude_names
        self.extract_value = extract_value
        self.partition = partition

    def load_data(self):
        # load break data    
        break_data = DatasetBreak(self.break_logic_form_path, target_file=self.break_logic_form_path, filter_subset='')

        # load grounding
        grounding_list = load_grounding_list_from_file(self.grounding_path)

        # load schemas
        self.schemas, self.eval_foreign_key_maps = load_tables(self.tables_path)

        # load spider data
        spider_data = json.load(open(self.spider_path))
        value_extractor = ValueExtractor(self.schemas, self.extract_value, self.partition)

        without_grounding = set()
        print('Load additional QDMRs')
        print()
        for i, (name, qdmr_entry) in enumerate(tqdm(break_data.qdmrs.items())):
            groundings = grounding_list.get(name)
            
            # get name and idx
            subset_name = name.split('_')[0]
            digits_in_name = ''.join(name.split('_')[1:])
            digits_in_name = re.findall(r"\d+.*", digits_in_name)  
            assert len(digits_in_name) == 1, digits_in_name
            subset_idx = digits_in_name[0]
            
            if subset_name in self.exclude_names:
                continue

            if subset_name == 'SPIDER':
                subset_idx = int(subset_idx)
                spider_entry = spider_data[subset_idx]
                db_id = spider_entry['db_id']
            else:
                spider_entry, db_id = None, None
            question = break_data.qdmr_full_table[i][1]

            if groundings is None:
                without_grounding.add(name)
                continue

            all_groundings = groundings['GROUNDINGS']   
            all_ast, all_values, all_column_data = [], [], []   
            count_valid_groundings = 0
            for grounding in all_groundings:
                try:  
                    ast = create_qdmr_ast(qdmr_entry, grounding)
                    values, column_data, question_tokens = value_extractor.get_values(qdmr_entry, grounding, db_id, question, i)
                    count_valid_groundings += 1
                    all_ast.append(ast)
                    all_values.append(values)
                    all_column_data.append(column_data)
                    break  
                except NotColumnGrndError:
                    continue
                except Exception as e:
                    print(name, 'exception', e)
                    continue

            if count_valid_groundings > 0:
                item = BreakItem(
                        subset_idx=subset_idx,
                        text=question,
                        qdmr_code=all_ast,
                        qdmr_ops=qdmr_entry.ops, 
                        qdmr_args=qdmr_entry.args,
                        grounding=all_groundings,
                        values=all_values,
                        column_data=all_column_data,
                        text_toks=spider_entry['question_toks'] if spider_entry is not None else None,
                        text_toks_for_val=question_tokens,
                        sql_code=spider_entry['sql'] if spider_entry is not None else None,
                        schema=self.schemas[db_id] if spider_entry is not None else None,
                        orig_spider_entry=spider_entry,
                        orig_schema=self.schemas[db_id].orig if spider_entry is not None else None,
                        subset_name=subset_name,
                        full_name=name,
                        db_id=db_id if spider_entry is not None else None,
                        )
                self.examples.append(item)
                assert name not in self.examples_with_name, name
                self.examples_with_name[name] = item

        print('original data', len(break_data))
        print('saved data', len(self.examples))
        print('data without grounding', len(without_grounding))
            
    def item_from_name(self, name):
        return self.examples_with_name.get(name)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __setitem__(self, idx, example):
        self.examples[idx] = example
