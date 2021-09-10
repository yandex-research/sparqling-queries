import json
import attr
import sqlite3
import torch
from tqdm import tqdm
from pathlib import Path

from text2qdmr.utils import registry
from text2qdmr.datasets.utils.spider_schema import load_tables
from text2qdmr.datasets.utils.extract_values import ValueExtractor, ValueUnit

from qdmr2sparql.structures import GroundingIndex, load_grounding_list_from_file
from qdmr2sparql.datasets import DatasetBreak


AGG_OPS = ('max', 'min', 'count', 'sum', 'avg')
ARITH_OPS = ('division', 'multiplication', 'difference')
ORDER_OPS = ('ascending', 'descending')
DEFAULT_AGG_TYPE = 'sum'

requires_distinct = lambda op: op == 'select' or op == 'project' or op == 'comparative'

class NotColumnGrndError(Exception):
    pass

@attr.s
class BreakItem:
    subset_idx = attr.ib()
    text = attr.ib()
    text_toks = attr.ib()
    text_toks_for_val = attr.ib()
    qdmr_code = attr.ib()
    qdmr_ops = attr.ib()
    qdmr_args = attr.ib()
    grounding = attr.ib()
    values = attr.ib()
    column_data = attr.ib()

    sql_code = attr.ib(default=None)
    schema = attr.ib(default=None)
    eval_graphs = attr.ib(default=None) # for computing metrics
    orig_schema = attr.ib(default=None)
    orig_spider_entry = attr.ib(default=None)
    db_id = attr.ib(default=None)

    subset_name = attr.ib(default='SPIDER')
    full_name = attr.ib(default=None)

@attr.s(frozen=True)
class QDMRStepArg:
    arg_type = attr.ib()
    arg = attr.ib()

def transform_superlative(op, cur_grounding, args_list):
    if op == 'comparative':
        assert len(args_list) == 2
        args_list.insert(0, QDMRStepArg('grounding', cur_grounding.keys[0]))
    elif op == 'filter':
        assert len(args_list) == 1
        args_list *= 2
        args_list.insert(0, QDMRStepArg('grounding', cur_grounding.keys[0])) 
    else:
        raise Exception(f"Only comparative and filter can be transformed to superlative, got {op}")
    return 'superlative', args_list

def merge_filter_comparative(op, args_list):
    assert op == 'filter', 'Only filter can be merged with comparative, got {}'.foramt(op)
    assert len(args_list) == 2
    # swap and double first arg
    args_list = 2 * args_list[:1] +  args_list[1:]
    return 'comparative', args_list


def create_qdmr_ast(qdmr_entry, grounding):
    ast = []
    distinct_steps = grounding.get('distinct')
    if distinct_steps:
        distinct_steps = [qdmr_entry.ref_to_index(step) for step in distinct_steps]
    else:
        distinct_steps = []
    fl = False
    for i, (op, all_args) in enumerate(zip(qdmr_entry.ops, qdmr_entry.args)):
        op = op.lower()
        # [QDMRStepArg(), ...]
        args_list = [] 

        assert op != 'union' or op == 'union' and len(all_args) <= 4, 'Union with {} args'.format(len(all_args))
        assert op != 'arithmetic' or op == 'arithmetic' and len(all_args) <= 3, 'Arithmetic with {} args'.format(len(all_args))

        for i_arg, arg in enumerate(all_args):
            refs = qdmr_entry.find_qdmr_refs_in_str(arg)
            assert len(refs) <= 1, 'too many refs {} in {}'.format( refs, arg)

            if arg in AGG_OPS or arg in ARITH_OPS: 
                cur_grounding = grounding.get(GroundingIndex(i, i_arg, arg))
                if not cur_grounding or op == 'superlative':
                    cur_grounding = arg
                elif not (op == 'aggregate' or op == 'group') or not cur_grounding.iscol():
                    raise NotColumnGrndError
            else:
                cur_grounding = grounding.get(GroundingIndex(i, i_arg, arg))
                if cur_grounding:
                    # transform comparative with sup_op to superlative
                    if cur_grounding.iscomp() and cur_grounding.keys[0] in ('max', 'min', 'min/max'):
                        op, args_list = transform_superlative(op, cur_grounding, args_list)
                        break
                    if cur_grounding.type == 'str' and not cur_grounding.keys[0]:
                        cur_grounding = None

            if refs and cur_grounding:
                assert op in ('comparative', 'filter') and cur_grounding.iscomp(), \
                    'ref and grounding can be only in comparative with ref arg, got {} op and {}, {}'.format(op, refs, cur_grounding)
                assert cur_grounding.keys[1] == refs[0], \
                    'ref arg and comparative key should be equal: got {} op and {}, {}'.format(op, refs, cur_grounding)
                args_list.append(QDMRStepArg('grounding', cur_grounding))
            elif refs:
                assert not cur_grounding, 'ref and grounding in one arg: {}, {}'.format(refs, cur_grounding)
                args_list.append(QDMRStepArg('ref', refs))
            elif cur_grounding:
                assert not refs, 'ref and grounding in one arg: {}, {}'.format(refs, cur_grounding)
                args_list.append(QDMRStepArg('grounding', cur_grounding))
            else:
                # None arg can be in project and comparative
                assert op == 'project' and i_arg == 0 or op == 'comparative' and i_arg == 2 \
                    or op == 'filter' and i_arg == 1, 'None arg can be in project and comparative, got {}, index of arg: {}'.format(op, i_arg)
                args_list.append(QDMRStepArg('grounding', None))
                

        # merge filter and comparative to new op 'comparative'
        if op == 'filter':
            op, args_list = merge_filter_comparative(op, args_list)

        if requires_distinct(op):
            is_distinct = i in distinct_steps
            # [operator, list of args, is_distinct]
            ast.append([op, args_list, is_distinct])
        else:
            # (operator, list of args)
            ast.append([op, args_list])
    return ast

@registry.register('dataset', 'qdmr')
class BreakDataset(torch.utils.data.Dataset):
    def __init__(self, paths, extract_value={}, partition=''):
        self.spider_path = paths['spider_path']
        self.tables_path = paths['tables_path']
        self.db_path = paths['db_path']      
        self.break_logic_form_path = paths['break_logic_form_path']
        self.grounding_path = paths['grounding_path']
        self.extracted_value_path = paths.get('extracted_value_path')
        self.examples = []
        self.examples_with_name = {}
        self.schemas = None
        self.extract_value = extract_value
        self.partition = partition

        # load grounding
        if self.grounding_path:
            self.grounding_list = load_grounding_list_from_file(self.grounding_path)
        else:
            self.grounding_list = {}

    def load_data(self):
        # load spider data
        spider_data = json.load(open(self.spider_path))

        # load break data    
        break_data = DatasetBreak(self.break_logic_form_path, target_file=self.break_logic_form_path)

        # load schemas
        self.schemas, self.eval_foreign_key_maps = load_tables(self.tables_path)

        # Backup in-memory copies of all the DBs and create the live connections
        for db_id, schema in tqdm(self.schemas.items(), desc="DB connections"):
            sqlite_path = Path(self.db_path) / db_id / f"{db_id}.sqlite"
            source: sqlite3.Connection
            with sqlite3.connect(str(sqlite_path)) as source:
                dest = sqlite3.connect(':memory:')
                dest.row_factory = sqlite3.Row
                dest.text_factory = lambda b: b.decode(errors = 'ignore')
                source.backup(dest)
            schema.connection = dest

        if not self.extracted_value_path:
            value_extractor = ValueExtractor(self.schemas, self.extract_value, self.partition)
        else:
            extracted_values = json.load(open(self.extracted_value_path, 'r'))
        
        without_grounding = set()
        # get name and partition
        subset_name = break_data.get_dataset_keyword_from_name(break_data.names[0])
        assert subset_name == 'SPIDER'
        break_partition = self.partition if self.partition != 'test' else 'dev'

        print('Load dataset, {} part'.format(self.partition))
        print()
        for subset_idx, spider_entry in enumerate(tqdm(spider_data)):
            db_id = spider_entry['db_id']

            name = break_data.get_item_name(subset_name, break_partition, subset_idx)
            if name in break_data.names:
                i = break_data.names.index(name)
                question = break_data.qdmr_full_table[i][1]
                if spider_entry['question'] != question:
                    print('Different text in {}, chose SPIDER'.format(name))
                    print('SPIDER: ', spider_entry['question'])
                    print('QDMR: ', question)
                    question = spider_entry['question']
            else:
                continue
            qdmr_entry = break_data.qdmrs.get(name)
            groundings = self.grounding_list.get(name)

            if name in self.grounding_list:                
                all_groundings = groundings['GROUNDINGS']
                all_ast, all_values, all_column_data = [], [], []   
                count_valid_groundings = 0
                if  self.extracted_value_path and name not in extracted_values:
                    print('{} not in extracted_values'.format(name))
                    continue
                for grounding in all_groundings:
                    try:  
                        ast = create_qdmr_ast(qdmr_entry, grounding)
                        if not self.extracted_value_path:
                            values, column_data, question_tokens = value_extractor.get_values(qdmr_entry, grounding, db_id, question, i)
                        else:
                            column_data = extracted_values[name]['column_data'][0]
                            question_tokens = extracted_values[name]['text_toks_for_val']
                            values = [ValueUnit(**val) for val in extracted_values[name]['values'][0]]

                        count_valid_groundings += 1
                        all_ast.append(ast)
                        all_values.append(values)
                        all_column_data.append(column_data)
                        break      
                    except NotColumnGrndError:
                        continue   
                    except Exception as e:
                        raise e
            elif self.partition == 'test':
                without_grounding.add(name)
                if not self.extracted_value_path:
                    values, column_data, question_tokens = value_extractor.get_values(qdmr_entry, None, db_id, question, i)
                else:
                    column_data = extracted_values[name]['column_data'][0]
                    question_tokens = extracted_values[name]['text_toks_for_val']
                    values = [ValueUnit(**val) for val in extracted_values[name]['values'][0]]
                all_values, all_column_data = [values], [column_data]
                count_valid_groundings = 1
            else:
                without_grounding.add(name)
                continue
            
            if count_valid_groundings > 0:
                item = BreakItem(
                        subset_idx=subset_idx,
                        text=question,
                        qdmr_code=all_ast if qdmr_entry and name in self.grounding_list else None,
                        qdmr_ops=qdmr_entry.ops if qdmr_entry else None, 
                        qdmr_args=qdmr_entry.args if qdmr_entry else None,
                        grounding=all_groundings if name in self.grounding_list else None,
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
                self.examples_with_name[name] = item
            else:
                print('Skipping {}: 0 valid groundings'.format(subset_idx))

        print('break data', len(break_data))
        print('saved data', len(self.examples))
        print('data without grounding', len(without_grounding))

            
    def item_from_name(self, name):
        return self.examples_with_name.get(name)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __del__(self):
        if self.schemas:
            for _, schema in self.schemas.items():
                if schema.connection:
                    schema.connection.close()