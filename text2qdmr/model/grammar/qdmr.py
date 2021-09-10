import collections
import os

import asdl
import attr

from text2qdmr.utils import registry, ast_util
from text2qdmr.datasets.qdmr import QDMRStepArg
from text2qdmr.datasets.utils.extract_values import GroundingKey, ValueUnit, transform_to_type


def bimap(first, second):
    return {f: s for f, s in  zip(first, second)}, {s: f for f, s in zip(first, second)}


def filter_nones(d):
    return {k: v for k, v in d.items() if v is not None and v != []}


AGG_TYPES_F, AGG_TYPES_B = bimap(
    ('count', 'sum', 'avg', 'max', 'min'),
    ('Count', 'Sum', 'Avg', 'MaxAgg', 'MinAgg'))

ORDERS_F, ORDERS_B = bimap(
    ('ascending', 'descending'),
    ('Asc', 'Desc'))

BOOLS_F, BOOLS_B = bimap(
    (True, False),
    ('True', 'False'))

SUP_F, SUP_B = bimap(
    ('max', 'min'),
    ('Max', 'Min'))

COMP_F, COMP_B = bimap(
    ('=', '>', '<', '>=', '<=', '!=', 'like'),
    ('Eq', 'Gt', 'Lt', 'Ge', 'Le', 'Ne','Like')
)

UNIT_F, UNIT_B = bimap(
    ('difference', 'sum', 'multiplication', 'division'),
    ('Minus', 'Plus', 'Times', 'Divide')
)


@attr.s(frozen=True)
class GroundingChoice:
    choice_type = attr.ib()
    choice = attr.ib()

@registry.register('grammar', 'qdmr')
class QdmrLanguage:

    root_type = 'root'

    def __init__(self, version=1, val_pointer=False):
        # collect pointers and checkers
        custom_primitive_type_checkers = {}
        self.ref = set()
        self.general_pointers = set()
        assert version == 1

        custom_primitive_type_checkers['ref'] = lambda x: isinstance(x, int)
        self.ref.add('ref')

        custom_primitive_type_checkers['grounding'] = lambda x: isinstance(x, int)
        self.general_pointers.add('grounding')
        
        ast_file = 'Qdmr.asdl'

        self.ast_wrapper = ast_util.ASTWrapper(
            asdl.parse(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    ast_file)),
            custom_primitive_type_checkers=custom_primitive_type_checkers)

    def schema_to_ids(self, schema):
        self.tab_to_id = {tab.orig_name: tab.id for tab in schema.tables}
        assert len(self.tab_to_id) == len(schema.tables) 
        self.col_to_id = {(col.table.orig_name, col.orig_name): col.id for col in schema.columns if col.table}
        assert len(self.col_to_id) == len(schema.columns) - 1 # do not count '*' 

    def ids_to_schema(self, schema):
        id_to_tab = {tab.id: tab.orig_name for tab in schema.tables}
        id_to_col = {0: (None, None)}
        for col in schema.columns:
            if col.table:
                id_to_col[col.id] = (col.table.orig_name, col.orig_name) 
        assert len(id_to_tab) == len(schema.tables)
        return id_to_tab, id_to_col

    def get_grounding_choices_to_ids(self, schema, value_unit_dict):
        if schema is not None:
            self.schema_to_ids(schema)
        self.val_to_id = {vals: val_unit[0].idx for vals, val_unit in value_unit_dict.items()}
        if schema is not None:
            all_dicts = {'table': self.tab_to_id, 'column': self.col_to_id, 'value': self.val_to_id}
        else:
            all_dicts = {'value': self.val_to_id}

        self.grounding_choices_to_ids = collections.OrderedDict()
        i = 0
        for dict_type, el_dict in all_dicts.items():
            for name, idx in el_dict.items():
                assert isinstance(idx, int)
                self.grounding_choices_to_ids[GroundingChoice(choice_type=dict_type, choice=name)] = i
                i += 1
        return self.grounding_choices_to_ids

    def get_ids_to_grounding_choices(self, schema, value_unit_dict):
        if schema is not None:
            id_to_tab, id_to_col = self.ids_to_schema(schema)
        id_to_val = {val_unit[0].idx: val_unit for vals, val_unit in value_unit_dict.items()}
        
        ids_to_grounding_choices = collections.OrderedDict()
        if schema is not None:
            all_dicts = {'table': id_to_tab, 'column': id_to_col, 'value': id_to_val}
        else:
            all_dicts = {'value': id_to_val}
        i = 0
        for dict_type, el_dict in all_dicts.items():
            for idx, name in el_dict.items():
                assert isinstance(idx, int)
                if name == (None, None):
                    continue
                ids_to_grounding_choices[i] = GroundingChoice(choice_type=dict_type, choice=name)
                i += 1
        return ids_to_grounding_choices

    def check_ref_arg(self, op, arg, idx_arg):
        assert arg.arg_type == 'ref', '{} {} arg should be ref, not grounding {}'.format(op, idx_arg + 1, arg.arg)

    def check_grounding_arg(self, op, arg, idx_arg):
        assert arg.arg_type == 'grounding', '{} {} arg should be grounding, not ref {}'.format(op, idx_arg + 1, arg.arg)

    def check_num_args_list(self, op, args_list, num_args):
        assert len(args_list) == num_args, '{} should be with {} args, got {}'.format(op, num_args, len(args_list))

    def parse(self, code, schema, value_unit_dict, column_data, section):
        self.from_qdmr = any([value_unit.source == 'qdmr'
                            for value_units in value_unit_dict.values() 
                            for value_unit in value_units])
        self.get_grounding_choices_to_ids(schema, value_unit_dict)
        self.column_data = column_data

        return filter_nones({
                '_type': 'root',
                'step': self.parse_step(code)
            })

    def unparse(self, tree, schema, value_unit_dict, column_data):
        ids_to_grounding_choices = self.get_ids_to_grounding_choices(schema, value_unit_dict)
        id_to_tab, id_to_col, value_unit_dict = None, None, None
        unparser = self.QDMRUnparser(id_to_tab, id_to_col, value_unit_dict, \
                    ids_to_grounding_choices=ids_to_grounding_choices, column_data=column_data)
        return unparser.unparse(tree)

    def parse_step(self, code):
        if len(code) == 0:
            return {'_type': 'FinalStep'}

        step_type = code[0][0]

        if step_type == 'select':
            return filter_nones({
                '_type': 'NextStepSelect',
                'select': self.parse_select(code)
            })
        elif step_type == 'aggregate':
            return filter_nones({
                '_type': 'NextStepAgg',
                'aggregate': self.parse_aggregate(code)
            })
        elif step_type == 'project':
            return filter_nones({
                '_type': 'NextStepProject',
                'project': self.parse_project(code)
            })
        elif step_type == 'union':
            return filter_nones({
                '_type': 'NextStepUnion',
                'union': self.parse_union(code)
            })
        elif step_type == 'comparative':
            return filter_nones({
                '_type': 'NextStepComp',
                'comparative': self.parse_comparative(code)
            })
        elif step_type == 'superlative':
            return filter_nones({
                '_type': 'NextStepSup',
                'superlative': self.parse_superlative(code)
            })
        elif step_type == 'intersection':
            return filter_nones({
                '_type': 'NextStepIntersect',
                'intersection': self.parse_intersect(code)
            })
        elif step_type == 'discard':
            return filter_nones({
                '_type': 'NextStepDiscard',
                'discard': self.parse_discard(code)
            })
        elif step_type == 'sort':
            return filter_nones({
                '_type': 'NextStepSort',
                'sort': self.parse_sort(code)
            })
        elif step_type == 'group':
            return filter_nones({
                '_type': 'NextStepGroup',
                'group': self.parse_group(code)
            })
        elif step_type == 'arithmetic':
            return filter_nones({
                '_type': 'NextStepArithmetic',
                'arithmetic': self.parse_arithmetic(code)
            })
        else:
            raise ValueError(step_type)

    def parse_select(self, code):
        op, args_list, is_distinct = code[0]
        self.check_num_args_list(op, args_list, 1)
        self.check_grounding_arg(op, args_list[0], 0)

        return filter_nones({
            '_type': 'select',
            'is_distinct': {'_type': BOOLS_F[is_distinct]},
            'grounding': self.parse_grounding(args_list[0].arg),
            'step': self.parse_step(code[1:])
        })

    def parse_intersect(self, code):
        op, args_list = code[0]
        assert 1 < len(args_list) < 4, '{} should be with 2 or 3 args, got {}'.format(op, len(args_list))
        for i in range(len(args_list)):
            self.check_ref_arg(op, args_list[i], i)

        return filter_nones({
            '_type': 'intersection',
            'ref1': self.parse_ref(args_list[0].arg),
            'ref2': self.parse_ref(args_list[1].arg),
            'union_3arg': self.parse_union_3arg(args_list, 2),
            'step': self.parse_step(code[1:])
        })

    def parse_arithmetic(self, code):
        op, args_list = code[0]
        self.check_num_args_list(op, args_list, 3)
        self.check_grounding_arg(op, args_list[0], 0)
        self.check_ref_arg(op, args_list[1], 1)
        self.check_ref_arg(op, args_list[2], 2)

        return filter_nones({
            '_type': 'arithmetic',
            'unit_ops': {
                '_type': UNIT_F[args_list[0].arg]
            },
            'ref1': self.parse_ref(args_list[1].arg),
            'ref2': self.parse_ref(args_list[2].arg),
            'step': self.parse_step(code[1:])
        })

    def parse_discard(self, code):
        op, args_list = code[0]
        self.check_num_args_list(op, args_list, 2)
        self.check_ref_arg(op, args_list[0], 0)
        self.check_ref_arg(op, args_list[1], 1)

        return filter_nones({
            '_type': 'discard',
            'ref1': self.parse_ref(args_list[0].arg),
            'ref2': self.parse_ref(args_list[1].arg),
            'step': self.parse_step(code[1:])
        })

    def parse_aggregate(self, code):
        op, args_list = code[0]
        self.check_num_args_list(op, args_list, 2)
        self.check_grounding_arg(op, args_list[0], 0)
        self.check_ref_arg(op, args_list[1], 1)

        agg_type = AGG_TYPES_F.get(args_list[0].arg)
        return filter_nones({
            '_type': 'aggregate',
            'agg_type': {
                    '_type': 'AggOp',
                            'agg_ops': {'_type': agg_type} 
                }   if agg_type 
                    else self.parse_compute_choice(args_list[0].arg),
            'ref': self.parse_ref(args_list[1].arg),
            'step': self.parse_step(code[1:])
        })

    def parse_compute_choice(self, compute_choice_arg):
        assert compute_choice_arg.iscol()
        return {
                '_type': 'UseColumn',
                'grounding': self.parse_grounding(compute_choice_arg),
                } 
        
    def parse_sort(self, code):
        op, args_list = code[0]
        assert 1 < len(args_list) < 4, '{} should be with 2 or 3 args, got {}'.format(op, len(args_list))
        self.check_ref_arg(op, args_list[0], 0)
        self.check_ref_arg(op, args_list[1], 1)

        if len(args_list) == 3:
            self.check_grounding_arg(op, args_list[2], 2)
            sort_order = args_list[2].arg.keys[0]
        else:
            sort_order = 'ascending'
                    
        return filter_nones({
            '_type': 'sort',
            'ref1': self.parse_ref(args_list[0].arg),
            'ref2': self.parse_ref(args_list[1].arg),
            'order': {'_type': ORDERS_F[sort_order]},
            'step': self.parse_step(code[1:])
        })

    
    def parse_superlative(self, code):
        op, args_list = code[0]
        self.check_num_args_list(op, args_list, 3)
        self.check_grounding_arg(op, args_list[0], 0)
        self.check_ref_arg(op, args_list[1], 1)
        self.check_ref_arg(op, args_list[2], 2)

        sup_op = args_list[0].arg
        return filter_nones({
            '_type': 'superlative',
            'superlative_op_type': {'_type': 'SupOp',
                                    'superlative_ops': {'_type': SUP_F[sup_op]} 
                                    } if sup_op in SUP_F.keys()
                                    else {'_type': 'SupUnknownOp'},
            'ref1': self.parse_ref(args_list[1].arg),
            'ref2': self.parse_ref(args_list[2].arg),
            'step': self.parse_step(code[1:])
        })

    def parse_project(self, code, is_distinct=False):
        op, args_list, is_distinct = code[0]
        self.check_num_args_list(op, args_list, 2)
        self.check_grounding_arg(op, args_list[0], 0)
        self.check_ref_arg(op, args_list[1], 1)

        return {
            '_type': 'project',
            'is_distinct': {'_type': BOOLS_F[is_distinct]},
            'project_1arg': self.parse_project_1arg(args_list[0].arg, is_distinct),
            'ref': self.parse_ref(args_list[1].arg),
            'step': self.parse_step(code[1:])
        }

    def parse_project_1arg(self, project_1arg, is_distinct):        
        if project_1arg is None:
            return {
                '_type': 'NoneProjectArg'
            }
        else:
            return {
                '_type': 'GroundingProjectArg',
                'grounding': self.parse_grounding(project_1arg)
            }

    def parse_comparative(self, code):
        op, args_list, is_distinct = code[0]
        self.check_num_args_list(op, args_list, 3)
        self.check_ref_arg(op, args_list[0], 0)
        self.check_ref_arg(op, args_list[1], 1)
        self.check_grounding_arg(op, args_list[2], 2)
        return {
            '_type': 'comparative',
            'is_distinct': {'_type': BOOLS_F[is_distinct]},
            'ref1': self.parse_ref(args_list[0].arg),
            'ref2': self.parse_ref(args_list[1].arg),
            'comparative_3arg_type': {'_type': 'NoneCompArg'} if args_list[2].arg is None \
                    else self.parse_comparative_3arg(args_list[2].arg),
            'step': self.parse_step(code[1:])
        }

    def parse_comparative_3arg(self, comparative_3arg):
        return {
            '_type': 'CompArg',
            'comp_op_type': self.parse_comp_op_type(comparative_3arg),
            'column_type': self.parse_column_type(comparative_3arg),
            'comp_val': self.parse_comp_val(comparative_3arg),
        }

    def parse_comp_op_type(self, comparative_3arg):
        if comparative_3arg.iscomp():
            comparative_op = comparative_3arg.keys[0]
            comparative_op = COMP_F.get(comparative_op)
            if comparative_op == 'Eq':
                return {'_type': 'NoOp'}
            elif comparative_op is not None:
                return {'_type': 'CompOp',
                    'comparative_ops': {'_type': comparative_op} 
                }
            elif comparative_3arg.keys[0] == 'UNK':
                return {'_type': 'UnknownOp'}
        elif comparative_3arg.istbl() or comparative_3arg.iscol() or comparative_3arg.isval() \
            or comparative_3arg.type == 'str':
            return {'_type': 'NoOp'}
    
    def parse_column_type(self, comparative_3arg):
        if comparative_3arg.iscomp():
            num_comp_args = len(comparative_3arg.keys)
            if num_comp_args == 2:
                return {'_type': 'NoColumnGrounding'} 
            else:
                # TODO not all comp2 are without columns, 
                # we can detect column grnd from qdmr
                col_gk = comparative_3arg.keys[-1]
                assert col_gk.iscol()
                return {'_type': 'ColumnGrounding',
                    'grounding': self.parse_grounding(col_gk)
                }
        elif comparative_3arg.isval():
            tbl_name, col_name, _ = comparative_3arg.keys
            col_gk = GroundingKey.make_column_grounding(tbl_name, col_name)
            return {'_type': 'ColumnGrounding',
                    'grounding': self.parse_grounding(col_gk)
            }
        elif comparative_3arg.istbl() or comparative_3arg.iscol():
            return {'_type': 'NoColumnGrounding'} 
        elif comparative_3arg.type == 'str':
            return {'_type': 'UnknownColumnGrounding'}
        
    def parse_comp_val(self, comparative_3arg):
        # TODO text grounding
        if comparative_3arg.iscomp():
            comp_val = comparative_3arg.keys[1]
            if comp_val[0] == '#':
                # compare with ref
                return {
                    '_type': 'CompRef',
                    'ref': self.parse_ref([comp_val])
                }
        
        return {
                    '_type': 'CompGrounding',
                    'grounding': self.parse_grounding(comparative_3arg)
        }

    def parse_comparative_arg(self, comparative_arg):
        comp_val = comparative_arg.keys[1]
        num_comp_args = len(comparative_arg.keys)

        if comp_val[0] == '#':
            # compare with ref
            return {
                '_type': 'CompRef',
                'ref': self.parse_ref([comp_val])
            }
        elif num_comp_args == 3:
            col_gk = comparative_arg.keys[-1]
            assert col_gk.iscol()
            tbl_name, col_name = col_gk.get_tbl_name(), col_gk.get_col_name()
            return {
                '_type': 'CompValFull',
                'val_unit': self.parse_val_unit(comp_val, col_name, tbl_name)
            }
        elif num_comp_args == 2:
            return {
                '_type': 'CompVal',
                'val_id': self.get_val_id(comp_val)
            }

    def get_val_id(self, orig_value, tbl_col_key=None, val_type=None):
        assert type(orig_value) == str

        # get column type
        if val_type is not None:
            assert val_type == 'text' 
            tbl_name, col_name = None, None
        elif tbl_col_key is None:
            tbl_name, col_name, val_type = None, None, 'unknown'
        else:
            tbl_name, col_name = tbl_col_key
            val_type = self.column_data[tbl_name][col_name]

        # transform to proper type
        if str(orig_value)[0] == '%':
            orig_value = orig_value[1:]
        if str(orig_value)[-1] == '%':
            orig_value = orig_value[:-1]
        value = transform_to_type(orig_value, val_type)
        target_value_unit = ValueUnit(value, orig_value=orig_value, value_type=val_type, column=col_name, table=tbl_name, source='qdmr') 
        str_val = str(target_value_unit)
        idx = None
        for grnd_choice in self.grounding_choices_to_ids:
            if grnd_choice.choice_type != 'value':
                continue
            if str_val in grnd_choice.choice:
                assert idx is None
                idx = self.grounding_choices_to_ids[grnd_choice]

        # TODO chech here
        if idx is None:
            assert not self.from_qdmr, (self.grounding_choices_to_ids, target_value_unit)
            return 0
        return idx

    def parse_group(self, code):
        op, args_list = code[0]
        self.check_num_args_list(op, args_list, 3)
        self.check_grounding_arg(op, args_list[0], 0)
        self.check_ref_arg(op, args_list[1], 1)
        self.check_ref_arg(op, args_list[2], 2)

        agg_type = AGG_TYPES_F.get(args_list[0].arg)
        return {
            '_type': 'group',
            'agg_type': {
                    '_type': 'AggOp',
                            'agg_ops': {'_type': agg_type} 
                }   if agg_type 
                    else self.parse_compute_choice(args_list[0].arg),
            'ref1': self.parse_ref(args_list[1].arg),
            'ref2': self.parse_ref(args_list[2].arg),
            'step': self.parse_step(code[1:])
        }

    def parse_union_3arg(self, args_list, i_arg=2):
        return {
                '_type': 'RefArg',
                'ref': self.parse_ref(args_list[i_arg].arg)
            } if len(args_list) >= i_arg + 1 else \
            {
                '_type': 'NoneUnionArg'
            }

    def parse_union(self, code):
        op, args_list = code[0]
        assert 1 < len(args_list) < 5, '{} should be with 2-4 args, got {}'.format(op, len(args_list))
        for i in range(len(args_list)):
            self.check_ref_arg(op, args_list[i], i)

        return filter_nones({
            '_type': 'union',
            'ref1': self.parse_ref(args_list[0].arg),
            'ref2': self.parse_ref(args_list[1].arg),
            'union_3arg': self.parse_union_3arg(args_list, 2),
            'union_4arg': self.parse_union_3arg(args_list, 3),
            'step': self.parse_step(code[1:])
        })

    def parse_ref(self, ref):
        assert len(ref) == 1, 'too many refs in one arg: {}, {}'.format(ref, len(ref))
        ref = ref[0]
        return int(ref[1:]) - 1
        
    def parse_grounding(self, entity):
        if entity.istbl():
            idx = self.grounding_choices_to_ids[GroundingChoice('table', entity.get_tbl_name())]
        elif entity.iscol():
            tbl_col_key = (entity.get_tbl_name(), entity.get_col_name())
            idx = self.grounding_choices_to_ids[GroundingChoice('column', tbl_col_key)]
        elif entity.isval():
            val, col_name, tbl_name = entity.get_val(), entity.get_col_name(), entity.get_tbl_name()
            idx = self.get_val_id(val, (tbl_name, col_name))
        elif entity.iscomp():
            comp_val = entity.keys[1]
            assert comp_val[0] != '#', entity
            num_comp_args = len(entity.keys)
            
            if num_comp_args == 3:
                col_gk = entity.keys[-1]
                assert col_gk.iscol()
                tbl_name, col_name = col_gk.get_tbl_name(), col_gk.get_col_name()
                tbl_col_key = (tbl_name, col_name)
            elif num_comp_args == 2:
                tbl_col_key = None
            
            idx = self.get_val_id(comp_val, tbl_col_key)
        elif entity.type == 'str':
            idx = self.get_val_id(entity.keys[0], val_type='text')
        else:
            raise ValueError(entity.type)
        return idx


    def parse_val_unit(self, val, col_name, tbl_name):
        tbl_col_key = (tbl_name, col_name)
        return {
            '_type': 'val_unit',
            'col_id': self.col_to_id[tbl_col_key],
            'val_id': self.get_val_id(val, tbl_col_key)
        }

    @attr.s
    class QDMRUnparser:
        id_to_tab = attr.ib()
        id_to_col = attr.ib()
        value_unit_dict = attr.ib()

        ids_to_grounding_choices = attr.ib(default=None)
        column_data = attr.ib(default=None)

        def unparse(self, tree):
            res = self.unparse_step(tree['step'])
            return res
        
        def unparse_step(self, tree):
            step_type = tree['_type']
            is_distinct = None
            
            if step_type == 'NextStepSelect':
                op = 'select'
                args_list, is_distinct = self.unparse_select(tree[op])
            elif step_type == 'NextStepAgg':
                op = 'aggregate'
                args_list = self.unparse_aggregate(tree[op])
            elif step_type == 'NextStepProject':
                op = 'project'
                args_list, is_distinct = self.unparse_project(tree[op])
            elif step_type == 'NextStepUnion':
                op = 'union'
                args_list = self.unparse_union(tree[op])
            elif step_type == 'NextStepComp':
                op = 'comparative'
                args_list, is_distinct = self.unparse_comparative(tree[op])
            elif step_type == 'NextStepSup':
                op = 'superlative'
                args_list = self.unparse_superlative(tree[op])
            elif step_type == 'NextStepIntersect':
                op = 'intersection'
                args_list = self.unparse_intersection(tree[op])
            elif step_type == 'NextStepDiscard':
                op = 'discard'
                args_list = self.unparse_discard(tree[op])
            elif step_type == 'NextStepSort':
                op = 'sort'
                args_list = self.unparse_sort(tree[op])
            elif step_type == 'NextStepGroup':
                op = 'group'
                args_list = self.unparse_group(tree[op])
            elif step_type == 'NextStepArithmetic':
                #return 
                op = 'arithmetic'
                args_list = self.unparse_arithmetic(tree[op])
            elif step_type == 'FinalStep':
                return
            else:
                raise ValueError(step_type)

            if is_distinct is not None:
                assert op in ('select', 'project', 'comparative'), 'distinct in {}'.format(op)
                qdmr_step = [op, args_list, is_distinct]
            else:
                qdmr_step = [op, args_list]

            qdmr = [qdmr_step]
            res = self.unparse_step(tree[op]['step'])
            if res:
                qdmr += res
            return qdmr

        def unparse_select(self, tree):
            is_distinct = BOOLS_B[tree['is_distinct']['_type']]
            arg = self.unparse_grounding(tree['grounding'])
            return [arg], is_distinct

        def unparse_ref(self, tree):
            assert isinstance(tree, int)
            return QDMRStepArg('ref', ['#' + str(tree + 1)])


        def unparse_grounding(self, tree):
            assert isinstance(tree, int)
            # TODO choose one from tuple
            entity = self.ids_to_grounding_choices[tree].choice
            entity_type = self.ids_to_grounding_choices[tree].choice_type
            if entity_type == 'table':
                # table
                grounding = GroundingKey.make_table_grounding(entity)
            elif entity_type == 'column':
                # (table, column)
                grounding = GroundingKey.make_column_grounding(*entity)
            elif entity_type == 'value':
                # value unit
                if self.column_data is None:
                    # full break
                    if len(entity) > 1:
                        print('{} possible values in full break comparative: {}'.format(len(entity), entity))
                    value = entity[0].value
                    grounding = GroundingKey.make_text_grounding(value)
                else:
                    # here table and column can be unknown
                    entity = self.filter_value_units_with_column_use(entity, use_column=True)
                    if len(entity) != 1:
                        print('{} possible values in select/project: {}'.format(len(entity), entity))
                    entity = list(entity.values())[0]
                    table_name, col_name, val = entity.table, entity.column, entity.value
                    grounding = GroundingKey.make_value_grounding(table_name, col_name, val)
            else:
                print(self.ids_to_grounding_choices[tree])
                raise RuntimeError

            return QDMRStepArg('grounding', grounding)

        def unparse_val_unit(self, tree):
            table_name, col_name = self.get_tbl_col_names(tree['col_id'])
            val = self.unparse_value(tree['val_id'])
            return GroundingKey.make_value_grounding(table_name, col_name, val)

        def unparse_value(self, val_idx):
            for val_unit in self.value_unit_dict.values():
                if val_unit[0].idx == val_idx:
                    return val_unit[0].value
            print(self.value_unit_dict, val_idx)
            raise RuntimeError 

        def unparse_aggregate(self, tree):
            if tree['agg_type']['_type'] == 'AggOp':
                agg_type = AGG_TYPES_B[tree['agg_type']['agg_ops']['_type']]
            else:
                assert tree['agg_type']['_type'] == 'UseColumn', tree['agg_type']['_type']
                agg_type = self.unparse_grounding(tree['agg_type']['grounding']).arg

            arg1 = QDMRStepArg('grounding', agg_type)
            arg2 = self.unparse_ref(tree['ref'])
            return [arg1, arg2]

        def get_column_grounding(self, col_id):
            table_name, col_name = self.get_tbl_col_names(col_id)
            return GroundingKey.make_column_grounding(table_name, col_name)

        def unparse_comparative(self, tree):
            is_distinct = BOOLS_B[tree['is_distinct']['_type']]
            arg1 = self.unparse_ref(tree['ref1'])
            arg2 = self.unparse_ref(tree['ref2'])
            arg3 = self.unparse_comparative_3arg_type(tree['comparative_3arg_type'])
            return [arg1, arg2, arg3], is_distinct

        def unparse_comparative_3arg_type(self, tree):
            comparative_3arg_type = tree['_type']

            if comparative_3arg_type == 'CompArg':
                comp_op = self.unparse_comp_op_type(tree['comp_op_type'])
                column = self.unparse_column_type(tree['column_type'])
                arg = self.unparse_comp_val(tree['comp_val'], comp_op, column)
            elif comparative_3arg_type == 'NoneCompArg':
                arg = QDMRStepArg('grounding', None)
            else:
                print('problem with comparative 3arg type {}'.format(comparative_3arg_type))
                raise RuntimeError
            return arg

        def unparse_comp_op_type(self, tree):
            comp_op_type = tree['_type']
            if comp_op_type == 'CompOp':
                return COMP_B[tree['comparative_ops']['_type']]
            elif comp_op_type == 'NoOp':
                return None
            else:
                print('problem with comp_op_type {}'.format(comp_op_type))
                raise RuntimeError

        def unparse_column_type(self, tree):
            column_type = tree['_type']
            if column_type == 'ColumnGrounding':
                return self.unparse_grounding(tree['grounding']).arg
            elif column_type == 'NoColumnGrounding':
                return None
            else:
                print('problem with comparative column_type {}'.format(column_type))
                raise RuntimeError

        def filter_value_units_with_column_use(self, val_units, use_column=False):
            new_col_val_units = {}
            # get values witout columns
            for val_unit in val_units:
                if use_column is False and val_unit.column is None or \
                    use_column is True and val_unit.column:
                    #  and val_unit.column is None:
                    new_col_val_units[val_unit.value] = val_unit
            return new_col_val_units

        def unparse_comp_val(self, tree, comp_op, column):
            comp_val_type = tree['_type']
            if comp_val_type == 'CompGrounding':
                  # TODO choose one from tuple
                if tree['grounding'] not in self.ids_to_grounding_choices:
                    print(tree)
                    print(self.ids_to_grounding_choices)
                entity_type = self.ids_to_grounding_choices[tree['grounding']].choice_type

                if entity_type == 'table' or entity_type == 'column':
                    qdmr_arg = self.unparse_grounding(tree['grounding'])
                    assert qdmr_arg.arg_type == 'grounding'
                    grounding = qdmr_arg.arg
                    assert not comp_op and not column, (comp_op, column, qdmr_arg)

                elif entity_type == 'value':
                    val_units = self.ids_to_grounding_choices[tree['grounding']].choice
                    if self.column_data is None:
                        # full break  
                        if len(val_units) > 1:
                            print('{} possible values in full break comparative: {}'.format(len(val_units), val_units))
                        
                        value = val_units[0].value
                        if comp_op is None:
                            grounding = GroundingKey.make_text_grounding(value)
                        else:
                            grounding = GroundingKey.make_comparative_grounding(comp_op, value)
                    else:
                        assert not column or column.iscol(), column
                        if column is not None:
                            table_name, column_name = column.get_tbl_name(), column.get_col_name()
                            val_type = self.column_data[table_name][column_name]
                            typed_val_units = {}
                            # get values with correct column
                            for val_unit in val_units:
                                if val_unit.column == column_name and val_unit.table == table_name:
                                    typed_val_units[val_unit.value] = val_unit
                            # get values with correct type
                            if len(typed_val_units) < 1:
                                for val_unit in val_units:
                                    if val_unit.value_type == val_type:
                                        typed_val_units[val_unit.value] = val_unit
                            if len(typed_val_units) != 1:
                                print('{} possible values in comp value with col: {}'.format(len(typed_val_units), typed_val_units))
                            value = list(typed_val_units.keys())[0]
                        else:
                            no_col_val_units = self.filter_value_units_with_column_use(val_units, use_column=False)
                            if len(no_col_val_units) != 1:
                                print('{} possible values in comp value without col: {}'.format(len(no_col_val_units), no_col_val_units))
                            value = list(no_col_val_units.keys())[0]
                        if comp_op is None:
                            comp_op = '='
                        grounding = GroundingKey.make_comparative_grounding(comp_op, value, column)
            elif comp_val_type == 'CompRef':
                ref_arg = self.unparse_ref(tree['ref'])
                if comp_op is None:
                    comp_op = '='
                assert comp_op, comp_op
                grounding = GroundingKey.make_comparative_grounding(comp_op, ref_arg.arg[0])
            else:
                print('problem with comparative value type {}'.format(comp_val_type))
                raise RuntimeError
            return QDMRStepArg('grounding', grounding)

        def unparse_comparative_3arg(self, tree):
            comparative_3arg_type = tree['_type']

            if comparative_3arg_type == 'GroundingArg':
                arg = self.unparse_grounding(tree['grounding'])
            elif comparative_3arg_type == 'CompArg':
                comp_op = COMP_B[tree['comp_op']['_type']]
                arg = self.unparse_comparative_arg(comp_op, tree['comparative_arg'])
            elif comparative_3arg_type == 'NoneCompArg':
                arg = QDMRStepArg('grounding', None)
            else:
                print('problem with comparative 3arg type {}'.format(comparative_3arg_type))
                raise RuntimeError
            return arg

        def unparse_comparative_arg(self, comp_op, tree):
            comparative_arg_type = tree['_type']
            column_grounding = None
            if comparative_arg_type == 'CompValFull':
                val_grounding = self.unparse_val_unit(tree['val_unit'])
                val = val_grounding.get_val()
                col, tbl = val_grounding.get_col_name(), val_grounding.get_tbl_name()
                column_grounding = GroundingKey.make_column_grounding(tbl, col)
            elif comparative_arg_type == 'CompVal':
                val = self.unparse_value(tree['val_id'])
            elif comparative_arg_type == 'CompRef':
                val = self.unparse_ref(tree['ref']).arg[0]
            else:
                print('problems with comp arg {}'.format(tree))
                raise RuntimeError
            
            grounding = GroundingKey.make_comparative_grounding(comp_op, val, column_grounding)
            return QDMRStepArg('grounding', grounding)

        def unparse_project(self, tree):
            is_distinct = BOOLS_B[tree['is_distinct']['_type']]
            arg1 = self.unparse_project_1arg(tree['project_1arg'])
            arg2 = self.unparse_ref(tree['ref'])
            return [arg1, arg2], is_distinct

        def unparse_project_1arg(self, tree):
            project_1arg_type = tree['_type']
            if project_1arg_type == 'GroundingProjectArg':
                return self.unparse_grounding(tree['grounding'])
            elif project_1arg_type == 'NoneProjectArg':
                return QDMRStepArg('grounding', None)
            else:
                print('problems with project 1st arg {}'.format(tree))
                raise RuntimeError

        def unparse_superlative(self, tree):
            sup_op_type = tree['superlative_op_type']['_type']
            if sup_op_type == 'SupOp':
                arg1 = QDMRStepArg('grounding', SUP_B[tree['superlative_op_type']['superlative_ops']['_type']])
            else: 
                print('problems with superlative 1st arg {}'.format(tree))
                raise RuntimeError
            arg2 = self.unparse_ref(tree['ref1'])
            arg3 = self.unparse_ref(tree['ref2'])
            return [arg1, arg2, arg3]

        def unparse_intersection(self, tree):
            arg1 = self.unparse_ref(tree['ref1'])
            arg2 = self.unparse_ref(tree['ref2'])
            arg3 = self.unparse_union_3arg(tree['union_3arg'])
            return [arg1, arg2, arg3]

        def unparse_union_3arg(self, tree):
            union_3arg_type = tree['_type']
            if union_3arg_type == 'RefArg':
                return self.unparse_ref(tree['ref'])
            elif union_3arg_type == 'NoneUnionArg':
                return QDMRStepArg('grounding', None)
            else:
                print('problems with union 3rd arg')
                raise RuntimeError

        def unparse_discard(self, tree):
            arg1 = self.unparse_ref(tree['ref1'])
            arg2 = self.unparse_ref(tree['ref2'])
            return [arg1, arg2]

        def unparse_union(self, tree):
            arg1 = self.unparse_ref(tree['ref1'])
            arg2 = self.unparse_ref(tree['ref2'])
            arg3 = self.unparse_union_3arg(tree['union_3arg'])
            arg4 = self.unparse_union_3arg(tree['union_4arg'])
            return [arg1, arg2, arg3, arg4]

        def unparse_sort(self, tree):
            arg1 = self.unparse_ref(tree['ref1'])
            arg2 = self.unparse_ref(tree['ref2'])
            arg3 = QDMRStepArg('grounding', ORDERS_B[tree['order']['_type']])
            return [arg1, arg2, arg3]

        def unparse_group(self, tree):
            if tree['agg_type']['_type'] == 'AggOp':
                agg_type = AGG_TYPES_B[tree['agg_type']['agg_ops']['_type']]
            else:
                assert tree['agg_type']['_type'] == 'UseColumn', tree['agg_type']['_type']
                agg_type = self.unparse_grounding(tree['agg_type']['grounding']).arg

            arg1 = QDMRStepArg('grounding', agg_type)
            arg2 = self.unparse_ref(tree['ref1'])
            arg3 = self.unparse_ref(tree['ref2'])
            return [arg1, arg2, arg3]

        def unparse_arithmetic(self, tree):
            arg1 = QDMRStepArg('grounding', UNIT_B[tree['unit_ops']['_type']])
            arg2 = self.unparse_ref(tree['ref1'])
            arg3 = self.unparse_ref(tree['ref2'])
            return [arg1, arg2, arg3]
