from collections import defaultdict
from copy import deepcopy

import numpy as np

from text2qdmr.datasets.qdmr import AGG_OPS, ARITH_OPS, ORDER_OPS, DEFAULT_AGG_TYPE, requires_distinct
from text2qdmr.datasets.qdmr import QDMRStepArg
from text2qdmr.datasets.utils.extract_values import GroundingKey
from text2qdmr.datasets.utils.visualization import Visualizer

from qdmr2sparql.structures import GroundingIndex, QdmrInstance
from qdmr2sparql.structures import QueryResult
from qdmr2sparql.query_generator import create_sparql_query_from_qdmr
from qdmr2sparql.utils_qdmr2sparql import handle_exception_sparql_process, time_limit, TimeoutException
from qdmr2sparql.utils_qdmr2sparql import SparqlGenerationError, SparqlRunError, SparqlWrongAnswerError
from utils.spider_evaluation import count_component1, count_component2, count_others

class Metrics:
    def __init__(self, writer=None, logdir=None, virtuoso_server=None):
        self.visualizer = Visualizer(writer, logdir)
        self.virtuoso_server = virtuoso_server

        self.scores = {'val': {}, 'test': {}}

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    def postprocess_grounding(self, op, grounding_arg):
        if grounding_arg in AGG_OPS or grounding_arg in ARITH_OPS:
            if op == 'superlative':
                return GroundingKey.make_comparative_grounding(grounding_arg, None)
            else:
                assert op == 'aggregate' or op == 'group' or op == 'arithmetic', op
                return grounding_arg
        elif grounding_arg in ORDER_OPS:
            assert op == 'sort', op
            return GroundingKey.make_sortdir_grounding(grounding_arg == 'ascending')
        else:
            assert isinstance(grounding_arg, GroundingKey), (op, grounding_arg)
            return grounding_arg

    def requires_none_arg(self, op, i_arg):
        if op == 'project' and i_arg == 0 or \
            op == 'comparative' and i_arg == 2:
            return True
        elif (op == 'union' or op == 'intersection') and i_arg >= 2:
            return False
        else: 
            raise Exception(f'None in {op}, {i_arg}')

    def postprocess_inferred_code(self, inferred_code):
        distinct_idx = []
        break_operators = []
        qdmr_op_args = []
        grounding = dict()

        for i, qdmr_step in enumerate(inferred_code):
            op, args_list = qdmr_step[:2]
            break_operators.append(op)

            if requires_distinct(op):
                is_distinct = qdmr_step[2]
                if is_distinct:
                    distinct_idx.append(QdmrInstance.index_to_ref(i))

            args_list_transformed = []
            for i_arg, step_arg in enumerate(args_list):
                assert isinstance(step_arg, QDMRStepArg), step_arg
                if step_arg.arg is None:
                    arg_str = str(step_arg.arg) if self.requires_none_arg(op, i_arg) else None
                elif step_arg.arg_type == 'ref':
                    arg_str = step_arg.arg[0]
                elif step_arg.arg_type == 'grounding':
                    grounding_arg = self.postprocess_grounding(op, step_arg.arg)

                    if isinstance(grounding_arg, GroundingKey):
                        arg_str = repr(grounding_arg)
                        if op == 'aggregate' or op == 'group' and i_arg == 0:
                            arg_str = DEFAULT_AGG_TYPE
                        grounding[GroundingIndex(i, i_arg, arg_str)] = grounding_arg
                    else:
                        assert grounding_arg in AGG_OPS or grounding_arg in ARITH_OPS, grounding_arg
                        arg_str = grounding_arg

                if arg_str:
                    args_list_transformed.append(arg_str)

            qdmr_op_args.append(args_list_transformed)
        
        if distinct_idx:
            grounding['distinct'] = distinct_idx

        return QdmrInstance(break_operators, qdmr_op_args), grounding

    def compare_sql_qdmr(self, qdmr, sql, schema, rdf_graph, grnd, tl=60, ordered=False, verbose=True):
        got_correct_answer, error_details = False, None
        try:
            sql_result = QueryResult.execute_query_sql(sql, schema)
        except Exception as e:
            return False, None

        try:
            with time_limit(tl):
                try:
                    sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grnd)
                except Exception as e:
                    raise SparqlGenerationError() from e

                try:
                    result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=self.virtuoso_server)
                except Exception as e:
                    raise SparqlRunError() from e

                try:
                    equal, message = result.is_equal_to(sql_result,
                            require_column_order=False,
                            require_row_order=ordered,
                            weak_mode_argmax=True,
                            return_message=True)
                    assert equal, message
                except Exception as e:
                    raise SparqlWrongAnswerError() from e

                got_correct_answer = True
        except TimeoutException as e:
            error_details = 'TL'
        except Exception as e:
            error_details = handle_exception_sparql_process(e, verbose=False)
            assert error_details['sparql_error_type']
            
        return got_correct_answer, error_details

    def format_gold_qdmr_args(self, qdmr, grounding):
        for i, step_args in enumerate(qdmr.args):
            for i_arg, arg in enumerate(step_args):
                grounding_arg = grounding.get(GroundingIndex(i, i_arg, arg))
                if grounding_arg:
                    qdmr.args[i][i_arg] = repr(grounding_arg)
        return qdmr

    def _evaluate_one(self, break_item, inferred_code, section, check_gold=False, verbose=False):
        if not inferred_code:
            return False, False, None
        if section != 'test':
            exact_match = (inferred_code == break_item.qdmr_code[0])
        else:
            exact_match = False

        sql_hardness = 'unknown'

        got_correct_answer, error_details = False, None
        qdmr, grounding = self.postprocess_inferred_code(inferred_code)

        if break_item.schema:
            schema, rdf_graph = break_item.eval_graphs

            sql = break_item.orig_spider_entry['query']
            sql_hardness = self.eval_hardness(break_item.orig_spider_entry['sql'])
            ordered = True if break_item.sql_code['orderBy'] else False
        
            got_correct_answer, error_details = self.compare_sql_qdmr(qdmr, sql, schema, rdf_graph, grounding, \
                                                                        ordered=ordered, verbose=verbose)

        if section != 'test':
            gold_qdmr = QdmrInstance(break_item.qdmr_ops, break_item.qdmr_args)
            gold_grounding = break_item.grounding[0]
            if check_gold:
                gold_got_correct_answer, _ = self.compare_sql_qdmr(gold_qdmr, sql, schema, rdf_graph, gold_grounding, \
                                                                ordered=ordered, verbose=verbose)
                if not gold_got_correct_answer:
                    print('Gold Error in {}'.format(break_item.subset_idx))

            gold_distinct_idx = gold_grounding.get('distinct')
            gold_qdmr = self.format_gold_qdmr_args(gold_qdmr, gold_grounding)
            gold_qdmr_info = (gold_qdmr, gold_distinct_idx)
        else:
            gold_qdmr_info = None 
        distinct_idx = grounding.get('distinct')
        qdmr_info = (qdmr, distinct_idx)
        self.visualizer.visualization(break_item, qdmr_info, gold_qdmr_info, got_correct_answer, sql_hardness)

        if verbose and error_details and error_details['sparql_error_type'] in ('SparqlGenerationError, SparqlRunError'):
            print()
            if not got_correct_answer:
                print('Error in {}'.format(break_item.subset_idx))
                print(error_details)
            
        elif error_details == 'TL':
            print('TL in {}'.format(break_item.subset_idx))

        return exact_match, got_correct_answer, sql_hardness

    def add(self, break_item, inferred_code, section):
        if not break_item:
            return 
        
        exact_match, exec_match, sql_hardness = self._evaluate_one(break_item, inferred_code, section)
        self.scores[section][break_item.full_name] = {'ex': exec_match, 'em': exact_match, 'hardness': sql_hardness}

    def finalize(self):
        self.visualizer.finalize()

        mean_exec_match = float(np.mean([el['ex'] for v in self.scores.values() for el in v.values()]))
        mean_exact_match = float(np.mean([el['em'] for v in self.scores.values() for el in v.values()]))
        res = {
            'per_item': [{'name': name, 'ex': score['ex'], 'em': score['em'], 'hardness': score['hardness']} for sec_d in self.scores.values() for name, score in sec_d.items()],
            'total_scores': {'ex': mean_exec_match, 'em': mean_exact_match},
        }

        levels = {'easy', 'medium', 'hard', 'extra'}
        for section in self.scores.keys():
            res['total_scores']['ex_{}'.format(section)] = float(np.mean([el['ex'] for el in self.scores[section].values()]))
            for lvl in levels:
                res['total_scores']['ex_{}_{}'.format(section, lvl)] = float(np.mean([el['ex'] for el in self.scores[section].values() if el['hardness'] == lvl]))

        return res
