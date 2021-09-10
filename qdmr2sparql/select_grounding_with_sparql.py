import os
import copy
import argparse
import time

from qdmr2sparql.datasets import DatasetBreak, DatasetSpider
from qdmr2sparql.structures import GroundingKey, GroundingIndex
from qdmr2sparql.structures import save_grounding_to_file, load_grounding_from_file, load_grounding_list_from_file, assert_check_grounding_save_load
from qdmr2sparql.structures import RdfGraph
from qdmr2sparql.structures import QueryResult
from qdmr2sparql.query_generator import create_sparql_query_from_qdmr
from qdmr2sparql.process_sql import replace_orderByLimit1_to_subquery

from qdmr2sparql.utils_qdmr2sparql import handle_exception_sparql_process, TimeoutException, handle_exception
from qdmr2sparql.utils_qdmr2sparql import SparqlGenerationError, SparqlRunError, SparqlWrongAnswerError


SQL_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'distinct')
JOIN_KEYWORDS = ('join', 'on', 'by', 'having') #'as'
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
COND_OPS = ('and', 'or')
ORDER_OPS = ('desc', 'asc')


def parse_args():
    parser = argparse.ArgumentParser(description='Build grounding between QDMR and SQL.')
    parser.add_argument('--qdmr_path', type=str, help='path to break dataset')
    parser.add_argument('--spider_path', type=str, help='path to spider dataset')
    parser.add_argument('--db_path', type=str, default="database", help='path from --spider_path to get databases, default - "database"')
    parser.add_argument('--output_path', type=str, default=None,help='path to output file with grounding (found correct SPARQL script)')
    parser.add_argument('--output_path_all', type=str, default=None,help='path to output file with grounding')
    parser.add_argument('--dev', action='store_true', help='if true, use dev, else use train')
    parser.add_argument('--start_spider_idx', type=int, default=None, help='index of first spider example')
    parser.add_argument('--end_spider_idx', type=int, default=None, help='index of last spider example')
    parser.add_argument('--input_grounding', type=str, default=None, help='grounding to start from')
    parser.add_argument('--not_all_sql', action='store_true', default=False, help='allows not grounded some sql')
    parser.add_argument('--without_sql', action='store_true', default=False, help='ground only scheme, without sql args')
    parser.add_argument('--time_limit', type=int, default=600, help='time limit in seconds to process one example')
    parser.add_argument('--virtuoso_server', type=str, default=None, help='Path to Virtuoso HTTP service (looks like "http://localhost:8890/sparql/"')
    parser.add_argument('--spider_idx', type=str, help='index of spider example, use only for debugging')
    args = parser.parse_args()
    return args


def create_list_of_groundings_to_try(input_grounding, essential_groundings=None, unique_essential_groundings=True):
    list_to_try = []

    indices_to_search = [k for k in input_grounding if isinstance(k, GroundingIndex)]
    indices_everywhere = [k for k in input_grounding if not isinstance(k, GroundingIndex)]

    grnd_joint = {k: input_grounding[k] for k in indices_everywhere}
    index_list = []

    def get_all_combination_from_index(i):
        if i == len(indices_to_search):
            return [copy.deepcopy(grnd_joint)] # list of dicts

        endings = get_all_combination_from_index(i + 1)
        options = input_grounding[indices_to_search[i]]

        list_to_try = []
        for e in endings:
            for o in options:
                grnd = {indices_to_search[i] : o}
                grnd.update(e)
                list_to_try.append(grnd)
        return list_to_try

    list_to_try = get_all_combination_from_index(0)

    # filter combinations based on essential_groundings if provided
    if essential_groundings:
        list_to_try_base = copy.deepcopy(list_to_try)
        list_to_try = []
        for grnd in list_to_try_base:
            available_essential_groundings = copy.deepcopy(essential_groundings)
            assigned = [False] * len(available_essential_groundings)
            have_duplicate_groundings = False

            for k, v in grnd.items():
                if isinstance(k, GroundingIndex):

                    def match_val_to_comp(a, b):
                        if not isinstance(a, GroundingKey) or not isinstance(b, GroundingKey):
                            return False
                        return a.isval() and b.iscomp() and b.keys[0] == "=" and b.keys[1] == a.get_val()

                    essential_index = None
                    for i_e, essential in enumerate(available_essential_groundings):
                        if essential == v or match_val_to_comp(v, essential)  or match_val_to_comp(essential, v):
                            essential_index = i_e
                            break

                    if essential_index is not None:
                        if not assigned[essential_index]:
                            assigned[essential_index] = True
                        else:
                            have_duplicate_groundings = True

            if unique_essential_groundings and have_duplicate_groundings:
                continue

            if all(assigned):
                list_to_try.append(grnd)

    return list_to_try


def select_grounding(qdmr, qdmr_name, dataset_spider, db_path, grounding=None, verbose=True, time_limit=None, use_extra_tests=False, virtuoso_server=None):
    if grounding is None:
        grounding = {}

    sql_data = dataset_spider.sql_data[qdmr_name]
    db_id = sql_data['db_id']
    schema = dataset_spider.schemas[db_id]
    sql_query = sql_data['query']

    if verbose:
        print("db_id:", db_id)
        print("Question:", sql_data["question"])
        print("SQL query:", sql_query)
        print(f"QDMR:\n{qdmr}")
        print(f"Groundings: {grounding}")
        print(f"Database schema {db_id}:")
        for tbl_name, cols in schema.column_names.items():
            print(f"{tbl_name}: {cols}")

    schema.load_table_data(db_path)
    rdf_graph = RdfGraph(schema)

    schemas_to_test = [schema]
    rdf_graphs_to_test = [rdf_graph]
    if use_extra_tests:
        schema.load_test_table_data(db_path)
        for s in schema.test_schemas:
            test_rdf_graph = RdfGraph(s)
            rdf_graphs_to_test.append(test_rdf_graph)
            schemas_to_test.append(s)

    essential_groundings = grounding["ESSENTIAL_GROUNDINGS"] if "ESSENTIAL_GROUNDINGS" in grounding else None
    groundings_to_try = create_list_of_groundings_to_try(grounding, essential_groundings)

    groundings_all_results = []
    groundings_positive_results = []

    time_start = time.time()
    time_limit_exceeded = False

    # try query modifications; if not successful run query as is
    try: 
        sql_query_modified = replace_orderByLimit1_to_subquery(sql_query, schema.column_names)
        sql_results = [QueryResult.execute_query_sql(sql_query_modified, s) for s in schemas_to_test]
    except:
        sql_results = [QueryResult.execute_query_sql(sql_query, s) for s in schemas_to_test]

    # is_equal = sql_results[0].is_equal_to(sql_results_modified[0], require_column_order=True, require_row_order=True, schema=schemas_to_test[0])
    
    if verbose:
        for sql_result in sql_results:
            print("SQL result:", sql_result)

    for grnd in groundings_to_try:
        if verbose:
            grnd_for_printing = copy.deepcopy(grnd)
            if "MESSAGES" in grnd_for_printing:
                del grnd_for_printing["MESSAGES"]
            print("Trying", grnd_for_printing)
        got_correct_answer = False

        try:
            try:
                sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grnd, strict_mode=True)
            except Exception as e:
                raise SparqlGenerationError() from e

            for cur_schema, cur_rdf, cur_sql_result in zip(schemas_to_test, rdf_graphs_to_test, sql_results):
                try:
                    result = QueryResult.execute_query_to_rdf(sparql_query, cur_rdf, cur_schema, virtuoso_server=virtuoso_server)
                    if verbose:
                        print("SPARQL result:", result)
                except Exception as e:
                    raise SparqlRunError() from e

                try:
                    ordered = True if sql_data["sql"]["orderBy"] and not (sql_data["sql"]["limit"] and sql_data["sql"]["limit"] == 1) else False

                    equal, message = result.is_equal_to(cur_sql_result,
                            require_column_order=True,
                            require_row_order=ordered,
                            weak_mode_argmax=False,
                            return_message=True)
                    assert equal, message
                except Exception as e:
                    raise SparqlWrongAnswerError() from e

            got_correct_answer = True
            message = f"{os.path.basename(__file__)}: OK"
        except TimeoutException as e:
            # timeout
            raise e
        except Exception as e:
            error_details = handle_exception_sparql_process(e, verbose=False)
            message = f"{os.path.basename(__file__)}: SPARQL_error_type: {error_details['sparql_error_type']}, ERROR: {error_details['type']}:{error_details['message']}, file: {error_details['file']}, line {error_details['line_number']}"
        
        cur_result = copy.deepcopy(grnd)
        if "MESSAGES" in cur_result:
            cur_result["MESSAGES"].append(message)
        else:
            cur_result["MESSAGES"] = [message]

        groundings_all_results.append(cur_result)
        if got_correct_answer:
            groundings_positive_results.append(cur_result)
        else:
            cur_result["ERRORS"] = [error_details]

        if time_limit is not None and (time.time() - time_start > time_limit):
            time_limit_exceeded = True
            break

    warning_message = []
    if not sql_results[0]:
        warning_message.append("WARNING: empty SQL result")

    if time_limit_exceeded:
        warning_message.append(f"WARNING: time limit of {time_limit}s exceeded")

    if len(groundings_positive_results) == 0:
        warning_message.append("WARNING: no correct result found")
    elif len(groundings_positive_results) > 1:
        warning_message.append(f"WARNING: multiple ({len(groundings_positive_results)}) correct results found")
    if warning_message:
        warning_message = "\n" + "\n".join(warning_message)
    else:
        warning_message = ""

    # add warning messages: posotive groundings are from the same object so it's enough only to append to groundings_all_results
    for res in groundings_all_results:
        res["MESSAGES"][-1] = res["MESSAGES"][-1] + warning_message

    message = f"{os.path.basename(__file__)}, {qdmr_name}: "
    if len(groundings_positive_results) == 1:
        message = message + "OK"
    elif len(groundings_positive_results) > 1:
        message = message + f"OK: multiple groundings {len(groundings_positive_results)}"
    else:
        message = message + "No correct match"
    if not sql_results[0]:
        message = message + "; empty SQL result"
    if time_limit_exceeded:
        message = message + f"; time limit {time_limit}s exceeded"

    if verbose:
        print(message)

    if "OK" in message:
        return groundings_positive_results, message
    else:
        return groundings_all_results, message


def main(args):
    print(args)

    split_name = 'dev' if args.dev else 'train'

    db_path = os.path.join(args.spider_path, args.db_path)
    dataset_break = DatasetBreak(args.qdmr_path, split_name)
    dataset_spider = DatasetSpider(args.spider_path, split_name)

    if args.input_grounding:
        input_grounding = load_grounding_from_file(args.input_grounding)
    else:
        input_grounding = {}


    if args.spider_idx is not None:
        qdmr_name = None
        for name in dataset_break.names:
            idx = dataset_break.get_index_from_name(name)
            if idx == args.spider_idx:
                qdmr_name = name
                break

        assert qdmr_name is not None, "Could find QDMR with index {args.spider_idx}"
        qdmr = dataset_break.qdmrs[qdmr_name]
        qdmr_grounding = input_grounding[qdmr_name] if qdmr_name in input_grounding else None

        # debugging
        print()
        print(qdmr_name)

        groundings, _ = select_grounding(qdmr, qdmr_name, dataset_spider, db_path, grounding=qdmr_grounding, verbose=True,
                                         time_limit=args.time_limit, virtuoso_server=args.virtuoso_server)
    else:
        groundings_all = {}
        groundings_only_positive = {}

        if split_name == "train":
            qdmr_list = [(qdmr_name, qdmr) for qdmr_name, qdmr in dataset_break.make_iterator(args.start_spider_idx, args.end_spider_idx)]
        else:
            qdmr_list = [(qdmr_name, qdmr) for qdmr_name, qdmr in dataset_break]

        for qdmr_name, qdmr in qdmr_list:
            spider_idx = DatasetBreak.get_index_from_name(qdmr_name)
            qdmr_grounding = input_grounding[qdmr_name] if qdmr_name in input_grounding else None

            try:
                groundings, message = select_grounding(qdmr, qdmr_name, dataset_spider, db_path, grounding=qdmr_grounding, verbose=False,
                                                       time_limit=args.time_limit, virtuoso_server=args.virtuoso_server)
                groundings_all[qdmr_name] = {"GROUNDINGS": groundings}
                groundings_all[qdmr_name]["MESSAGES"] = [message]
                if "OK" in message:
                    groundings_only_positive[qdmr_name] = {"GROUNDINGS": groundings}
                    groundings_only_positive[qdmr_name]["MESSAGES"] = [message]

                print(message)
            except Exception as e:
                error_details = handle_exception(e, verbose=False)
                print(f"ERROR: {error_details['type']}:{error_details['message']}, file: {error_details['file']}, line {error_details['line_number']}")
                groundings_all[qdmr_name] = {"ERRORS" : [error_details]}

            if args.output_path:
                save_grounding_to_file(args.output_path, groundings_only_positive)

                check = load_grounding_list_from_file(args.output_path)
                assert_check_grounding_save_load(groundings_only_positive, check)

            if args.output_path_all:
                save_grounding_to_file(args.output_path_all, groundings_all)

                check = load_grounding_list_from_file(args.output_path_all)
                assert_check_grounding_save_load(groundings_all, check)


if __name__ == "__main__":
    args = parse_args()
    main(args)
