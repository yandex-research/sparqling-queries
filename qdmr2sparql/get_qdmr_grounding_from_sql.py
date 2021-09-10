import os
import argparse

from qdmr2sparql.grounding_utils import stem_match, measure_similarity

from qdmr2sparql.datasets import DatasetBreak, DatasetSpider
from qdmr2sparql.structures import GroundingKey, GroundingIndex, QdmrInstance
from qdmr2sparql.structures import save_grounding_to_file, load_grounding_from_file, assert_check_grounding_save_load

from qdmr2sparql.utils_qdmr2sparql import handle_exception



## CONSTANTS FROM SPIDER (https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql)
CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')
#####

keywords_distinct = ["distinct", "different", "distinctive"]
stop_words_distinct = ["#REF", "the", "of", "for", "that", "are", "is", "which"]


def parse_args():
    parser = argparse.ArgumentParser(description='Check correctness of QDMR annotation.')
    parser.add_argument('--dev', action='store_true', help='if true, use dev, else use train')
    parser.add_argument('--spider_idx', type=str, help='index of spider example, use only for debugging')
    parser.add_argument('--qdmr_path', type=str, help='path to break dataset')
    parser.add_argument('--spider_path', type=str, help='path to spider dataset')
    parser.add_argument('--output_path', type=str, default=None, help='file to save the fixed QDMR ops')
    args = parser.parse_args()
    return args

def split_camel_case(s):
    words = [[s[0]]]
    for c in s[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
    return " ".join(["".join(word) for word in words])

def is_text_match(a, b):
    good_match = False
    for match_function in [stem_match, measure_similarity]:
        for b_candidate in [b, b.replace("_", " "), split_camel_case(b)]:
            good_match = good_match or match_function(a, b_candidate)

    return good_match != 0

def compute_grounding(qdmr, qdmr_name, dataset_spider, verbose=True):

    sql_data = dataset_spider.sql_data[qdmr_name]
    db_id = sql_data['db_id']
    table_data = dataset_spider.table_data[db_id]
    sql_unit = sql_data['sql']
    schema = dataset_spider.schemas[db_id]

    if verbose:
        print("db_id:", db_id)
        print("Question:", sql_data["question"])
        print("SQL query:", sql_data['query'])
        print(f"QDMR:\n{qdmr}")

    head_grounding = parse_spider_select_field(sql_unit['select'], sql_unit, table_data)

    grounding = {}
    i_op = len(qdmr) - 1
    op_grounder[qdmr.ops[i_op]](head_grounding, sql_unit, i_op, qdmr, sql_data, table_data, grounding)

    # check that all the gounded args are similar to the targets via text
    message = []
    warning_messages = []
    for grnd_index, grnd in grounding.items():
        grnd = grnd[0]
        good_match = False
        if grnd_index == "distinct":
            # do not check for distinct flags here
            good_match = True
        elif grnd == "":
            # added empty grounding for distinct only case (e.g., SPIDER_dev_310) not to search for grounding later
            good_match = True
        elif grnd.iscol():
            col_name = grnd.get_col_name()
            tbl_name = grnd.get_tbl_name()

            good_match = is_text_match(grnd_index.key_str, col_name)
            if not good_match:
                # try more complicated matchings

                # check if we grounded to the primary key, but the text matches to the table name - just change the grounding to the table name
                if col_name == schema.primary_keys[tbl_name]:
                    if is_text_match(grnd_index.key_str, tbl_name):
                        grounding[grnd_index] = [GroundingKey.make_table_grounding(tbl_name)]
                        good_match = True
        elif grnd.istbl():
            tbl_name = grnd.get_tbl_name()

            good_match = is_text_match(grnd_index.key_str, tbl_name)
            if not good_match:
                # try more complicated matchings

                # check if we grounded to the table but the text matches to the primary key - ground to the primary key
                if is_text_match(grnd_index.key_str, schema.primary_keys[tbl_name]):
                    grounding[grnd_index] = [GroundingKey.make_column_grounding(tbl_name, schema.primary_keys[tbl_name])]
                    good_match = True
        elif grnd.issortdir():
            # always trust sortdir matches
            good_match = True
        elif grnd.iscomp():
            # for now always trust comparative matches
            good_match = True

        if not good_match:
            warning_messages += [f"WARNING: text does not match: {grnd_index} and {grnd}"]

    # count the number of grounded args for printing
    args_to_ground = 0
    args_grounded = 0
    for i_op, (qdmr_op, qdmr_args) in enumerate(qdmr):
        for i_arg in op_args_to_ground[qdmr_op]:
            args_to_ground += 1
            if i_arg < len(qdmr_args):
                if GroundingIndex(i_op, i_arg, qdmr_args[i_arg]) in grounding:
                    args_grounded += 1

    message += [f"{os.path.basename(__file__)}: grounded {args_grounded} args from {args_to_ground}, {grounding}"]
    message += warning_messages
    for m in message:
        print(m)

    message = "\n".join(message)

    return grounding, message


def parse_spider_select_field(select_field, sql_unit, table_data):
    # 'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])

    is_distinct_total, out_list = select_field

    groundings = []
    for out in out_list:
        agg_op1 = AGG_OPS[out[0]]
        agg_op2, grnd = parse_spider_val_unit(out[1], sql_unit, table_data)

        assert agg_op2 is None or agg_op2 == "none", "Internal agrregator is not none, need to implement this case"
        groundings.append( (agg_op1, grnd) )

    return groundings


def parse_spider_col_unit(col_unit, sql_unit, table_data):
    agg_id, col_id, is_distinct = col_unit
    agg_op = AGG_OPS[agg_id]

    table_id, col_name = table_data["column_names_original"][col_id]
    if table_id != -1:
        table_name = table_data["table_names_original"][table_id]
        grnd = GroundingKey.make_column_grounding(table_name, col_name)
    else:
        table_units = sql_unit['from']['table_units']
        if len(table_units) == 1 and table_units[0][0] == "table_unit":
            assert len(table_units) == 1, f"have {len(table_units)} table_units and '*'"
            assert table_units[0][0] == "table_unit"
            table_id = table_units[0][1]
            table_name = table_data["table_names_original"][table_id]
            grnd = GroundingKey.make_table_grounding(table_name)
        else:
            agg_op, grnd = None, None

    return agg_op, grnd


def add_grounding_with_check(grounding_index, grounding_key, grounding_out):
    if grounding_key is not None:
        if grounding_index in grounding_out:
            assert grounding_key in grounding_out[grounding_index], f"Trying to ground {grounding_index} to {grounding_key}, but have {grounding_out[grounding_index]}"
        else:
            grounding_out[grounding_index] = [grounding_key]


def parse_spider_val_unit(val_unit, sql_unit, table_data):
    unit_op, col_unit1, col_unit2 = val_unit
    unit_op = UNIT_OPS[unit_op]

    assert unit_op == "none" and col_unit2 is None, "Do not support ops in the SQL output for now - will need this for ARITHMETIC only"
    return parse_spider_col_unit(col_unit1, sql_unit, table_data)


def ground_select(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "select"
    args = qdmr.args[i_op]

    assert len(head_grounding) == 1, f"SELECT can produce only one output but have {head_grounding}"

    if head_grounding[0][1] is None:
        return

    assert head_grounding[0][0] == "none", f"SELECT with args {args} has {head_grounding[0][1]} aggregator from SQL"
    grnd = head_grounding[0][1]

    i_arg_distinct = 0

    has_distinct = any(w in args[i_arg_distinct].lower() for w in keywords_distinct)
    content_str = remove_stop_words(args[i_arg_distinct], keywords_distinct + stop_words_distinct)
    if has_distinct:
        add_distinct_grounding(i_op, grounding_out)

    if has_distinct and not content_str:
        # add empty grounding to the current op
        add_grounding_with_check(GroundingIndex(i_op, i_arg_distinct, args[i_arg_distinct]),
                                 "",
                                 grounding_out)
    else:
        add_grounding_with_check(GroundingIndex(i_op, i_arg_distinct, args[i_arg_distinct]),
                                 grnd,
                                 grounding_out)


def ground_filter(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "filter"
    args = qdmr.args[i_op]

    i_arg_source = 0
    i_arg_distinct = 1

    has_distinct = any(w in args[i_arg_distinct].lower() for w in keywords_distinct)
    content_str = remove_stop_words(args[i_arg_distinct], keywords_distinct + stop_words_distinct)
    if has_distinct:
        add_distinct_grounding(i_op, grounding_out)

    if has_distinct and not content_str:
        # add empty grounding to the current op
        add_grounding_with_check(GroundingIndex(i_op, i_arg_distinct, args[i_arg_distinct]),
                                 "",
                                 grounding_out)

    i_op_source = QdmrInstance.ref_to_index(args[i_arg_source], max_index=i_op)
    op_grounder[qdmr.ops[i_op_source]](head_grounding, sql_unit, i_op_source, qdmr, sql_spider_data, table_data, grounding_out)

    # can we say something from sql_unit["where"]


def remove_stop_words(s, stop_words):
    words = s.split(" ")
    stop_words_lower = [w.lower() for w in stop_words]
    words = [w for w in words if w.lower() not in stop_words_lower]
    return " ".join(words)


def add_distinct_grounding(i_op, grounding):
    if "distinct" not in grounding:
        grounding["distinct"] = []
    key = f"#{i_op + 1}"
    if key not in grounding["distinct"]:
        grounding["distinct"].append(key)


def ground_project(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "project"
    args = qdmr.args[i_op]
    i_arg = 0
    i_arg_source = 1

    assert len(head_grounding) == 1, f"PROJECT can produce only one output but have {head_grounding}"
    assert head_grounding[0][0] == "none", f"PROJECT with args {args} has {head_grounding[0][1]} aggregator from SQL"
    grnd = head_grounding[0][1]

    has_distinct = any(w in args[i_arg].lower() for w in keywords_distinct)
    content_str = remove_stop_words(args[i_arg], keywords_distinct + stop_words_distinct)
    if has_distinct:
        add_distinct_grounding(i_op, grounding_out)

    if has_distinct and not content_str:
        # with have project with distinct only - pass the grounding further
        i_op_source = QdmrInstance.ref_to_index(args[i_arg_source], max_index=i_op)
        op_grounder[qdmr.ops[i_op_source]](head_grounding, sql_unit, i_op_source, qdmr, sql_spider_data, table_data, grounding_out)

        # add empty grounding to the current op
        add_grounding_with_check(GroundingIndex(i_op, i_arg, args[i_arg]),
                                 "",
                                 grounding_out)
    else:
        add_grounding_with_check(GroundingIndex(i_op, i_arg, args[i_arg]),
                                 grnd,
                                 grounding_out)

    # do not know how to propagate grounding  further through to args[1] - can be anything


def ground_union(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "union"
    args = qdmr.args[i_op]
    if len(head_grounding) == len(args):
        for grnd, arg in zip(head_grounding, args):
            i_op_arg = QdmrInstance.ref_to_index(arg, max_index=i_op)
            op_grounder[qdmr.ops[i_op_arg]]([grnd], sql_unit, i_op_arg, qdmr, sql_spider_data, table_data, grounding_out)
    else:
        if len(head_grounding) == 1:
            # we have vertival union
            assert head_grounding[0][0] == "none", "Non none aggregator in vertical grounding"
            grnd = ("none", head_grounding[0][1])
            for arg in args:
                i_op_arg = QdmrInstance.ref_to_index(arg, max_index=i_op)
                op_grounder[qdmr.ops[i_op_arg]]([grnd], sql_unit, i_op_arg, qdmr, sql_spider_data, table_data, grounding_out)
        else:
            raise NotImplementedError("Unknown type of union op")


def ground_intersection(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "intersection"
    args = qdmr.args[i_op]

    for arg in [args[0]]: # only the first arg of intersection has to be grounded to the same place as the output
        i_op_arg = QdmrInstance.ref_to_index(arg, max_index=i_op)
        op_grounder[qdmr.ops[i_op_arg]](head_grounding, sql_unit, i_op_arg, qdmr, sql_spider_data, table_data, grounding_out)


def ground_sort(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "sort"
    if len(qdmr.args[i_op]) == 3:
        data_arg, sort_arg, sort_dir_arg = qdmr.args[i_op]
    else:
        data_arg, sort_arg = qdmr.args[i_op]
        sort_dir_arg = None

    # deal with data argument
    i_op_data_arg = QdmrInstance.ref_to_index(data_arg, max_index=i_op)
    op_grounder[qdmr.ops[i_op_data_arg]](head_grounding, sql_unit, i_op_data_arg, qdmr, sql_spider_data, table_data, grounding_out)

    # deal with the sort argument
    if sql_unit["orderBy"]:
        sort_dir, sort_args = sql_unit['orderBy']
        if sort_dir.lower() == 'asc':
            is_ascending_sort = True
        elif sort_dir.lower() == 'desc':
            is_ascending_sort = False
        else:
            raise RuntimeError(f"Unknown order {sort_dir}")

        if sort_dir_arg is not None:
            add_grounding_with_check(GroundingIndex(i_op, 2, sort_dir_arg),
                                     GroundingKey.make_sortdir_grounding(is_ascending_sort),
                                     grounding_out)
        else:
            assert sort_dir == "asc", f"sort_dir_arg is not given, but sort is not default: asc"

        i_op_sort_arg = QdmrInstance.ref_to_index(sort_arg, max_index=i_op)

        sort_arg_grnds = []
        for val_unit in sort_args:
            grnd = parse_spider_val_unit(val_unit, sql_unit, table_data)
            if grnd[0] == "none":
                sort_arg_grnds.append(grnd)

        if sort_arg_grnds:
            op_grounder[qdmr.ops[i_op_sort_arg]](sort_arg_grnds, sql_unit, i_op_sort_arg, qdmr, sql_spider_data, table_data, grounding_out)
    else:
        raise RuntimeError(f"SORT op: should have orderBy in the corresponding SQL unit")


def ground_comparative(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "comparative"
    args = qdmr.args[i_op]

    i_arg_source = 0
    i_arg_distinct = 1

    has_distinct = any(w in args[i_arg_distinct].lower() for w in keywords_distinct)
    content_str = remove_stop_words(args[i_arg_distinct], keywords_distinct + stop_words_distinct)
    if has_distinct:
        add_distinct_grounding(i_op, grounding_out)

    if has_distinct and not content_str:
        # add empty grounding to the current op
        add_grounding_with_check(GroundingIndex(i_op, i_arg_distinct, args[i_arg_distinct]),
                                 "",
                                 grounding_out)

    i_op_source = QdmrInstance.ref_to_index(args[i_arg_source], max_index=i_op)
    op_grounder[qdmr.ops[i_op_source]](head_grounding, sql_unit, i_op_source, qdmr, sql_spider_data, table_data, grounding_out)

    # can we say something from sql_unit["where"]


def ground_superlative(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "superlative"
    args = qdmr.args[i_op]

    add_grounding_with_check(GroundingIndex(i_op, 0, args[0]),
                             GroundingKey.make_comparative_grounding(args[0], "None"),
                             grounding_out)

    i_op_source = QdmrInstance.ref_to_index(args[1], max_index=i_op)
    op_grounder[qdmr.ops[i_op_source]](head_grounding, sql_unit, i_op_source, qdmr, sql_spider_data, table_data, grounding_out)

    # can we say something from sql_unit["where"]


def ground_aggregate(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "aggregate"
    args = qdmr.args[i_op]

    i_op_arg = QdmrInstance.ref_to_index(args[1], max_index=i_op)

    all_args_matches = True
    grnd_list = []
    for grnd in head_grounding:
        if args[0] == grnd[0]:
            assert args[0] == grnd[0], f"AGGREGATE: Have aggregator {grnd[0]} in SQL but {args[0]} in QDMR"
            # removing the aggregator
            grnd = ("none", grnd[1])
            grnd_list.append(grnd) 
        else:
            all_args_matches = False
    
    if all_args_matches:
        op_grounder[qdmr.ops[i_op_arg]](grnd_list, sql_unit, i_op_arg, qdmr, sql_spider_data, table_data, grounding_out)
    else:
        op_grounder[qdmr.ops[i_op_arg]](head_grounding, sql_unit, i_op_arg, qdmr, sql_spider_data, table_data, grounding_out)


def ground_group(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "group"
    args = qdmr.args[i_op]

    i_op_data = QdmrInstance.ref_to_index(args[1], max_index=i_op)
    i_op_index = QdmrInstance.ref_to_index(args[2], max_index=i_op)

    if len(head_grounding) == 1:
        if sql_unit["groupBy"]:
            assert args[0] == head_grounding[0][0], f"GROUP: Have aggregator {head_grounding[0][0]} in SQL but {args[0]} in QDMR"
            # removing the aggregator
            grnd = ("none", head_grounding[0][1])

            op_grounder[qdmr.ops[i_op_data]]([grnd], sql_unit, i_op_data, qdmr, sql_spider_data, table_data, grounding_out)

            grnds = [parse_spider_col_unit(col_unit, sql_unit, table_data) for col_unit in sql_unit["groupBy"]]
            try:
                op_grounder[qdmr.ops[i_op_index]](grnds, sql_unit, i_op_index, qdmr, sql_spider_data, table_data, grounding_out)
            except:
                pass
        else:
            op_grounder[qdmr.ops[i_op_data]](head_grounding, sql_unit, i_op_data, qdmr, sql_spider_data, table_data, grounding_out)
    else:
        if sql_unit["groupBy"]:
            grnds = [parse_spider_col_unit(col_unit, sql_unit, table_data)[1] for col_unit in sql_unit["groupBy"]]
            assert len(grnds) == 1
            assert head_grounding[0] == grnds[0], f"GroupBy args {grnds} do not match groundings {head_grounding}"

            op_grounder[qdmr.ops[i_op_data]]([head_grounding[1]], sql_unit, i_op_data, qdmr, sql_spider_data, table_data, grounding_out)
            op_grounder[qdmr.ops[i_op_index]](grnds, sql_unit, i_op_index, qdmr, sql_spider_data, table_data, grounding_out)
        # else:
        #     raise RuntimeError("Do not see the groupBy field for the GROUP op")


def ground_discard(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    assert qdmr.ops[i_op] == "discard"
    args = qdmr.args[i_op]

    for arg in args:
        i_op_arg = QdmrInstance.ref_to_index(arg, max_index=i_op)
        op_grounder[qdmr.ops[i_op_arg]](head_grounding, sql_unit, i_op_arg, qdmr, sql_spider_data, table_data, grounding_out)

    # can we say something from sql_unit["where"]


def ground_arithmetic(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    raise NotImplementedError()


def ground_boolean(head_grounding, sql_unit, i_op, qdmr, sql_spider_data, table_data, grounding_out):
    raise NotImplementedError()


op_grounder = {}
op_grounder["select"] = ground_select
op_grounder["filter"] = ground_filter
op_grounder["project"] = ground_project
op_grounder["union"] = ground_union
op_grounder["intersection"] = ground_intersection
op_grounder["sort"] = ground_sort
op_grounder["comparative"] = ground_comparative
op_grounder["superlative"] = ground_superlative
op_grounder["aggregate"] = ground_aggregate
op_grounder["group"] = ground_group
op_grounder["discard"] = ground_discard
op_grounder["arithmetic"] = ground_arithmetic
op_grounder["boolean"] = ground_boolean

op_args_to_ground = {}
op_args_to_ground["select"] = [0]
op_args_to_ground["filter"] = [1]
op_args_to_ground["project"] = [0]
op_args_to_ground["union"] = []
op_args_to_ground["intersection"] = []
op_args_to_ground["sort"] = [2]
op_args_to_ground["comparative"] = [2]
op_args_to_ground["superlative"] = [0]
op_args_to_ground["aggregate"] = []
op_args_to_ground["group"] = []
op_args_to_ground["discard"] = []
op_args_to_ground["arithmetic"] = []
op_args_to_ground["boolean"] = []


def main(args):
    print(args)

    split_name = 'dev' if args.dev else 'train'
    dataset_break = DatasetBreak(args.qdmr_path, split_name)
    dataset_spider = DatasetSpider(args.spider_path, split_name)

    if args.spider_idx is not None:
        qdmr_name = None
        for name in dataset_break.names:
            idx = dataset_break.get_index_from_name(name)
            if idx == args.spider_idx:
                qdmr_name = name
                break

        assert qdmr_name is not None, "Could find QDMR with index {args.spider_idx}"
        qdmr = dataset_break.qdmrs[qdmr_name]

        # debugging
        print()
        print(qdmr_name)

        grounding, _ = compute_grounding(qdmr, qdmr_name, dataset_spider, verbose=True)
    else:
        all_grounding = {}
        for qdmr_name, qdmr in dataset_break:
            spider_idx = DatasetBreak.get_index_from_name(qdmr_name)

            print(qdmr_name, end=" ")
            try:
                grounding, message = compute_grounding(qdmr, qdmr_name, dataset_spider, verbose=False)
                grounding["MESSAGES"] = [message]
                all_grounding[qdmr_name] = grounding
            except Exception as e:
                error_details = handle_exception(e, verbose=False)
                print(f"ERROR: {error_details['type']}:{error_details['message']}, file: {error_details['file']}, line {error_details['line_number']}")
                all_grounding[qdmr_name] = {"ERRORS" : [error_details]}

        if args.output_path:
            # save data to the new file
            save_grounding_to_file(args.output_path, all_grounding)

            # check correctness of save-load
            check = load_grounding_from_file(args.output_path)
            assert_check_grounding_save_load(all_grounding, check)


if __name__ == '__main__':
    args = parse_args()
    main(args)
