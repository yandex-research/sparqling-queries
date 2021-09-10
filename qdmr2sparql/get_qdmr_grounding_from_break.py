import os
import argparse
import copy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from qdmr2sparql.grounding_utils import stem_match, measure_similarity

from qdmr2sparql.datasets import DatasetBreak
from qdmr2sparql.structures import GroundingKey, GroundingIndex, QdmrInstance
from qdmr2sparql.structures import save_grounding_to_file, load_grounding_list_from_file, assert_check_grounding_save_load

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
stop_words_distinct = ["#REF"]


def parse_args():
    parser = argparse.ArgumentParser(description='Check correctness of QDMR annotation.')
    parser.add_argument('--dev', action='store_true', help='if true, use dev, else use train')
    parser.add_argument('--break_idx', type=int, help='index of spider example, use only for debugging')
    parser.add_argument('--full_break', action='store_true', help='if true, use all datasets, not only spider')
    parser.add_argument('--qdmr_path', type=str, help='path to break dataset')
    parser.add_argument('--output_path', type=str, default=None, help='path to output file with grounding when parsed correctly')
    parser.add_argument('--output_path_all', type=str, default=None,help='path to output file with grounding including error messages')
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

def make_grounding(qdmr, qdmr_name, dataset_break, verbose=True):

    question = dataset_break.questions[qdmr_name]

    if verbose:
        print("Question:", question)
        print(f"QDMR:\n{qdmr}")

    grounding = {}
    for i_op in range(len(qdmr)):
        op = qdmr.ops[i_op]
        assert op in op_grounder, f"Could not find function to ground op {op}"
        op_grounder[op](i_op, qdmr, grounding)

    message = []

    # count the number of grounded args for printing
    args_grounded = 0
    for i_op, (qdmr_op, qdmr_args) in enumerate(qdmr):
        for i_arg in range(len(qdmr_args)):
            if GroundingIndex(i_op, i_arg, qdmr_args[i_arg]) in grounding:
                args_grounded += 1

    message += [f"{os.path.basename(__file__)}: OK: grounded {args_grounded} args: {grounding}"]
    # message += warning_messages
    for m in message:
        print(m)

    message = "\n".join(message)

    return grounding, message


def add_grounding_with_check(grounding_index, grounding_key, grounding_out):
    if grounding_key is not None:
        if grounding_index in grounding_out:
            assert grounding_key == grounding_out[grounding_index], f"Trying to ground {grounding_index} to {grounding_key}, but have {grounding_out[grounding_index]}"
        else:
            grounding_out[grounding_index] = grounding_key


def ground_null(i_op, qdmr, grounding_out):
    pass


def ground_select_project(i_op, qdmr, grounding_out):
    assert qdmr.ops[i_op] in ["select", "project"]
    args = qdmr.args[i_op]

    i_arg_distinct = 0
    text_arg = args[i_arg_distinct]

    content_str, has_distinct, tokens_without_sw = extract_distinct_and_content(text_arg)

    if has_distinct:
        add_distinct_grounding(i_op, grounding_out)

    if has_distinct and not content_str:
        # add empty grounding to the current op
        add_grounding_with_check(GroundingIndex(i_op, i_arg_distinct, text_arg),
                                 "",
                                 grounding_out)
    else:
        if not content_str and qdmr.ops[i_op] == "select":
            content_str, has_distinct, tokens_without_sw = extract_distinct_and_content(text_arg, remove_stopwords=False)
        grnd = GroundingKey.make_text_grounding(content_str)

        add_grounding_with_check(GroundingIndex(i_op, i_arg_distinct, text_arg),
                                 grnd,
                                 grounding_out)

COMPARATIVE_OP_KEYWORDS = {}
COMPARATIVE_OP_KEYWORDS["larger than or equal to"] = ">="
COMPARATIVE_OP_KEYWORDS["smaller than or equal to"] = "<="
COMPARATIVE_OP_KEYWORDS["at least"] = ">="
COMPARATIVE_OP_KEYWORDS["at most"] = "<="

# special words inserted in fix_qdmr_static.py
COMPARATIVE_OP_KEYWORDS["betweenleftside"] = ">="
COMPARATIVE_OP_KEYWORDS["betweenrightside"] = "<="

COMPARATIVE_OP_KEYWORDS["lower"] = "<"
COMPARATIVE_OP_KEYWORDS["less"] = "<"
COMPARATIVE_OP_KEYWORDS["smaller"] = "<"
COMPARATIVE_OP_KEYWORDS["fewer"] = "<"
COMPARATIVE_OP_KEYWORDS["worse"] = "<"
COMPARATIVE_OP_KEYWORDS["shorter"] = "<"
COMPARATIVE_OP_KEYWORDS["under"] = "<"
COMPARATIVE_OP_KEYWORDS["before"] = "<"
COMPARATIVE_OP_KEYWORDS["larger"] = ">"
COMPARATIVE_OP_KEYWORDS["bigger"] = ">"
COMPARATIVE_OP_KEYWORDS["higher"] = ">"
COMPARATIVE_OP_KEYWORDS["greater"] = ">"
COMPARATIVE_OP_KEYWORDS["longer"] = ">"
COMPARATIVE_OP_KEYWORDS["better"] = ">"
COMPARATIVE_OP_KEYWORDS["more"] = ">"
COMPARATIVE_OP_KEYWORDS["over"] = ">"
COMPARATIVE_OP_KEYWORDS["after"] = ">"
COMPARATIVE_OP_KEYWORDS["equal"] = "="
COMPARATIVE_OP_KEYWORDS["equals"] = "="

COMPARATIVE_OP_KEYWORDS["younger"] = "UNK"
COMPARATIVE_OP_KEYWORDS["older"] = "UNK"
COMPARATIVE_OP_KEYWORDS["earlier"] = "UNK"
COMPARATIVE_OP_KEYWORDS["later"] = "UNK"
COMPARATIVE_OP_KEYWORDS["further"] = "UNK"
COMPARATIVE_OP_KEYWORDS["later"] = "UNK"
COMPARATIVE_OP_KEYWORDS["unknown_sign"] = "UNK"

COMPARATIVE_OP_KEYWORDS["biggest"] = "max"
COMPARATIVE_OP_KEYWORDS["highest"] = "max"
COMPARATIVE_OP_KEYWORDS["largest"] = "max"
COMPARATIVE_OP_KEYWORDS["greatest"] = "max"
COMPARATIVE_OP_KEYWORDS["smallest"] = "min"
COMPARATIVE_OP_KEYWORDS["fewest"] = "min"
COMPARATIVE_OP_KEYWORDS["lowest"] = "min"

COMPARATIVE_OP_KEYWORDS["youngest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["oldest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["farthest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["furthest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["cheapest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["closest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["earliest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["latest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["shortest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["nearest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["sparsest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["narrowest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["darkest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["stingiest"] = "min/max"
COMPARATIVE_OP_KEYWORDS["best"] = "min/max"
COMPARATIVE_OP_KEYWORDS["worst"] = "min/max"


def ground_filter_comparative(i_op, qdmr, grounding_out):
    assert qdmr.ops[i_op] in ["filter", "comparative"]
    args = qdmr.args[i_op]

    i_arg_distinct = 1 if qdmr.ops[i_op] == "filter" else 2
    text_arg = args[i_arg_distinct]

    content_str, has_distinct, tokens_without_sw = extract_distinct_and_content(text_arg, keyword_exceptions=list(COMPARATIVE_OP_KEYWORDS.keys()))

    if has_distinct:
        add_distinct_grounding(i_op, grounding_out)

    if has_distinct and not content_str:
        # add empty grounding to the current op
        add_grounding_with_check(GroundingIndex(i_op, i_arg_distinct, text_arg),
                                 "",
                                 grounding_out)
    else:
        if QdmrInstance.is_good_qdmr_ref(content_str):
            grnd = GroundingKey.make_comparative_grounding("=", content_str)
        else:
            grnd = GroundingKey.make_text_grounding(content_str)
            tokens_without_sw = [t.lower() for t in tokens_without_sw]
            for keyword, sign in COMPARATIVE_OP_KEYWORDS.items():
                if keyword in tokens_without_sw or (" " in keyword and keyword in text_arg):
                    if keyword in tokens_without_sw:
                        tokens = copy.deepcopy(tokens_without_sw)
                        tokens.remove(keyword)
                        text_value = " ".join(tokens)
                        text_value = text_value.replace("# ", "#")
                    else:
                        text_value = text_arg[:text_arg.find(keyword)].strip() + " " + text_arg[text_arg.find(keyword) + len(keyword):].strip()
                        text_value, _, _ = extract_distinct_and_content(text_value)

                    if text_value or sign in ["min", "max", "min/max"]:
                        grnd = GroundingKey.make_comparative_grounding(sign, text_value)
                        break

        refs = QdmrInstance.find_qdmr_refs_in_str(text_arg)
        if grnd.isstr() and refs:
            grnd = GroundingKey.make_comparative_grounding(COMPARATIVE_OP_KEYWORDS["unknown_sign"], refs[0])
        if grnd.iscomp():
            refs = QdmrInstance.find_qdmr_refs_in_str(str(grnd.get_val()))
            if refs:
                grnd = GroundingKey.make_comparative_grounding(grnd.keys[0], refs[0])

        add_grounding_with_check(GroundingIndex(i_op, i_arg_distinct, text_arg),
                                 grnd,
                                 grounding_out)


def remove_stop_words(s, stop_words):
    words = s.split(" ")
    stop_words_lower = [w.lower() for w in stop_words]
    words = [w for w in words if w.lower() not in stop_words_lower]
    return " ".join(words)


def extract_distinct_and_content(s, keyword_exceptions=[], remove_stopwords=True):
    has_distinct = any(w in s.lower() for w in keywords_distinct)
    content_str = remove_stop_words(s, keywords_distinct + stop_words_distinct)

    text_tokens = word_tokenize(content_str)
    if remove_stopwords:
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words("english") or word in keyword_exceptions]
    else:
        tokens_without_sw = [word for word in text_tokens]
    content_str = " ".join(tokens_without_sw)
    content_str = content_str.replace("# ", "#")

    return content_str, has_distinct, tokens_without_sw


def add_distinct_grounding(i_op, grounding):
    if "distinct" not in grounding:
        grounding["distinct"] = []
    key = f"#{i_op + 1}"
    if key not in grounding["distinct"]:
        grounding["distinct"].append(key)


def ground_sort(i_op, qdmr, grounding_out):
    assert qdmr.ops[i_op] == "sort"
    if len(qdmr.args[i_op]) == 3:
        data_arg, sort_arg, sort_dir_arg = qdmr.args[i_op]
    else:
        data_arg, sort_arg = qdmr.args[i_op]
        sort_dir_arg = None

    # deal with the sort argument
    if sort_dir_arg is not None:
        is_ascending_sort = "descend" not in sort_dir_arg
        add_grounding_with_check(GroundingIndex(i_op, 2, sort_dir_arg),
                                    GroundingKey.make_sortdir_grounding(is_ascending_sort),
                                    grounding_out)

op_grounder = {}
op_grounder["select"] = ground_select_project
op_grounder["filter"] = ground_filter_comparative
op_grounder["project"] = ground_select_project
op_grounder["union"] = ground_null
op_grounder["intersection"] = ground_null
op_grounder["sort"] = ground_sort
op_grounder["comparative"] = ground_filter_comparative
op_grounder["superlative"] = ground_null
op_grounder["aggregate"] = ground_null
op_grounder["group"] = ground_null
op_grounder["discard"] = ground_null
op_grounder["arithmetic"] = ground_null


def main(args):
    print(args)

    split_name = 'dev' if args.dev else 'train'
    filter_subset = "spider" if not args.full_break else ""
    dataset_break = DatasetBreak(args.qdmr_path, split_name, filter_subset=filter_subset)

    if args.break_idx is not None:
        qdmr_name = dataset_break.names[args.break_idx]
        qdmr = dataset_break.qdmrs[qdmr_name]

        # debugging
        print()
        print(qdmr_name)

        grounding, _ = make_grounding(qdmr, qdmr_name, dataset_break, verbose=True)
    else:
        groundings_all = {}
        groundings_only_positive = {}

        for qdmr_name, qdmr in dataset_break:
            break_idx = DatasetBreak.get_index_from_name(qdmr_name)

            print(qdmr_name, end=" ")
            try:
                grounding, message = make_grounding(qdmr, qdmr_name, dataset_break, verbose=False)
                groundings_all[qdmr_name] = {"GROUNDINGS": [grounding]}
                groundings_all[qdmr_name]["MESSAGES"] = [message]
                if "OK" in message:
                    groundings_only_positive[qdmr_name] = {"GROUNDINGS": [grounding]}
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


if __name__ == '__main__':
    args = parse_args()
    main(args)
