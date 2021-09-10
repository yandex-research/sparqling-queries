import argparse
import re
import copy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from qdmr2sparql.datasets import DatasetBreak
from qdmr2sparql.structures import QdmrInstance


COMPARATIVE_OPS = ('=', '>', '<', '>=', '<=', '!=', 'like')
SUPERLATIVE_OPS = ('min', 'max')
AGG_OPS = ('max', 'min', 'count', 'sum', 'avg')
ARITHMETIC_OPS = ('sum', 'difference', 'division', 'multiplication')


def parse_args():
    parser = argparse.ArgumentParser(description='Check correctness of QDMR annotation.')
    parser.add_argument('--dev', action='store_true', help='if true, use dev, else use train')
    parser.add_argument('--start_break_idx', type=int, help='index of the first BREAK example')
    parser.add_argument('--end_break_idx', type=int, help='index of the last Break example')
    parser.add_argument('--unique_grounding', action='store_true', help='allow only unique grounding')
    parser.add_argument('--full_break', action='store_true', help='if true, use all datasets, not only spider')
    parser.add_argument('--break_idx', type=int, help='index of the BREAK example, use only for debugging')
    parser.add_argument('--qdmr_path', type=str, help='path to the BREAK dataset')
    parser.add_argument('--output_path', type=str, default=None, help='file to save the fixed QDMR ops')

    args = parser.parse_args()
    return args


def check_example(qdmr, qdmr_name, question, verbose=True):
    qdmr = copy.deepcopy(qdmr)

    if verbose:
        print("Question:", question)
        print(f"QDMR:\n{qdmr}")

    op_checker = {}
    op_checker["select"] = check_select
    op_checker["filter"] = check_filter
    op_checker["project"] = check_project
    op_checker["union"] = check_union
    op_checker["intersection"] = check_intersection
    op_checker["sort"] = check_sort
    op_checker["comparative"] = check_comparative
    op_checker["superlative"] = check_superlative
    op_checker["aggregate"] = check_aggregate
    op_checker["group"] = check_group
    op_checker["discard"] = check_discard
    op_checker["arithmetic"] = check_arithmetic
    op_checker["boolean"] = check_boolean

    max_num_runs = 20
    cur_run = 0
    run_again = True
    all_ok = True
    something_corrected = False
    change_stage = 0
    max_change_stage = 1

    while cur_run < max_num_runs and (run_again or (change_stage <= max_change_stage )):
        cur_run += 1
        run_again = False
        all_ok = True

        for i_op, (qdmr_op, qdmr_args) in enumerate(qdmr):

            if qdmr_op in op_checker:
                op_ok, corrected = op_checker[qdmr_op](qdmr_args, i_op, qdmr, change_stage)
                all_ok = all_ok and op_ok
                if not op_ok:
                    if corrected is None:
                        print(f"{qdmr_name}: broken op #{i_op}: {qdmr.step_to_str(i_op)}")
                    else:
                        print(f"{qdmr_name}: correcting op #{i_op}: {qdmr.step_to_str(i_op)}")
                        qdmr = corrected
                        run_again = True
                        something_corrected = True
                        break
            else:
                print(f"{qdmr_name}: unrecognized op #{i_op}: {qdmr.step_to_str(i_op)}")

        if not run_again:
            change_stage += 1

    if verbose and something_corrected:
        print(f"Corrected QDMR:\n{qdmr}")

    return qdmr


def check_select(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    ok = ok and len(qdmr_args) == 1
    return ok, None


BETWEEN_RE_PATTERN = r"(.*?)between\s(.*?)\sand\s(.*?)$"

def check_filter(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    corrected = None
    ok = ok and len(qdmr_args) == 2
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[0], i_op)
    # check for the "between - and" construction in the argument
    matches = re.findall(BETWEEN_RE_PATTERN, qdmr_args[1], flags=re.IGNORECASE)
    if matches:
        ok = False
        group = matches[0]

        corrected = insert_qdmr_op("filter", [qdmr_args[0], " ".join([group[0].strip(), "betweenleftside", group[1].strip()]).strip()], i_op, qdmr)
        corrected.ops[i_op+1] = "filter"
        corrected.args[i_op+1] = [QdmrInstance.index_to_ref(i_op), " ".join([group[0].strip(), "betweenrightside", group[2].strip()]).strip()]

    return ok, corrected


def check_project(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    corrected = None
    ok = ok and len(qdmr_args) == 2
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)
    if ok and change_stage == 1:
        op_for_keyword = {}
        op_for_keyword["number of "] = "count"
        op_for_keyword["the number of "] = "count"

        for bait, op in [("average", "avg"), ("total", "sum"), ("suml", "sum"), ("minimum", "min"), ("maximum", "max")]:
            op_for_keyword[f"{bait} of "] = op
            op_for_keyword[f"the {bait} of "] = op
            op_for_keyword[f"{bait} "] = op
            op_for_keyword[f"the {bait} "] = op

        for keyword in op_for_keyword:
            if qdmr_args[0].startswith(keyword):
                ok = False

                corrected = insert_qdmr_op("group", [op_for_keyword[keyword], QdmrInstance.index_to_ref(i_op), qdmr_args[1]], i_op + 1, qdmr)
                corrected.args[i_op][0] = qdmr_args[0][len(keyword):]

                # change all the REFs from project to group
                change_qdmr_indices(corrected.args[i_op+2:], QdmrInstance.index_to_ref(i_op), QdmrInstance.index_to_ref(i_op + 1))

                # apply only one key word at a time
                break

    return ok, corrected


def change_qdmr_indices(qdmr_args_all_ops, src, tgt):
    for args in qdmr_args_all_ops:
        for i_arg, arg in enumerate(args):
            if arg == src:
                args[i_arg] = tgt


def check_union(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    for arg in qdmr_args:
        ok = ok and QdmrInstance.is_good_qdmr_ref(arg, i_op)
    return ok, None


def extract_content_str(s):
    text_tokens = word_tokenize(s)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    content_str = " ".join(tokens_without_sw)
    return content_str


def check_intersection(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    corrected = None
    ok = ok and len(qdmr_args) == 3
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[2], i_op)

    if ok and not QdmrInstance.is_good_qdmr_ref(qdmr_args[0], i_op):
        ok = False

        refs = QdmrInstance.find_qdmr_refs_in_str(qdmr_args[0])
        if not refs:
            corrected = insert_qdmr_op("select", [qdmr_args[0]], i_op, qdmr)
            corrected.args[i_op + 1][0] = QdmrInstance.index_to_ref(i_op)
        elif len(refs) == 1:
            ref = refs[0]
            content_str = extract_content_str(qdmr_args[0]).replace("# ", "#")
            if QdmrInstance.is_good_qdmr_ref(content_str, i_op):
                corrected = copy.deepcopy(qdmr)
                corrected.args[i_op][0] = content_str
            else:
                text = qdmr_args[0].replace(ref, "#REF")
                corrected = insert_qdmr_op("filter", [ref, text], i_op, qdmr)
                corrected.args[i_op + 1][0] = QdmrInstance.index_to_ref(i_op)

    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[0], i_op)
    return ok, corrected


def check_sort(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    corrected = None
    ok = ok and (len(qdmr_args) == 2 or len(qdmr_args) == 3)
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[0], i_op)
    target_arg_text = copy.deepcopy(qdmr_args[0])
    if ok and len(qdmr_args) == 2:
        ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)
        if not ok:
            sort_arg = qdmr_args[1].replace("'", "")

            list_of_refs, ref_spans = QdmrInstance.find_qdmr_refs_in_str(sort_arg, return_positions=True)
            if len(list_of_refs) < 1:
                index_arg_text = target_arg_text
                sort_dir_arg = sort_arg.strip()
            else:
                span = ref_spans[0]
                index_arg_text = list_of_refs[0]
                sort_dir_arg = (qdmr_args[1][:span[0]] + qdmr_args[1][span[1]:]).strip()

            corrected = copy.deepcopy(qdmr)

            corrected.args[i_op] = [target_arg_text, index_arg_text, sort_dir_arg]
        pass
    elif ok and len(qdmr_args) == 3:
        ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)
        # check that there is no QDMR ref in the third arg
        list_of_refs = QdmrInstance.find_qdmr_refs_in_str(qdmr_args[2])
        ok = ok and len(list_of_refs) == 0

    return ok, corrected


def check_comparative(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    corrected = None
    ok = ok and len(qdmr_args) == 3
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[0], i_op)
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)

    # check for the "between - and" construction in the argument
    matches = re.findall(BETWEEN_RE_PATTERN, qdmr_args[2], flags=re.IGNORECASE)
    if matches:
        ok = False
        group = matches[0]

        corrected = insert_qdmr_op("comparative", [qdmr_args[0], qdmr_args[1], " ".join([group[0], "betweenleftside", group[1]])], i_op, qdmr)
        corrected.ops[i_op+1] = "comparative"
        corrected.args[i_op+1] = [QdmrInstance.index_to_ref(i_op), qdmr_args[1], " ".join([group[0], "betweenrightside", group[2]])]

    return ok, corrected


def check_superlative(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    ok = ok and qdmr_args[0] in SUPERLATIVE_OPS
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[2], i_op)
    return ok, None


def check_aggregate(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    ok = ok and len(qdmr_args) == 2
    ok = ok and qdmr_args[0].replace("'", "") in AGG_OPS
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)
    return ok, None


def insert_qdmr_op(op, args, i_op, qdmr, change_stage=0):
    corrected = copy.deepcopy(qdmr)
    corrected.ops.insert(i_op, op)
    corrected.args.insert(i_op, args)
    # increment all QDMR indices to elements after the insertion
    # assuming that QDMR commands refers only to the previous indices
    for args in corrected.args[i_op+1:]:
        for i_arg, arg in enumerate(args):
            list_of_refs, ref_spans = QdmrInstance.find_qdmr_refs_in_str(arg, return_positions=True)
            if list_of_refs:
                arg_new = arg[:ref_spans[0][0]]
                for i_span, (ref, span) in enumerate(zip(list_of_refs, ref_spans)):
                    idx = QdmrInstance.ref_to_index(ref)
                    if idx >= i_op:
                        idx += 2 # adding 2 because QdmrInstance.ref_to_index returns a 0-based index
                        ref = QdmrInstance.index_to_ref(idx - 1)

                    arg_new += ref
                    if i_span < len(ref_spans) - 1:
                        arg_new += arg[span[1]:ref_spans[i_span+1][0]]
                    else:
                        arg_new += arg[span[1]:]
                args[i_arg] = arg_new

    return corrected


def delete_qdmr_op(i_op_to_delete, qdmr):
    # remove op
    del qdmr.ops[i_op_to_delete]
    del qdmr.args[i_op_to_delete]

    # fix all indices after the removed item
    for i_op_tmp in range(i_op_to_delete, len(qdmr)):
        args_tmp = qdmr.args[i_op_tmp]
        for i_arg, a in enumerate(args_tmp):
            if QdmrInstance.is_good_qdmr_ref(a, i_op_tmp + 1):
                qdmr_index_tmp = QdmrInstance.ref_to_index(a)
                assert qdmr_index_tmp != i_op_to_delete, "Cannot delete QDMR op when it is used later"
                if qdmr_index_tmp > i_op_to_delete:
                    args_tmp[i_arg] = QdmrInstance.index_to_ref(qdmr_index_tmp - 1) # references after the deleted op are decreased by one


def check_group(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    corrected = None
    ok = ok and len(qdmr_args) == 3
    ok = ok and qdmr_args[0].replace("'", "") in AGG_OPS

    if ok and not QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op) and QdmrInstance.is_good_qdmr_ref(qdmr_args[2], i_op):
        ok = False
        corrected = insert_qdmr_op("project", [qdmr_args[1], qdmr_args[2]], i_op, qdmr)
        corrected.args[i_op + 1][1] = QdmrInstance.index_to_ref(i_op)

    if ok and not QdmrInstance.is_good_qdmr_ref(qdmr_args[2], i_op) and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op):
        ok = False
        corrected = insert_qdmr_op("project", [qdmr_args[2], qdmr_args[1]], i_op, qdmr)
        corrected.args[i_op + 1][2] = QdmrInstance.index_to_ref(i_op)

    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[2], i_op)

    if ok and (i_op + 1 == len(qdmr)):
        # if GROUP is the last op - add a union op after group
        ok = False
        # by default we will output the index var first and the aggregated var second
        corrected = insert_qdmr_op("union", [qdmr_args[2], qdmr_args[1]], i_op + 1, qdmr)
        corrected.args[i_op + 1][1] = QdmrInstance.index_to_ref(i_op)

    return ok, corrected


def check_discard(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    corrected = None
    ok = ok and len(qdmr_args) == 2
    if ok and not QdmrInstance.is_good_qdmr_ref(qdmr_args[0], i_op):
        ok = False
        
        refs = QdmrInstance.find_qdmr_refs_in_str(qdmr_args[0])
        if not refs:
            # insert op with select
            corrected = insert_qdmr_op("select", [qdmr_args[0]], i_op, qdmr)
            corrected.args[i_op + 1][0] = QdmrInstance.index_to_ref(i_op)
        else:
            # insert op with project
            arg_text = qdmr_args[0].replace(refs[0], "#REF")
            corrected = insert_qdmr_op("project", [arg_text, refs[0]], i_op, qdmr)
            corrected.args[i_op + 1][0] = QdmrInstance.index_to_ref(i_op)

    if ok and not QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op) and QdmrInstance.is_good_qdmr_ref(qdmr_args[0], i_op):
        ok = False
        # insert op with filter
        corrected = insert_qdmr_op("filter", [qdmr_args[0], qdmr_args[1]], i_op, qdmr)
        corrected.args[i_op + 1][1] = QdmrInstance.index_to_ref(i_op)

    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[0], i_op)
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)
    return ok, corrected


def check_arithmetic(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    ok = ok and len(qdmr_args) == 3
    ok = ok and qdmr_args[0].replace("'", "") in ARITHMETIC_OPS
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[1], i_op)
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[2], i_op)
    return ok, None


def check_boolean(qdmr_args, i_op, qdmr, change_stage=0):
    ok = True
    ok = ok and len(qdmr_args) == 2
    ok = ok and QdmrInstance.is_good_qdmr_ref(qdmr_args[0], i_op)
    return ok, None


def main(args):
    print(args)

    split_name = 'dev' if args.dev else 'train'
    filter_subset = "spider" if not args.full_break else ""
    dataset_break = DatasetBreak(args.qdmr_path, split_name, filter_subset=filter_subset)

    if args.break_idx is not None:
        qdmr_name = dataset_break.names[args.break_idx]
        qdmr = dataset_break.qdmrs[qdmr_name]
        question = dataset_break.questions[qdmr_name]

        # debugging
        print()
        print(qdmr_name)

        check_example(qdmr, qdmr_name, question, verbose=True)
    else:
        for qdmr_name, qdmr in dataset_break:
            break_idx = DatasetBreak.get_index_from_name(qdmr_name)
            dataset_keyword = DatasetBreak.get_dataset_keyword_from_name(qdmr_name)
            question = dataset_break.get_question_by_subset_indx(break_idx, dataset_keyword)

            try:
                qdmr_corrected = check_example(qdmr, qdmr_name, question, verbose=False)
                dataset_break.qdmrs[qdmr_name] = qdmr_corrected
            except Exception as e:
                print(f"{qdmr_name}: ERROR: {e}")

        if args.output_path:
            # save data to the new file
            dataset_break.save_break_to_csv_file(args.output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)