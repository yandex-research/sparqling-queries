import os
import argparse
from collections import OrderedDict

from structures import save_grounding_to_file, load_grounding_from_file, load_grounding_list_from_file, assert_check_grounding_save_load

# collect results into a single file

# run jobs with this command:
# python $QDMR_ROOT/utils/cluster/launcher_select_groundings_train_collect.py --output_path $GRND_PATH/grnd_list_positive_train.json --output_path_all $GRND_PATH/grnd_list_all_train.json


def parse_args():
    parser = argparse.ArgumentParser(description='Build grounding between QDMR and SQL.')
    parser.add_argument('--output_path', type=str, default=None,help='path to output file with grounding (found correct SPARQL script)')
    parser.add_argument('--output_path_all', type=str, default=None,help='path to output file with grounding')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # load default launcher parameters
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    qdmr_root_path = os.path.abspath(os.path.join(script_dir, "..", ".."))

    grounding_path = os.path.abspath(os.path.join(qdmr_root_path, "data", "break", "groundings"))

    config_job_name = "spider_train"
    main_command = f"python {qdmr_root_path}/qdmr2sparql/select_grounding_with_sparql.py"

    log_path = os.path.abspath(os.path.join(script_dir, "output", "compute_grounding_train"))

    def add_job(start_spider_idx, end_spider_idx):
        job_name = f"{config_job_name}.{start_spider_idx}-{end_spider_idx}"

        d = OrderedDict()

        d["--qdmr_path"] = f"{qdmr_root_path}/data/break/logical-forms-fixed"
        d["--spider_path"] = f"{qdmr_root_path}/data/spider"
        d["--input_grounding"] = f"{grounding_path}/grnd_complete_train.json"
        d["--start_spider_idx"] = start_spider_idx
        d["--end_spider_idx"] = end_spider_idx
        d["--output_path"] = f"{log_path}/grnd_list_positive_train-{start_spider_idx}-{end_spider_idx}.json"
        d["--output_path_all"] = f"{log_path}/grnd_list_all_train-{start_spider_idx}-{end_spider_idx}.json"
        d["--virtuoso_server"] = "http://cn-006:20456/sparql/"


        log_folder = log_path
        log_file_prefix = job_name + "."

        grnd_positive = load_grounding_list_from_file(d["--output_path"])
        grnd_all = load_grounding_list_from_file(d["--output_path_all"])

        return grnd_positive, grnd_all


    # need to process SPIDER_train_0 to SPIDER_train_6999
    groundings_only_positive = {}
    groundings_all = {}

    examples_in_job = 100
    start_spider_idx, end_spider_idx = 0, 7000
    idx = start_spider_idx
    while idx < end_spider_idx:
        job_start_spider_idx, job_end_spider_idx = idx, min(idx + examples_in_job, end_spider_idx)
        idx = idx + examples_in_job

        try:
            grnd_positive, grnd_all = add_job(job_start_spider_idx, job_end_spider_idx)
            groundings_only_positive.update(grnd_positive)
            groundings_all.update(grnd_all)

            print(f"Batch from {job_start_spider_idx} to {job_end_spider_idx}: read {len(grnd_positive)} positives ({len(grnd_positive)/len(grnd_all)*100:.0f}%) and {len(grnd_all)} examples")
        except:
            print(f"ERROR: failed to load batch from {job_start_spider_idx} to {job_end_spider_idx}")

    print(f"Overall, read {len(groundings_only_positive)} positives ({len(groundings_only_positive)/len(groundings_all)*100:.0f}%) and {len(groundings_all)} examples")

    if args.output_path:
        save_grounding_to_file(args.output_path, groundings_only_positive)

        check = load_grounding_list_from_file(args.output_path)
        assert_check_grounding_save_load(groundings_only_positive, check)

    if args.output_path_all:
        save_grounding_to_file(args.output_path_all, groundings_all)

        check = load_grounding_list_from_file(args.output_path_all)
        assert_check_grounding_save_load(groundings_all, check)
