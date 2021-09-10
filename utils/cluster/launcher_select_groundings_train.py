import os
from collections import OrderedDict

import launcher


# run jobs with this command:
# PYTHONPATH=$QDMR_ROOT/utils/cluster:$PYTHONPATH python $QDMR_ROOT/utils/cluster/launcher_select_groundings_train.py --slurm --num-gpus 0 --num-cpus 1 --timeout 100


if __name__ == "__main__":
    # load default launcher parameters
    parser = launcher.create_args_parser()
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    qdmr_root_path = os.path.abspath(os.path.join(script_dir, "..", ".."))

    grounding_path = os.path.abspath(os.path.join(qdmr_root_path, "data", "break", "groundings"))


    config_job_name = "spider_train"
    main_command = f"python {qdmr_root_path}/qdmr2sparql/select_grounding_with_sparql.py"

    log_path = os.path.abspath(os.path.join(script_dir, "output", "compute_grounding_train"))

    exp_job_names = []
    exp_log_paths = []
    exp_commands = []
    exp_log_file_prefix = []

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

        commands = [main_command + " " + launcher.parameters_to_str(d)]

        exp_job_names.append(job_name)
        exp_commands.append(commands)
        exp_log_paths.append(log_folder)
        exp_log_file_prefix.append(log_file_prefix)


    # need to process SPIDER_train_0 to SPIDER_train_6999
    examples_in_job = 100
    start_spider_idx, end_spider_idx = 0, 7000
    idx = start_spider_idx
    while idx < end_spider_idx:
        job_start_spider_idx, job_end_spider_idx = idx, min(idx + examples_in_job, end_spider_idx)
        idx = idx + examples_in_job
        add_job(job_start_spider_idx, job_end_spider_idx)


    for job_name, log_path, commands, log_file_prefix in zip(exp_job_names, exp_log_paths, exp_commands, exp_log_file_prefix):
        launcher.add_job(job_name=job_name,
                         log_path=log_path,
                         commands=commands,
                         log_file_prefix=log_file_prefix)
    launcher.launch_all_jobs(args)

