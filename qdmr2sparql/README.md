# Log of cleaning the BREAR and SPIDER datasets
The BREAK dataset is stored in `BREAK=$QDMR_ROOT/data/break`.

## Static processing of BREAK entries for SPIDER
We use the logical form files - QDMRs (those are started as all subsets of BREAK logical forms that are built for SPIDER):
* `$BREAK/logical-forms/dev_spider.csv`
* `$BREAK/logical-forms/train_spider.csv`

### Script for static processing
The script `$ROOT/qdmr2sparql/fix_qdmr_static.py` looks only at QDMRs and checks them according to simple rules.

Usage (save new dataset files to `$BREAK/logical-forms-fixed`):
```bash
export QDMR_ROOT=`pwd`
export PYTHONPATH=$QDMR_ROOT:$QDMR_ROOT/qdmr2sparql:$PYTHONPATH
export BREAK=$QDMR_ROOT/data/break
export GRND_PATH=$BREAK/groundings

mkdir -p $BREAK/logical-forms-fixed
mkdir -p $GRND_PATH

python $QDMR_ROOT/qdmr2sparql/fix_qdmr_static.py --qdmr_path $BREAK/logical-forms --dev --output_path $BREAK/logical-forms-fixed/dev_spider.csv

python $QDMR_ROOT/qdmr2sparql/fix_qdmr_static.py --qdmr_path $BREAK/logical-forms --output_path $BREAK/logical-forms-fixed/train_spider.csv
```

## Partial groundings: matching QDMR and SQL to get grounding:
The script `$ROOT/qdmr2sparql/get_qdmr_grounding_from_sql.py` looks at both QDMR and SQL from SPIDER and computes grounding.

Usage:
```bash
python $QDMR_ROOT/qdmr2sparql/get_qdmr_grounding_from_sql.py --dev --qdmr_path $BREAK/logical-forms-fixed --spider_path $QDMR_ROOT/data/spider --output_path $GRND_PATH/grnd_partial_dev.json | tee $GRND_PATH/grnd_partial_dev_log.txt

python $QDMR_ROOT/qdmr2sparql/get_qdmr_grounding_from_sql.py --qdmr_path $BREAK/logical-forms-fixed --spider_path $QDMR_ROOT/data/spider --output_path $GRND_PATH/grnd_partial_train.json | tee $GRND_PATH/grnd_partial_train_log.txt
```

### Correcting SQL queries
SQL queries are corrected in the following files:
* $QDMR_ROOT/data/spider/dev.json - for the dev subset

To reparse the SQL run this command:
```bash
PYTHONPATH=$QDMR_ROOT/spider:$PYTHONPATH python $QDMR_ROOT/qdmr2sparql/parse_raw_json.py
```

## Fixing databases
```bash
python $QDMR_ROOT/qdmr2sparql/fix_databases.py --spider_path $QDMR_ROOT/data/spider
```

## Complete groundings: matching text and search over missing elements
The script `$QDMR_ROOT/qdmr2sparql/get_complete_qdmr_groundings.py` looks at both QDMR and SQL from SPIDER and computes grounding.

Usage (uses partial groundings stored in `$GRND_PATH`):
```bash
python $QDMR_ROOT/qdmr2sparql/get_complete_qdmr_groundings.py --dev --qdmr_path $BREAK/logical-forms-fixed --spider_path $QDMR_ROOT/data/spider --input_grounding $GRND_PATH/grnd_partial_dev.json  --output_path $GRND_PATH/grnd_complete_dev.json | tee $GRND_PATH/grnd_complete_dev_log.txt

# runs < 1 min
python $QDMR_ROOT/qdmr2sparql/get_complete_qdmr_groundings.py --qdmr_path $BREAK/logical-forms-fixed --spider_path $QDMR_ROOT/data/spider --input_grounding $GRND_PATH/grnd_partial_train.json  --output_path $GRND_PATH/grnd_complete_train.json | tee $GRND_PATH/grnd_complete_train_log.txt
```

## Select correct groundings: generating and executing SPARQL scripts
The script `$ROOT/qdmr2sparql/select_grounding_with_sparql.py` looks at both QDMR and SQL from SPIDER and computes grounding.

Usage (uses groundings stored in `$GRND_PATH`):
```bash
python $QDMR_ROOT/qdmr2sparql/select_grounding_with_sparql.py --dev --qdmr_path $BREAK/logical-forms-fixed --spider_path $QDMR_ROOT/data/spider --input_grounding $GRND_PATH/grnd_complete_dev.json --output_path $GRND_PATH/grnd_list_positive_dev.json --output_path_all $GRND_PATH/grnd_list_all_dev.json | tee $GRND_PATH/grnd_list_dev_log.json

# this will run forever: need to run this in parallel
python $QDMR_ROOT/qdmr2sparql/select_grounding_with_sparql.py --qdmr_path $BREAK/logical-forms-fixed --spider_path $QDMR_ROOT/data/spider --input_grounding $GRND_PATH/grnd_complete_train.json --output_path $GRND_PATH/grnd_list_positive_train.json --output_path_all $GRND_PATH/grnd_list_all_train.json | tee $GRND_PATH/grnd_list_train_log.json

# to run this on the HSE cluster in parallel use these scripts:
PYTHONPATH=$QDMR_ROOT/utils/cluster:$PYTHONPATH python $QDMR_ROOT/utils/cluster/launcher_select_groundings_train.py --slurm --num-gpus 0 --num-cpus 1 --timeout 100
# merging results from all jobs:
python $QDMR_ROOT/utils/cluster/launcher_select_groundings_train_collect.py --output_path $GRND_PATH/grnd_list_positive_train.json --output_path_all $GRND_PATH/grnd_list_all_train.json
# Overall, read 4350 positives (63%) and 6921 examples
```

# Process the non-SPIDER part of BREAK
The script `$ROOT/qdmr2sparql/fix_qdmr_static.py` looks only at QDMRs and checks them according to simple rules.

Usage (save new dataset files to `$BREAK/logical-forms-fixed/logical-forms`):
```bash
export QDMR_ROOT=`pwd`
export PYTHONPATH=$QDMR_ROOT:$QDMR_ROOT/qdmr2sparql:$PYTHONPATH

mkdir -p $BREAK/logical-forms-fixed

python $QDMR_ROOT/qdmr2sparql/fix_qdmr_static.py --qdmr_path $BREAK/logical-forms --dev --full_break --output_path $BREAK/logical-forms-fixed/dev.csv

python $QDMR_ROOT/qdmr2sparql/fix_qdmr_static.py --qdmr_path $BREAK/logical-forms --full_break --output_path $BREAK/logical-forms-fixed/train.csv
```

## Text-based groundings
The script `$ROOT/qdmr2sparql/get_qdmr_grounding_from_break.py` looks at both QDMR and SQL from SPIDER and computes grounding.

Usage:
```bash
export GRND_PATH=$BREAK/groundings
mkdir -p $GRND_PATH

python $QDMR_ROOT/qdmr2sparql/get_qdmr_grounding_from_break.py --dev --full_break --qdmr_path $QDMR_ROOT/Break/break_dataset_static_fix --output_path $GRND_PATH/grnd_full_break_dev.json --output_path_all $GRND_PATH/grnd_full_with_errors_break_dev.json | tee $GRND_PATH/grnd_full_break_dev_log.txt

python $QDMR_ROOT/qdmr2sparql/get_qdmr_grounding_from_break.py --full_break --qdmr_path $QDMR_ROOT/Break/break_dataset_static_fix --output_path $GRND_PATH/grnd_full_break_train.json --output_path_all $GRND_PATH/grnd_full_with_errors_break_train.json | tee $GRND_PATH/grnd_full_break_train_log.txt
```
