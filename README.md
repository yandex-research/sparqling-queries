# SPARQLing Database Queries from Intermediate Question Decompositions

This repo is the implementation of the following paper:

SPARQLing Database Queries from Intermediate Question Decompositions<br>
Irina Saparina and Anton Osokin<br>
In proceedings of EMNLP'21

`[31.05.2022]:` **We fixed several bugs in the decoding process, usage of the GraPPa tokenization (affect our model) and SQL-SQL comparison (affect on baselines). The current code reproduces the results from the updated version of the paper.**

## License
This software is released under the [MIT license](./LICENSE), which means that you can use the code in any way you want.

## Dependencies
### Conda env with pytorch 1.9
Create conda env with pytorch 1.9 and many other packages upgraded: [conda_env_with_pytorch1.9.yaml](conda_env_with_pytorch1.9.yaml):
```bash
conda env create -n env-torch1.9 -f conda_env_with_pytorch1.9.yaml
conda activate env-torch1.9
```

Download some nltk resourses, Bert and GraPPa:
``` bash
mkdir -p third_party

pip install -r requirements.txt && \
pip install entmax && \
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-large-uncased-whole-word-masking'); AutoModel.from_pretrained('Salesforce/grappa_large_jnt')"

mkdir -p third_party && \
cd third_party && \
curl https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip | jar xv
```

## Data
We currently provide both Spider and Break inside our repos. Note that datasets differ from original ones as we fixed some annotation errors. 
Download databases:
```bash
bash ./utils/wget_gdrive.sh spider_temp.zip 11icoH_EA-NYb0OrPTdehRWm_d7-DIzWX
unzip spider_temp.zip -d spider_temp
cp -r spider_temp/spider/database ./data/spider
rm -rf spider_temp/
python ./qdmr2sparql/fix_databases.py --spider_path ./data/spider
```

**To reproduce our annotation procedure see [qdmr2sparql/README.md](qdmr2sparql/README.md).**

For testing qdmr2sparql translator run [qdmr2sparql/test_qdmr2sparql.py](qdmr2sparql/test_qdmr2sparql.py)

## Experiments
Every experiment has its own config file in `text2qdmr/configs/experiments`.
The pipeline of working with any model version or dataset is: 

``` bash
python run_text2qdmr.py preprocess experiment_config_file  # preprocess the data
python run_text2qdmr.py train experiment_config_file       # train a model
python run_text2qdmr.py eval experiment_config_file        # evaluate the results

# multiple GPUs on one machine:
export NGPUS=4 # set $NGPUS manually
python -m torch.distributed.launch --nproc_per_node=$NGPUS --use_env --master_port `./utils/get_free_port.sh`  run_text2qdmr.py train experiment_config_file
```

Note that preprocessing and evaluation use execution and take some time. **To speed up the evaluation, you can install Virtuoso server (see [qdmr2sparql/README_Virtuoso.md](qdmr2sparql/README_Virtuoso.md)).**

## Checkpoints and samples

The dev and test examples of model output are in [model_samples/](model_samples/).

Checkpoints of our best models:

| Model name  | Dev | Test | Config | Link |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| grappa-aug        | **82.0**   | 62.4 | [text2qdmr/configs/eval-checkpoints/grappa_qdmr_aug.jsonnet](text2qdmr/configs/eval-checkpoints/grappa_qdmr_aug.jsonnet) | https://drive.google.com/file/d/1xfTxIYlqJ1G-tSrgyI7h20hTE5_we-jy/view?usp=sharing |
| grappa-full_break-aug | 81.6   | **65.3** | [text2qdmr/configs/eval-checkpoints/grappa_full_break_aug.jsonnet](text2qdmr/configs/eval-checkpoints/grappa_full_break_aug.jsonnet)  | https://drive.google.com/file/d/1wwAZGr6d6v_gP_mMcOaUZEk_A1qX2z0H/view?usp=sharing |

To reproduce, firstly download the checkpoints and put them into new folders:
``` bash
mkdir logdir/grappa-aug/bs=6,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,att=1
mv grappa-aug logdir/grappa-aug/bs=6,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,att=1/model_checkpoint-00080000

mkdir logdir/grappa-full_break-aug/bs=6,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,att=1
mv grappa-full_break-aug logdir/grappa-full_break-aug/bs=6,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,att=1/model_checkpoint-00081000
```

Then use the corresponding `config_file` for evaluation:
``` bash
python run_text2qdmr.py preprocess path_to_config_file
python run_text2qdmr.py eval path_to_config_file
```

## Acknowledgements
Text2qdmr module is based on [RAT-SQL code](https://github.com/microsoft/rat-sql), the implementation of ACL'20 paper ["RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers"](https://arxiv.org/abs/1911.04942v5) by Wang et al.

[Spider dataset](https://yale-lily.github.io/spider) was proposed by Yi et al. in EMNLP'18 paper ["Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task"](https://arxiv.org/abs/1809.08887v5).

[Break dataset](https://allenai.github.io/Break/) was proposed by Wolfson et al. in TACL paper ["Break It Down: A Question Understanding Benchmark"](https://arxiv.org/abs/2001.11770v1).
