# SPARQLing Database Queries from Intermediate Question Decompositions

This repo is the implementation of the following paper:

SPARQLing Database Queries from Intermediate Question Decompositions<br>
Irina Saparina and Anton Osokin<br>
To appear in proceedings of EMNLP'21

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

The dev and test examples of model output are `model_samples/`.

Checkpoints of our best models:

| Model name  | Dev | Test | Link |
| ----------- | ----------- | ----------- | ----------- |
| grappa-aug        | **80.4**   | 62.0 | https://www.dropbox.com/s/t9z1uwvohuakig8/grappa-aug_model_checkpoint-00072000?dl=0 |
| grappa-full_break | 74.6   | **62.6** | https://www.dropbox.com/s/bf6vyhtep4knmm7/full-break-grappa_model_checkpoint-00075000?dl=0 |

## Acknowledgements
Text2qdmr module is based on [RAT-SQL code](https://github.com/microsoft/rat-sql), the implementation of ACL'20 paper ["RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers"](https://arxiv.org/abs/1911.04942v5) by Wang et al.

[Spider dataset](https://yale-lily.github.io/spider) was proposed by Yi et al. in EMNLP'18 paper ["Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task"](https://arxiv.org/abs/1809.08887v5).

[Break dataset](https://allenai.github.io/Break/) was proposed by Wolfson et al. in TACL paper ["Break It Down: A Question Understanding Benchmark"](https://arxiv.org/abs/2001.11770v1).
