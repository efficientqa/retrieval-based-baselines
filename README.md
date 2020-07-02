# README for retrieval-based baselines

This repo provides guidelines for training and testing retrieval-based baselines for [NeurIPS Competition on Efficient Open-domain Question Answering](http://efficientqa.github.io/).

We provide two retrieval-based baselines:

- TF-IDF: TF-IDF retrieval built on fixed-length passages, adapted from the [DrQA system's implementation](https://github.com/facebookresearch/DrQA).
- DPR: A learned dense passage retriever, detailed in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) (Karpukhin et al, 2020). Our baseline is adapted from the [original implementation](https://github.com/facebookresearch/DPR).


Note that for both baselines, we use text blocks of 100 words as passages and a BERT-base multi-passage reader. See more details in the [DPR paper](https://arxiv.org/pdf/2004.04906.pdf).

We provide two variants for each model, using (1) full Wikipedia (`full`) and (2) A subset of Wikipedia articles which are found relevant to the questions on the train data (`subset`). In particular, we think of `subset` as a naive way to reduce the disk memory usage for the retrieval-based baselines.


*Note: If you want to try parameter-only baselines (T5-based) for the competition, please note that implementations on T5-based Closed-book QA model is available [here](https://github.com/google-research/google-research/tree/master/t5_closed_book_qa).*

*Note: If you want simple guidelines on making end-to-end QA predictions using pretrained models, please refer to [this tutorial](https://github.com/efficientqa/efficientqa.github.io/blob/master/getting_started.md).*

## Content

1. [Getting ready](#getting-ready)
2. [TFIDF retrieval](#tfidf-retrieval)
3. [DPR retrieval](#dpr-retrieval)
4. [DPR reader](#dpr-reader)
5. [Result](#result)

## Getting ready

### Git clone

```bash
git clone https://github.com/facebookresearch/DPR.git # dependency
git clone https://github.com/efficientqa/retrieval-based-baselines.git # this repo
```

### Download data

Follow [DPR repo][dpr] in order to download NQ data and Wikipedia DB. Specificially, after running `cd DPR` and let `base_dir` as your base directory to store data and pretrained models,


1. Download QA pairs by `python3 data/download_data.py --resource data.retriever.qas --output_dir ${base_dir}` and `python3 data/download_data.py --resource data.retriever.nq --output_dir ${base_dir}`.
2. Download wikipedia DB by `python3 data/download_data.py --resource data.wikipedia_split --output_dir ${base_dir}`.
3. Download gold question-passage pairs by `python3 data/download_data.py --resource data.gold_passages_info --output_dir ${base_dir}`.

Optionally, if you want to try `subset` variant, run `cd ../retrieval-based-baselines; python3 filter_subset_wiki.py --db_path ${base_dir}/data/wikipedia_split/psgs_w100.tsv --data_path ${base_dir}/data/retriever/nq-train.json`. This script will create a new passage DB containing passages which originated articles are those paired with question on the original NQ data (78,050 unique articles; 1,642,855 unique passages).
This new DB will be stored at `${base_dir}/data/wikipedia_split/psgs_w100_subset.tsv`.

From now on, we will denote Wikipedia DBs (either full or subset) as `db_path`.


## TFIDF retrieval

Make sure to be in `retrieval-based-baselines` directory to run scripts for TFIDF (largely adapted from [DrQA repo][drqa]).

**Step 1**: Run `pip install -r requirements.txt`

**Step 2**: Build Sqlite DB via:
```
mkdir -p {base_dir}/tfidf
python3 build_db.py ${db_path} ${base_dir}/tfidf/db.db --num-workers 60`.
```
**Step 3**: Run the following command to build TFIDF index.
```
python3 build_tfidf.py ${base_dir}/tfidf/db.db ${base_dir}/tfidf
```
It will save TF-IDF index in `${base_dir}/tfidf`

**Step 4**: Run inference code to save retrieval results.
```
python3 inference_tfidf.py --qa_file ${base_dir}/data/retriever/qas/nq-{train|dev|test}.csv --db_path ${db_path} --out_file ${base_dir}/tfidf/nq-{train|dev|test}.json --tfidf_path {path_to_tfidf_index}
```

The resulting files, `${base_dir}/tfidf/nq-{train|dev|test}-tfidf.json` are ready to be fed into the DPR reader.

## DPR retrieval

Follow [DPR repo][dpr] to train DPR retriever and make inference. You can follow steps until [Retriever validation](https://github.com/facebookresearch/DPR/tree/master#retriever-validation-against-the-entire-set-of-documents).


If you want to use retriever checkpoint provided by DPR, follow these three steps.

**Step 1**: Make sure to be in `DPR` directory, and download retriever checkpoint by `python3 data/download_data.py --resource checkpoint.retriever.multiset.bert-base-encoder --output_dir ${base_dir}`.

**Step 2**: Save passage vectors by following [Generating representations](https://github.com/facebookresearch/DPR/tree/master#retriever-validation-against-the-entire-set-of-documents). Note that you can replace `ctx_file` to your own `db_path` if you are trying "seen only" version. In particular, you can do
```
python3 generate_dense_embeddings.py --model_file ${base_dir}/checkpoint/retriever/multiset/bert-base-encoder.cp --ctx_file ${db_path} --shard_id {0-19} --num_shards 20 --out_file ${base_dir}/dpr_ctx
```

**Step 3**: Save retrieval results by following [Retriever validation](https://github.com/facebookresearch/DPR/tree/master#retriever-validation-against-the-entire-set-of-documents). In particular, you can do
```
mkdir -p ${base_dir}/dpr_retrieval
python3 dense_retriever.py \
  --model_file ${base_dir}/checkpoint/retriever/single/nq/bert-base-encoder.cp \
  --ctx_file  ${dp_path} \
  --qa_file ${base_dir}/data/retriever/qas/nq-{train|dev|test}.csv \
  --encoded_ctx_file ${base_dir}/'dpr_ctx*' \
  --out_file ${base_dir}/dpr_retrieval/nq-{train|dev|test}.json \
  --n-docs 200 \
  --save_or_load_index # this to save the dense index if it was built for the first time, and load it next times.
```

Now, `${base_dir}/dpr_retrieval/nq-{train|dev|test}.json` is ready to be fed into DPR reader.

## DPR reader

*Note*: The following instruction is identical to instructions from [DPR README](https://github.com/facebookresearch/DPR#optional-reader-model-input-data-pre-processing), but we rewrite it with hyperparamters specified for our baselines.

The following instruction is for training the reader using TFIDF results, saved in `${base_dir}/tfidf/nq-{train|dev|test}-tfidf.json`. In order to use DPR retrieval results, simply replace paths to these files to `${base_dir}/dpr_retrieval/nq-{train|dev|test}.json`

**Step 1**: Preprocess data.

```
python3 preprocess_reader_data.py \
  --retriever_results ${base_dir}/tfidf/nq-{train|dev|test}.json \
  --gold_passages ${base_dir}/data/gold_passages_info/nq_{train|dev|test}.json \
  --do_lower_case \
  --pretrained_model_cfg bert-base-uncased \
  --encoder_model_type hf_bert \
  --out_file ${base_dir}/tfidf/nq-{train|dev|test}-tfidf \
  --is_train_set # specify this only when it is train data
```

**Step 2**: Train the reader.
```
python3 train_reader.py \
        --encoder_model_type hf_bert \
        --pretrained_model_cfg bert-base-uncased \
        --train_file ${base_dir}/tfidf/'nq-train*.pkl' \
        --dev_file ${base_dir}/tfidf/'nq-dev*.pkl' \
        --output_dir ${base_dir}/checkpoints/reader_from_tfidf \
        --seed 42 \
        --learning_rate 1e-5 \
        --eval_step 2000 \
        --eval_top_docs 50 \
        --warmup_steps 0 \
        --sequence_length 350 \
        --batch_size 16 \
        --passages_per_question 24 \
        --num_train_epochs 100000 \
        --dev_batch_size 72 \
        --passages_per_question_predict 50
```

**Step 3**: Test the reader.
```
python train_reader.py \
  --prediction_results_file ${base_dir}/checkpoints/reader_from_tfidf/dev_predictions.json \
  --eval_top_docs 10 20 40 50 80 100 \
  --dev_file ${base_dir}/tfidf/`nq-dev*.pkl` \
  --model_file ${base_dir}/checkpoints/reader_from_tfidf/{checkpoint file} \
  --dev_batch_size 80 \
  --passages_per_question_predict 100 \
  --sequence_length 350
```

[drqa]: https://github.com/facebookresearch/DrQA/
[dpr]: https://github.com/facebookresearch/DPR

## Result

|Model|Exact Mach|Disk usage (gb)|
|---|---|---|
|TFIDF-full|32.0|20.1|
|TFIDF-subset|31.0|2.8|
|DPR-full|41.0|66.4|
|DPR-subset|34.8|5.9|


