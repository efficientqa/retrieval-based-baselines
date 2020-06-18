# README for retrieval-based baselines

*Working on progress*

This repo provides guidelines for training and testing retrieval-based baselines for [NeuRIPS Competition on Efficient Open-domain Question Answering](http://efficientqa.github.io/).

We provide tutorials for two retrieval-based baselines.

- DrQA: Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes. [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051). ACL 2017. [[Original implementation][drqa]]
- DPR: Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), Preprint 2020. [[Original implementation][dpr]]

Retrieval-based baselines are composed with two steps.
- **Retrieval** is for retrieving Wikipedia passages that are related to the question. We use DrQA and DPR retrieval as examples of sparse retrieval (TF-IDF) and dense retrieval, respectively.
- **Reader** is for reading retrieved passages to find a text span as an answer to the question. We use DPR reader for both retrieval methods.

All codes are largely based on the original implementation, and we provide command lines to train and test the model specifically for our competition.

*If you want to try parameter-only baselines (T5-based) for the competition, please note that implementations on T5-based Closed-book QA model is available [here](https://github.com/google-research/google-research/tree/master/t5_closed_book_qa).*

*If you want simple guidelines on making end-to-end QA predictions using pretrained models, please refer to [this tutorial](https://github.com/efficientqa/efficientqa.github.io/blob/master/getting_started.md).*

## Content

1. [Getting ready](#getting-ready)
2. [DrQA retrieval](#drqa-retrieval)
3. [DPR retrieval](#dpr-retrieval)
4. [DPR reader](#dpr-reader)

## Getting ready

### Git clone

```bash
git clone https://github.com/facebookresearch/DPR.git # dependency
git clone https://github.com/efficientqa/retrieval-based-baselines.git # this repo
cd retrieval-based-baselines
```

### Download data

Follow [DPR repo][dpr] in order to download NQ data and Wikipedia DB. Specificially,

1. Download QA pairs by `python3 data/download_data.py --resource data.retriever.qas --output_dir {base_dir}`.
2. Download wikipedia DB by `python3 data/download_data.py --resource data.wikipedia_split --output_dir {base_dir} --keep-gzip`.
3. Download gold question-passage pairs by `python3 data/download_data.py --resource data.gold_passages_info --output_dir {base_dir}`.

Optionally, if you want to use Wikipedia articles found from the train data only (78,050 unique articles; 1,642,855 unique passages), run `python3 keep_seen_docs_only --db_path {base_dir}/data/wikipedia_split/psgs_w100.tsv --data_path {base_dir}/data/retriever/qas/nq-train.csv`. This script will save new Wikipedia DB with seen articles at `{base_dir}/data/wikipedia_split/psgs_w100_seen_only.tsv`.

From now on, we will refer Wikipedia DBs (either full or seen only) as `db_path`.


## DrQA retrieval

Scripts for DrQA is largely adapted from [the original DrQA repo][drqa].

**Step 1**: Run `pip install -r requirements.txt`

**Step 2**: Build Sqlite DB via:
```
mkdir -p {base_dir}/drqa_retrieval
python3 build_db.py {db_path} {base_dir}/drqa_retrieval/db.db --num-workers 60`.
```
**Step 3**: Run the following command to build TF-IDF index.
```
python3 build_tfidf.py {base_dir}/drqa_retrieval/db.db {base_dir}/drqa_retrieval
```
It will save TF-IDF index in `{base_dir}/drqa_retrieval`

**Step 4**: Run inference code to save retrieval results.
```
python3 inference_tfidf.py --qa_file {base_dir}/data/retriever/qas/nq-{train|dev|test}.csv --db_path {db_path} --out_file {base_dir}/drqa_retrieval/nq-{train|dev|test}-tfidf.json --tfidf_path {path_to_tfidf_index}
```

The resulting files, `{base_dir}/drqa_retrieval/nq-{train|dev|test}-tfidf.json` are ready to be fed into the DPR reader.

## DPR retrieval

Follow [DPR repo][dpr] to train DPR retriever and make inference. You can follow steps until [Retriever validation](https://github.com/facebookresearch/DPR/tree/master#retriever-validation-against-the-entire-set-of-documents).


If you want to use retriever checkpoint provided by DPR, follow these three steps.

**Step 1**: Download retriever checkpoint by `python3 data/download_data.py --resource checkpoint.retriever.multiset.bert-base-encoder --output_dir {base_dir}`.

**Step 2**: Save passage vectors by following [Generating representations](https://github.com/facebookresearch/DPR/tree/master#retriever-validation-against-the-entire-set-of-documents). Note that you can replace `ctx_file` to your own `db_path` if you are trying "seen only" version. In particular, you can do
```
python3 generate_dense_embeddings.py \
  --model_file {base_dir}/checkpoint/retriever/multiset/bert-base-encoder.cp \
  --ctx_file {db_path} --shard_id {0-19} --num_shards 20 --out_file {base_dir}/dpr_ctx
```

**Step 3**: Save retrieval results by following [Retriever validation](https://github.com/facebookresearch/DPR/tree/master#retriever-validation-against-the-entire-set-of-documents). In particular, you can do
```
mkdir -p {base_dir}/dpr_retrieval
python3 dense_retriever.py \
  --model_file {base_dir}/checkpoint/retriever/multiset/bert-base-encoder.cp \
  --ctx_file  {dp_path} \
  --qa_file {base_dir}/data/retriever/qas/nq-{train|dev|test}.csv \
  --encoded_ctx_file {base_dir}/'dpr_ctx*' \
  --out_file {base_dir}/dpr_retrieval/nq-{train|dev|test}.json \
  --n-docs 200 \
  --save_or_load_index # this to save the dense index if it was built for the first time, and load it next times.
```

Now, `{base_dir}/dpr_retrieval/nq-{train|dev|test}.json` is ready to be fed into DPR reader.

## DPR reader

*Note*: The following instruction is identical to instructions from [DPR README](https://github.com/facebookresearch/DPR#optional-reader-model-input-data-pre-processing), but we rewrite it with hyperparamters specified for our baselines.

The following instruction is for training the reader using DrQA retrieval results, saved in `{base_dir}/drqa_retrieval/nq-{train|dev|test}-tfidf.json`. In order to use DPR retrieval results, simply replace paths to these files to `{base_dir}/dpr_retrieval/nq-{train|dev|test}.json`

**Step 1**: Preprocess data.

```
python3 preprocess_reader_data.py \
  --retriever_results {base_dir}/drqa_retrieval/nq-{train|dev|test}-tfidf.json \
  --gold_passages {base_dir}/data/gold_passages_info/nq_{train|dev|test}.json \
  --do_lower_case \
  --pretrained_model_cfg bert-base-uncased \
  --encoder_model_type hf_bert \
  --out_file {base_dir}/drqa_retrieval/nq-{train|dev|test}-tfidf \
  --is_train_set # specify this only when it is train data
```

**Step 2**: Train the reader.
```
python3 train_reader.py \
        --encoder_model_type hf_bert \
        --pretrained_model_cfg bert-base-uncased \
        --train_file {base_dir}/drqa_retrieval/'nq-train-tfidf*.pkl' \
        --dev_file {base_dir}/drqa_retrieval/'nq-dev-tfidf*.pkl' \
        --output_dir {base_dir}/checkpoints/reader_from_drqa \
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
  --prediction_results_file {base_dir}/checkpoints/reader_from_drqa/dev_predictions.json \
  --eval_top_docs 10 20 40 50 80 100 \
  --dev_file {base_dir}/drqa_retrieval/`nq-dev-tfidf*.pkl` \
  --model_file {base_dir}/checkpoints/reader_from_drqa/{checkpoint file} \
  --dev_batch_size 80 \
  --passages_per_question_predict 100 \
  --sequence_length 350
```

[drqa]: https://github.com/facebookresearch/DrQA/
[dpr]: https://github.com/facebookresearch/DPR




