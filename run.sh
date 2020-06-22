#!/bin/sh

RETRIEVAL=$1 # drqa-full / drqa-seen-only / dpr-full / dpr-seen-only

drqa_index="random"
dpr_index="random"
dpr_retrieval_checkpoint="random"
n_paragraphs="100"
python3 ../DPR/data/download_data.py --resource data.retriever.qas.nq --output_dir ${base_dir}
python3 ../DPR/data/download_data.py --resource data.wikipedia_split --output_dir ${base_dir}

if [ $RETRIEVAL = "drqa-full" ]
then
    python3 ../DPR/data/download_data.py --resource indexes.drqa.nq.full --output_dir ${base_dir} # DrQA index
    python3 ../DPR/data/download_data.py --resource checkpoint.reader.nq-drqa.hf-bert-base --output_dir ${base_dir} # reader checkpoint
    drqa_index="${base_dir}/indexes/drqa/nq/full.npz"
    reader_checkpoint="${base_dir}/checkpoint/reader/nq-drqa/hf-bert-base.cp"
    retrieval_type="drqa"
    db_name="psgs_w100.tsv"
elif [ $RETRIEVAL = "drqa-seen-only" ]
then
    python3 ../DPR/data/download_data.py --resource data.retriever.nq-train --output_dir ${base_dir}
    python3 keep_seen_docs_only.py --db_path ${base_dir}/data/wikipedia_split/psgs_w100.tsv --data_path ${base_dir}/data/retriever/nq-train.json
    python3 ../DPR/data/download_data.py --resource indexes.drqa.nq.seen_only --output_dir ${base_dir} # DrQA index
    python3 ../DPR/data/download_data.py --resource checkpoint.reader.nq-drqa-seen_only.hf-bert-base --output_dir ${base_dir} # reader checkpoint
    drqa_index="${base_dir}/indexes/drqa/nq/seen_only.npz"
    reader_checkpoint="${base_dir}/checkpoint/reader/nq-drqa-seen_only/hf-bert-base.cp"
    retrieval_type="drqa"
    db_name="psgs_w100_seen_only.tsv"
elif [ $RETRIEVAL = "dpr-full" ]
then
    python3 ../DPR/data/download_data.py --resource checkpoint.retriever.single.nq.bert-base-encoder --output_dir ${base_dir} # retrieval checkpoint
    python3 ../DPR/data/download_data.py --resource indexes.single.nq.full --output_dir ${base_dir} # DPR index
    python3 ../DPR/data/download_data.py --resource checkpoint.reader.nq-single.hf-bert-base --output_dir ${base_dir} # reader checkpoint
    dpr_retrieval_checkpoint="${base_dir}/checkpoint/retriever/single/nq/bert-base-encoder.cp"
    dpr_index="${base_dir}/indexes/single/nq/full"
    reader_checkpoint="${base_dir}/checkpoint/reader/nq-single/hf-bert-base.cp"
    retrieval_type="dpr"
    db_name="psgs_w100.tsv"
    n_paragraphs="40"
elif [ $RETRIEVAL = "dpr-seen-only" ]
then
    python3 ../DPR/data/download_data.py --resource data.retriever.nq-train --output_dir ${base_dir}
    python3 keep_seen_docs_only.py --db_path ${base_dir}/data/wikipedia_split/psgs_w100.tsv --data_path ${base_dir}/data/retriever/nq-train.json
    python3 ../DPR/data/download_data.py --resource checkpoint.retriever.single.nq.bert-base-encoder --output_dir ${base_dir} # retrieval checkpoint
    python3 ../DPR/data/download_data.py --resource indexes.single.nq.seen_only --output_dir ${base_dir} # DPR index
    python3 ../DPR/data/download_data.py --resource checkpoint.reader.nq-single-seen_only.hf-bert-base --output_dir ${base_dir} # reader checkpoint
    dpr_retrieval_checkpoint="${base_dir}/checkpoint/retriever/single/nq/bert-base-encoder.cp"
    dpr_index="${base_dir}/indexes/single/nq/seen_only"
    reader_checkpoint="${base_dir}/checkpoint/reader/nq-single-seen_only/hf-bert-base.cp"
    retrieval_type="dpr"
    db_name="psgs_w100_seen_only.tsv"
    n_paragraphs="40"
fi
python3 run_inference.py \
  --qa_file ${base_dir}/data/retriever/qas/nq-test.csv \ # data file with questions
  --retrieval_type ${retrieval_type} \ # which retrieval to use
  --db_path ${base_dir}/data/wikipedia_split/${db_name} \
  --tfidf_path ${drqa_index} \ # only matters for drqa retrieval
  --drqa_model_file ${dpr_retrieval_checkpoint} \ # only matters for dpr retrieval
  --dense_index_path ${dpr_index} \ # only matters for dpr retrieval
  --model_file ${reader_checkpoint} \ # path to the reader checkpoint
  --dev_batch_size 8 \ # 8 is good for one 32gb GPU
  --pretrained_model_cfg bert-base-uncased --encoder_model_type hf_bert --do_lower_case \
  --sequence_length 350 --eval_top_docs 10 20 40 50 80 100 --passages_per_question_predict ${n_paragraphs} \
  --prediction_results_file ${RETRIEVAL}_test_predictions.json # path to save predictions; comparable to the official evaluation script
