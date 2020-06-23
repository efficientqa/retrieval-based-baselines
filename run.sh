#!/bin/sh

RETRIEVAL=$1

tfidf_index="random"
dpr_index="random"
dpr_retrieval_checkpoint="random"
n_paragraphs="100"
python3 ../DPR/data/download_data.py --resource data.retriever.qas.nq --output_dir ${base_dir}
python3 ../DPR/data/download_data.py --resource data.wikipedia_split --output_dir ${base_dir}

if [ $RETRIEVAL = "tfidf-full" ]
then
    python3 ../DPR/data/download_data.py --resource indexes.tfidf.nq.full --output_dir ${base_dir} # DrQA index
    python3 ../DPR/data/download_data.py --resource checkpoint.reader.nq-tfidf.hf-bert-base --output_dir ${base_dir} # reader checkpoint
    tfidf_index="${base_dir}/indexes/tfidf/nq/full.npz"
    reader_checkpoint="${base_dir}/checkpoint/reader/nq-tfidf/hf-bert-base.cp"
    retrieval_type="tfidf"
    db_name="psgs_w100.tsv"
elif [ $RETRIEVAL = "tfidf-subset" ]
then
    python3 ../DPR/data/download_data.py --resource data.retriever.nq-train --output_dir ${base_dir}
    python3 filter_subset_wiki.py --db_path ${base_dir}/data/wikipedia_split/psgs_w100.tsv --data_path ${base_dir}/data/retriever/nq-train.json
    python3 ../DPR/data/download_data.py --resource indexes.tfidf.nq.subset --output_dir ${base_dir} # DrQA index
    python3 ../DPR/data/download_data.py --resource checkpoint.reader.nq-tfidf-subset.hf-bert-base --output_dir ${base_dir} # reader checkpoint
    tfidf_index="${base_dir}/indexes/tfidf/nq/subset.npz"
    reader_checkpoint="${base_dir}/checkpoint/reader/nq-tfidf-subset/hf-bert-base.cp"
    retrieval_type="tfidf"
    db_name="psgs_w100_subset.tsv"
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
elif [ $RETRIEVAL = "dpr-subset" ]
then
    python3 ../DPR/data/download_data.py --resource data.retriever.nq-train --output_dir ${base_dir}
    python3 filter_subset_wiki.py --db_path ${base_dir}/data/wikipedia_split/psgs_w100.tsv --data_path ${base_dir}/data/retriever/nq-train.json
    python3 ../DPR/data/download_data.py --resource checkpoint.retriever.single.nq.bert-base-encoder --output_dir ${base_dir} # retrieval checkpoint
    python3 ../DPR/data/download_data.py --resource indexes.single.nq.subset --output_dir ${base_dir} # DPR index
    python3 ../DPR/data/download_data.py --resource checkpoint.reader.nq-single-subset.hf-bert-base --output_dir ${base_dir} # reader checkpoint
    dpr_retrieval_checkpoint="${base_dir}/checkpoint/retriever/single/nq/bert-base-encoder.cp"
    dpr_index="${base_dir}/indexes/single/nq/subset"
    reader_checkpoint="${base_dir}/checkpoint/reader/nq-single-subset/hf-bert-base.cp"
    retrieval_type="dpr"
    db_name="psgs_w100_subset.tsv"
    n_paragraphs="40"
fi
python3 run_inference.py \
  --qa_file ${base_dir}/data/retriever/qas/nq-test.csv \
  --retrieval_type ${retrieval_type} \
  --db_path ${base_dir}/data/wikipedia_split/${db_name} \
  --tfidf_path ${tfidf_index} \
  --dpr_model_file ${dpr_retrieval_checkpoint} \
  --dense_index_path ${dpr_index} \
  --model_file ${reader_checkpoint} \
  --dev_batch_size 64 \
  --pretrained_model_cfg bert-base-uncased --encoder_model_type hf_bert --do_lower_case \
  --sequence_length 350 --eval_top_docs 10 20 40 50 80 100 --passages_per_question_predict ${n_paragraphs} \
  --prediction_results_file ${RETRIEVAL}_test_predictions.json
