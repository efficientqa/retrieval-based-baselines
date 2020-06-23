#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import argparse
import csv
import json
import numpy as np

sys.path.append("../DPR")
from dense_retriever import parse_qa_csv_file, load_passages, validate, save_results
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
            add_tokenizer_params, add_cuda_params, add_training_params, add_reader_preprocessing_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument('--qa_file', required=True, type=str, default=None)
    parser.add_argument('--retrieval_type', type=str, default='drqa',
                        choices=['tfidf', 'dpr'])
    parser.add_argument('--dpr_model_file', type=str, default="/private/home/sewonmin/EfficientQA-baselines/DP")
    parser.add_argument('--db_path', type=str, default="/checkpoint/sewonmin/dpr/data/wikipedia_split/psgs_w100_seen_only.tsv")

    # retrieval specific params
    parser.add_argument('--dense_index_path', type=str, default="")
    parser.add_argument('--tfidf_path', type=str, default="/checkpoint/sewonmin/dpr/drqa_retrieval_seen_only/db-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'])
    parser.add_argument('--n-docs', type=int, default=100)
    #parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', default=True, help='If enabled, save index')

    # reader specific params
    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)
    add_reader_preprocessing_params(parser)


    parser.add_argument("--max_n_answers", default=10, type=int,
                        help="Max amount of answer spans to marginalize per singe passage")
    parser.add_argument('--passages_per_question', type=int, default=2,
                        help="Total amount of positive and negative passages per question")
    parser.add_argument('--passages_per_question_predict', type=int, default=50,
                        help="Total amount of positive and negative passages per question for evaluation")
    parser.add_argument("--max_answer_length", default=10, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument('--eval_top_docs', nargs='+', type=int,
                        help="top retrival passages thresholds to analyze prediction results for")
    parser.add_argument('--checkpoint_file_name', type=str, default='dpr_reader')
    parser.add_argument('--prediction_results_file', type=str, help='path to a file to write prediction results to')


    args = parser.parse_args()

    '''questions = []
    with open(args.qa_file) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            questions.append(row[0])'''
    questions = []
    question_answers = []
    for ds_item in parse_qa_csv_file(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    if args.retrieval_type=="tfidf":
        import drqa_retriever as retriever
        ranker = retriever.get_class('tfidf')(tfidf_path=args.tfidf_path)
        top_ids_and_scores = []
        for question in questions:
            psg_ids, scores = ranker.closest_docs(question, args.n_docs)
            top_ids_and_scores.append((psg_ids, scores))
    else:
        from dpr.models import init_biencoder_components
        from dpr.utils.data_utils import Tensorizer
        from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
        from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer
        from dense_retriever import DenseRetriever

        saved_state = load_states_from_checkpoint(args.dpr_model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)
        tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)
        encoder = encoder.question_model
        setup_args_gpu(args)
        encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                                args.local_rank,
                                                args.fp16)
        encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(encoder)
        prefix_len = len('question_model.')
        question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                                key.startswith('question_model.')}
        model_to_load.load_state_dict(question_encoder_state)
        vector_size = model_to_load.get_out_size()

        index_buffer_sz = args.index_buffer
        if args.hnsw_index:
            index = DenseHNSWFlatIndexer(vector_size)
            index_buffer_sz = -1  # encode all at once
        else:
            index = DenseFlatIndexer(vector_size)

        retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)
        retriever.index.deserialize_from(args.dense_index_path)

        questions_tensor = retriever.generate_question_vectors(questions)
        top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)


    all_passages = load_passages(args.db_path)

    retrieval_file = "tmp_{}.json".format(str(np.random.randint(0, 100000)).zfill(6))
    questions_doc_hits = validate(all_passages, question_answers, top_ids_and_scores,
                                  1, args.match)

    save_results(all_passages,
                 questions,
                 question_answers, #["" for _ in questions],
                 top_ids_and_scores,
                 questions_doc_hits, #[[False for _ in range(args.n_docs)] for _n in questions],
                 retrieval_file)
    setup_args_gpu(args)
    #print_args(args)
    args.dev_file = retrieval_file

    #from IPython import embed; embed()
    from train_reader import ReaderTrainer

    class MyReaderTrainer(ReaderTrainer):
        def _save_predictions(self, out_file, prediction_results):
            with open(out_file, 'w', encoding="utf-8") as output:
                save_results = []
                for r in prediction_results:
                    save_results.append({
                        'question': r.id,
                        'prediction': r.predictions[50].prediction_text
                    })
                output.write(json.dumps(save_results, indent=4) + "\n")

    trainer = MyReaderTrainer(args)
    trainer.validate()

    os.remove(retrieval_file)
    for i in range(args.num_workers):
        os.remove(retrieval_file.replace(".json", ".{}.pkl".format(i)))

