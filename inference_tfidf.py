import sys
import argparse
#sys.path.append('/private/home/sewonmin/EfficientQA-baselines/DPR')
#from dense_retriever import validate, save_results
import drqa_retriever as retriever


def main(args):
    questions = []
    question_answers = []

    for ds_item in parse_qa_csv_file(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    top_ids_and_scores = []
    for question in questions:
        psg_ids, scores = ranker.closest_docs(question, args.n_docs)
        top_ids_and_scores.append((psg_ids, scores))

    all_passages = load_passages(args.db_path)

    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

    questions_doc_hits = validate(all_passages, question_answers, top_ids_and_scores, args.validation_workers,
                                  args.match)

    if args.out_file:
        save_results(all_passages, questions, question_answers, top_ids_and_scores, questions_doc_hits, args.out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_file', required=True, type=str, default=None)
    parser.add_argument('--dpr_path', type=str, default="../DPR")
    parser.add_argument('--db_path', type=str, default="/checkpoint/sewonmin/dpr/data/wikipedia_split/psgs_w100_seen_only.tsv")
    parser.add_argument('--tfidf_path', type=str, default="/checkpoint/sewonmin/dpr/drqa_retrieval_seen_only/db-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'])
    parser.add_argument('--n-docs', type=int, default=100)
    parser.add_argument('--validation_workers', type=int, default=16)
    args = parser.parse_args()

    sys.path.append(args.dpr_path)
    from dense_retriever import parse_qa_csv_file, load_passages, validate, save_results

    ranker = retriever.get_class('tfidf')(tfidf_path=args.tfidf_path)

    main(args)


