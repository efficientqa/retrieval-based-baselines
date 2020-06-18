import json
import csv
import argparse

def main(args):
    with open(args.data_path, "r") as f:
        data = json.load(f)
    seen_doc_titles = set()
    for dp in data:
        #seen_doc_titles.add(dp["positive_ctxs"][0]["title"])
        seen_doc_titles |= set([ctx["title"] for ctx in dp["positive_ctxs"][:5]])
    print ("Consider {} seen docs".format(len(seen_doc_titles)))

    rows = []
    with open(args.db_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # file format: doc_id, doc_text, title
        rows.extend([(row[0], row[1], row[2]) for row in reader if row[0] != 'id'])
    orig_n_passages = len(rows)
    rows = [row for row in rows if row[2] in seen_doc_titles]
    print ("Reducing # of passages from {} to {}".format(orig_n_passages, len(rows)))

    exit()
    with open(args.db_path.replace(".tsv", "_seen_only3.tsv"), "w") as f:
        for row in rows:
            f.write("{}\t{}\t{}\n".format(row[0], row[1], row[2]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/checkpoint/sewonmin/dpr/data/retriever/nq-train.json")
    parser.add_argument('--db_path', type=str, default="/checkpoint/sewonmin/dpr/data/wikipedia_split/psgs_w100.tsv")

    args = parser.parse_args()
    main(args)

