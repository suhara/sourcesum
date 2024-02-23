import argparse
from glob import glob
import json
import os
import sys

import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="outputs")
    parser.add_argument("--input", type=str, required=True, choices=["xsum_pegasus", "xsum_reference", "cnn_pegasus", "cnn_reference"])
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join("outputs", "{}__examples.csv".format(args.input))

    scores_dict = {}
    ranks_dict = {}
    id_list = []
    label_list = []
    sent_list = []
    summary_list = []
    for i, filepath in enumerate(glob(os.path.join(args.input_dir, "{}__*.json".format(args.input)))):
        method = os.path.basename(filepath).split("__")[1].split(".")[0]
        scores = []
        ranks = []
        with open(filepath) as fin:
            examples = json.load(fin)
        for e in examples:
            cur_scores = []
            for s in e["annotations"]:
                cur_scores.append(s[method])
                if i == 0:
                    id_list.append(e["id"])
                    label_list.append(sum(s["label"]))
                    sent_list.append(s["sentence"])
                    summary_list.append(e["gen_summary"])
            cur_ranks = (- np.array(cur_scores)).argsort().argsort().tolist()
            ranks += cur_ranks
            scores += cur_scores

        scores_dict[method] = scores
        ranks_dict["{}_rank".format(method)] = ranks
        assert len(scores) == len(id_list)
        assert len(scores) == len(label_list)
        assert len(scores) == len(sent_list)
        assert len(scores) == len(summary_list)

    score_df = pd.DataFrame(scores_dict | ranks_dict)
    score_df["id"] = id_list
    score_df["label"] = label_list
    score_df["sent"] = sent_list
    score_df["gen_summary"] = summary_list

    columns = ["id", "gen_summary", "sent", "label",
                "lexrank_rank", "rouge1_rank", "bertscore_rank", "attention_rank", "perplexity_rank",
                "lexrank", "rouge1", "bertscore", "attention", "perplexity"]
    if "reference" in args.input:
        columns = list(filter(lambda x: "attention" not in x, columns))

    score_df = score_df[columns]

    score_df.to_csv(args.output, index=False)
