import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, ndcg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["rouge1", "bertscore", "lexrank", "attention", "perplexity", "simcse", "gptpmi", "text-davinci-003"])
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join("outputs", "{}__eval.csv".format(
            os.path.basename(args.input).split(".")[0]))

    with open(args.input) as fin:
        all_examples = json.load(fin)

    ndcg_list = []
    ap_list = []
    for example in all_examples:
        ndcg_labels = [[sum(x["label"]) for x in example["annotations"]]]
        ndcg_scores = np.array([[x[args.method] for x in example["annotations"]]])
        ndcg_scores[np.isnan(ndcg_scores)] = 0.

        ap_labels = [int(sum(x["label"]) > 1) for x in example["annotations"]]
        ap_scores = ndcg_scores[0]

        ndcg = ndcg_score(ndcg_labels, ndcg_scores)
        ap = average_precision_score(ap_labels, ap_scores)

        ndcg_list.append(ndcg)
        ap_list.append(ap)

    eval_df = pd.DataFrame({"ndcg": ndcg_list,
                            "ap": ap_list})
    eval_df.to_csv(args.output, index=False)