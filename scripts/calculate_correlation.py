import argparse
from glob import glob
import json
import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs")
    parser.add_argument("--input", type=str, required=True, choices=["xsum_pegasus", "xsum_reference", "cnn_pegasus", "cnn_reference"])
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join("outputs", "{}__corr.csv".format(args.input))

    scores_dict = {}
    for i, filepath in enumerate(glob(os.path.join(args.input_dir, "{}__*.json".format(args.input)))):
        method = os.path.basename(filepath).split("__")[1].split(".")[0]
        scores = []
        with open(filepath) as fin:
            examples = json.load(fin)
        for e in examples:
            for s in e["annotations"]:
                scores.append(s[method])
        scores_dict[method] = scores

    score_df = pd.DataFrame(scores_dict)
    corr_df = score_df.corr()

    if "pegasus" in args.input:
        method_order = ["lexrank", "rouge1", "bertscore", "simcse", "gptpmi", "text-davinci-003", "attention", "perplexity"]
        method_labels = ["LexRank", "ROUGE", "BERTScore", "SimCSE", "PMI", "GPT-3.5", "CrossAttn", "PPL"]
    elif "reference" in args.input:
        method_order = ["lexrank", "rouge1", "bertscore", "simcse", "gptpmi", "text-davinci-003", "perplexity"]
        method_labels = ["LexRank", "ROUGE", "BERTScore", "SimCSE", "PMI", "GPT-3.5", "PPL"]

    corr_df = corr_df[method_order].loc[method_order]

    corr_df.columns = method_labels
    corr_df.index = method_labels
    corr_df.to_csv(args.output)

    #cmap = "RdYlGn"
    #cmap = "RdBu"
    cmap = "Blues"
    matrix = np.triu(corr_df.values)
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap=cmap, mask=matrix)
    plt.tight_layout()
    plt.savefig(args.output.replace(".csv", ".pdf"))
    plt.close()



