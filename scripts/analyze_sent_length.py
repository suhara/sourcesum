import argparse
from glob import glob
import json
import os

from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
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
    sentlen_list = []
    for i, filepath in enumerate(glob(os.path.join(args.input_dir, "{}__*.json".format(args.input)))):
        method = os.path.basename(filepath).split("__")[1].split(".")[0]
        scores = []
        with open(filepath) as fin:
            examples = json.load(fin)
        for e in examples:
            for s in e["annotations"]:
                scores.append(s[method])
                if i == 0:
                    sentlen_list.append(len(word_tokenize(s["sentence"])))
        scores_dict[method] = scores
        if i == 0:
            scores_dict["sentlen"] = sentlen_list

    score_df = pd.DataFrame(scores_dict)

    if "pegasus" in args.input:
        method_order = ["lexrank", "rouge1", "bertscore", "attention", "perplexity", "sentlen"]
        method_labels = ["LexRank", "ROUGE", "BERTScore", "Cross-attention", "Perplexity Gain", "Sentence length"]
    elif "reference" in args.input:
        method_order = ["lexrank", "rouge1", "bertscore", "perplexity", "sentlen"]
        method_labels = ["LexRank", "ROUGE", "BERTScore", "Perplexity Gain", "Sentence length"]

    score_df = score_df[method_order]
    score_df.columns = method_labels

    for i in range(len(method_labels) - 1): # Except for sentence length
        score_df.plot(kind="scatter", x="Sentence length", y=method_labels[i])
        plt.tight_layout()
        plt.savefig("outputs/{}__sentlen-vs-{}.pdf".format(args.input, method_labels[i]))
        plt.close()

