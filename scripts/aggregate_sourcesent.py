import argparse
import os
import json

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["pegasus", "bart", "lexrank"])
    parser.add_argument("--dataset", type=str, required=True, choices=["xsum", "cnn"])
    parser.add_argument("--label_threshold", type=int, choices=[1, 2], default=2)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.output is None:
        args.output = "outputs/{}_sourcesentsumm__{}__table.csv".format(args.dataset, args.label_threshold)

    data_list = []
    for method in args.methods:
        for version in ["orig", "sourcesent"]:
            with open("outputs/{}_sourcesentsumm__{}__{}__{}.json".format(args.dataset, method, args.label_threshold, version)) as fin:
                rouge_dict = json.load(fin)
            data_list.append((args.dataset,
                            method + "__" + version,
                            rouge_dict["ROUGE-1_F"],
                            rouge_dict["ROUGE-2_F"],
                            rouge_dict["ROUGE-L_F"]))
    eval_df = pd.DataFrame(data_list, columns=["dataset", "method", "R1_F", "R2_F", "RL_F"])
    eval_df.to_csv(args.output, index=False)

