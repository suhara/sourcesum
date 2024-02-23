import argparse
import json
import os
import sys

from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
import pandas as pd
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizer
)

sys.path.append("lib")
from utils import rouge_scores2df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/xsum_reference.json")
    parser.add_argument("--method", type=str, choices=["lexrank", "pegasus", "bart"])  # TODO: Oracle?
    parser.add_argument("--label_threshold", type=int, choices=[1, 2], default=2)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    # * Step 1: Load JSON
    with open(args.input) as fin:
        examples = json.load(fin)

    # * Step 2: Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.method == "pegasus":
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum").to(device)
    elif args.method == "bart":
        # https://huggingface.co/facebook/bart-large-xsum
        # https://huggingface.co/facebook/bart-large-cnn
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum").to(device)
    elif args.method == "lexrank":
        lxr = LexRank([[x["sentence"] for x in e["annotations"]] for e in examples],
                    stopwords=STOPWORDS['en'])

    data_list = []
    for i, example in tqdm(enumerate(examples)):
        orig_sents = []
        source_sents = []
        for e in example["annotations"]:
            orig_sents.append(e["sentence"])
            if sum(e["label"]) >= args.label_threshold:
                source_sents.append(e["sentence"])

        sourcesent_doc = " ".join(source_sents)
        orig_doc = example["document"]  # == " ".join(orig_sents)
        ref = example["ref_summary"]

        if args.method in ["pegasus", "bart"]:
            sourcesent_outputs = model.generate(**tokenizer([sourcesent_doc], return_tensors="pt").to(device))
            sourcesent_summ = tokenizer.decode(sourcesent_outputs.squeeze(), skip_special_tokens=True)

            orig_outputs = model.generate(**tokenizer([orig_doc], return_tensors="pt").to(device))
            orig_summ = tokenizer.decode(orig_outputs.squeeze(), skip_special_tokens=True)

        elif args.method == "lexrank":
            sourcesent_summ = " ".join(lxr.get_summary(source_sents, summary_size=1, threshold=.1))
            orig_summ = " ".join(lxr.get_summary(orig_sents, summary_size=1, threshold=.1))

        else:
            raise ValueError(args.method)

        data_list.append((example["id"],
                          args.method,
                          orig_doc,
                          sourcesent_doc,
                          ref,
                          orig_summ,
                          sourcesent_summ))

    pred_df = pd.DataFrame(data_list, columns=["id","method", "orig_doc", "sourcesent_doc", "ref", "orig_summ", "sourcesent_summ"])

    # * Step 3: Evaluate ROUGE scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    orig_scores = [scorer.score(ref, gen) for ref, gen in zip(pred_df["ref"].tolist(), pred_df["orig_summ"].tolist())]
    orig_scores_df = rouge_scores2df(orig_scores)

    sourcesent_scores = [scorer.score(ref, gen) for ref, gen in zip(pred_df["ref"].tolist(), pred_df["sourcesent_summ"].tolist())]
    sourcesent_scores_df = rouge_scores2df(sourcesent_scores)

    pred_df.to_csv("outputs/xsum_sourcesentsumm__{}__{}__pred.csv".format(args.method, args.label_threshold), index=False)
    orig_scores_df.to_csv("outputs/xsum_sourcesentsumm__{}__{}__orig.csv".format(args.method, args.label_threshold), index=False)
    with open("outputs/xsum_sourcesentsumm__{}__{}__orig.json".format(args.method, args.label_threshold), "w") as fout:
        json.dump(orig_scores_df.mean().to_dict(), fout)
    sourcesent_scores_df.to_csv("outputs/xsum_sourcesentsumm__{}__{}__sourcesent.csv".format(args.method, args.label_threshold), index=False)
    with open("outputs/xsum_sourcesentsumm__{}__{}__sourcesent.json".format(args.method, args.label_threshold), "w") as fout:
        json.dump(sourcesent_scores_df.mean().to_dict(), fout)

