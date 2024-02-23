import argparse
import itertools
import json
import os
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, ndcg_score
import torch
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PegasusTokenizer,
    PegasusForConditionalGeneration
)

import sys
sys.path.append("./lib")
from attention import CrossAttentionScorer
from graph_based import LexRankScorer
from perplexity_gain import PerplexityGainScorer
from preprocess import Preprocessor
from similarity_based import SimilarityBasedSentencePairScorer, SimCSEScorer, GPTPMIScorer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input", type=str, default="data/xsum_pegasus.json")
    parser.add_argument("--method", type=str, choices=["rouge1", "bertscore", "lexrank", "attention", "perplexity", "simcse", "gptpmi"], required=True)
    parser.add_argument("--input", type=str, required=True) # default="data/xsum_reference.json")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join("outputs", "{}.json".format(
            "__".join((os.path.basename(args.input).split(".")[0],
                       args.method))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    if args.method in ["attention", "perplexity"]:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum").to(device)

    preprocessor = Preprocessor(tokenizer=tokenizer, device=device)

    # * Step 1: Load JSON
    with open(args.input) as fin:
        examples = json.load(fin)

    eval_examples = []
    for example in tqdm(examples):
        sentences = []
        labels = []
        for sent_info in example["annotations"]:
            sentences.append(sent_info["sentence"])
            labels.append(sum(sent_info["label"])) # e.g., 0, 1, 2, 3...
        inputs, input_sent_ids_list = preprocessor.process_input_sentences(sentences)
        outputs, output_sent_ids_list = preprocessor.process_output_sentences([example["gen_summary"]])

        eval_examples.append(
            {"inputs": inputs,
             "outputs": outputs,
             "input_sent_ids_list": input_sent_ids_list,
             "output_sent_ids_list": output_sent_ids_list,
             "input_sents": sentences,
             "output_sents": [example["gen_summary"]],
             "labels": labels})

    # * Step 2: Prepare scorer
    if args.method == "lexrank":
        scorer = LexRankScorer([x["input_sents"] for x in eval_examples])
    elif args.method in ["rouge1", "bertscore"]:
        scorer = SimilarityBasedSentencePairScorer(score_name=args.method)
    elif args.method == "perplexity":
        scorer = PerplexityGainScorer(model=model, tokenizer=tokenizer)
    elif args.method == "attention":
        scorer = CrossAttentionScorer(model=model, tokenizer=tokenizer)
    elif args.method == "simcse":
        scorer = SimCSEScorer()
    elif args.method == "gptpmi":
        scorer = GPTPMIScorer()
    else:
        raise ValueError(args.method)

    # * Calculate scores
    for example, eval_example in tqdm(zip(examples, eval_examples)):
        if args.method == "perplexity":
            # The input order is different
            out_in_scores = scorer.calc_out_in_scores(eval_example["inputs"], eval_example["outputs"],
                                                      eval_example["input_sent_ids_list"],
                                                      eval_example["output_sent_ids_list"])
        elif args.method == "attention":
            out_in_scores = scorer.calc_out_in_scores(eval_example["inputs"],
                                                      eval_example["input_sent_ids_list"],
                                                      eval_example["input_sents"],
                                                      eval_example["output_sents"])
        else:
            out_in_scores = scorer.calc_out_in_scores(eval_example["output_sents"],
                                                      eval_example["input_sents"])

        # * Different for CNN/DM
        assert len(example["annotations"]) == out_in_scores.shape[1]
        for idx, score in enumerate(out_in_scores[0]):
            example["annotations"][idx][args.method] = score

    with open(args.output, "w") as fout:
        json.dump(examples, fout)