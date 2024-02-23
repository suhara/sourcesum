import argparse
from collections import defaultdict
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
    parser.add_argument("--method", type=str, choices=["rouge1", "bertscore", "lexrank", "attention", "perplexity", "simcse", "gptpmi"], required=True)
    parser.add_argument("--input", type=str, required=True) # default="data/xsum_reference.json")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join("outputs", "{}.json".format(
            "__".join((os.path.basename(args.input).split(".")[0],
                       args.method))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
    if args.method in ["attention", "perplexity"]:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail").to(device)

    preprocessor = Preprocessor(tokenizer=tokenizer, device=device)

    # * Step 1: Load JSON
    with open(args.input) as fin:
        all_examples = json.load(fin)

    id_examples_dict = defaultdict(list)
    for e in all_examples:
        # xsum__test__000571afe702684d90c1d222ce70b1e1375c1016__pegasus-xsum__0
        summary_idx = int(e["id"].split("__")[-1])
        id = "__".join(e["id"].split("__")[:-1])
        e["summary_idx"] = summary_idx
        id_examples_dict[id].append(e)

    for id, examples in id_examples_dict.items():
        id_examples_dict[id] = sorted(examples, key=lambda x: x["summary_idx"])

    eval_examples = []
    for id, examples in tqdm(id_examples_dict.items()):
        labels_list = []
        summary_sentences = []
        inputs, input_sent_ids_list = None, None
        sentences = []
        for i, example in enumerate(examples):
            labels = []
            for sent_info in example["annotations"]:
                labels.append(sum(sent_info["label"])) # e.g., 0, 1, 2, 3...
            labels_list.append(labels)
            summary_sentences.append(example["gen_summary"])
            if i == 0:
                # input sentences are the same
                for sent_info in example["annotations"]:
                    sentences.append(sent_info["sentence"])
                inputs, input_sent_ids_list = preprocessor.process_input_sentences(sentences)
        outputs, output_sent_ids_list = preprocessor.process_output_sentences(summary_sentences)

        eval_examples.append(
            {"id": id,
             "inputs": inputs,
             "outputs": outputs,
             "input_sent_ids_list": input_sent_ids_list,
             "output_sent_ids_list": output_sent_ids_list,
             "input_sents": sentences,
             "output_sents": summary_sentences,
             "labels": labels_list})

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

    # * Step 3: Calculate scores
    for eval_example in tqdm(eval_examples):
        doc = id_examples_dict[eval_example["id"]][0]["document"]
        sents = [x["sentence"] for x in id_examples_dict[eval_example["id"]][0]["annotations"]]
        if args.method == "perplexity":
            # The input order is different
            out_in_scores = scorer.calc_out_in_scores(eval_example["inputs"], eval_example["outputs"],
                                                      eval_example["input_sent_ids_list"],
                                                      eval_example["output_sent_ids_list"])
        elif args.method == "attention":
            out_in_scores = scorer.calc_out_in_scores(eval_example["inputs"],
                                                      eval_example["input_sent_ids_list"],
                                                      eval_example["input_sents"],
                                                      eval_example["output_sents"],
                                                      doc=doc,
                                                      sents=sents)
        else:
            out_in_scores = scorer.calc_out_in_scores(eval_example["output_sents"],
                                                      eval_example["input_sents"])

        # * The examples are already sorted
        for i, e in enumerate(id_examples_dict[eval_example["id"]]):
            for j, score in enumerate(out_in_scores[i]):
                e["annotations"][j][args.method] = score

    with open(args.output, "w") as fout:
        json.dump(all_examples, fout)