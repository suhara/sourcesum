import argparse
from collections import defaultdict
import json
import os
import re
import time
from typing import Tuple, Dict, List, Any
from warnings import warn

import numpy as np
import openai
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
    parser.add_argument("--input", type=str, required=True) # default="data/xsum_reference.json")
    parser.add_argument("--output", type=str)
    parser.add_argument("--prompt_filepath", type=str, default="data/gpt_prompt_v2.txt")
    parser.add_argument("--method", type=str, default="text-davinci-003", choices=["text-davinci-003"])
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--retry_interval", type=int, default=300)
    parser.add_argument("--basic_interval", type=int, default=2)
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = "".join(open(args.prompt_filepath).readlines())

    if args.output is None:
        args.output = os.path.join("outputs", "{}.json".format(
            "__".join((os.path.basename(args.input).split(".")[0],
                       args.method))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")

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

    # * Step 2: Calculate scores
    for eval_example in tqdm(eval_examples):
        out_in_scores = np.zeros((len(eval_example["output_sents"]), len(eval_example["input_sents"])))
        doc = id_examples_dict[eval_example["id"]][0]["document"]
        sents = [x["sentence"] for x in id_examples_dict[eval_example["id"]][0]["annotations"]]
        for i, output_sent in enumerate(eval_example["output_sents"]):
            for j, input_sent in enumerate(eval_example["input_sents"]):
                time.sleep(args.basic_interval)
                input_prompt = prompt.format(summary=output_sent, sentence=input_sent)

                success = False
                retry_count = 0
                while not success and retry_count < args.max_retry:
                    try:
                        response = openai.Completion.create(
                            model=args.method,
                            prompt=input_prompt)
                        output_text = response["choices"][0]["text"]
                        m = re.search(r"\d+", output_text)
                        if m:
                            score = int(m.group(0))
                            success = True
                    except Exception as e:
                        print(e)

                    if not success:
                        warn("Invalid output ({}-th trial): {}".format(retry_count + 1, output_text))
                        warn("  Sleep for {} sec.".format(args.retry_interval))
                        time.sleep(args.retry_interval)
                        retry_count += 1

                if success:
                    out_in_scores[i][j] = float(score)
                else:
                    out_in_scores[i][j] = np.nan

        # * The examples are already sorted
        for i, e in enumerate(id_examples_dict[eval_example["id"]]):
            for j, score in enumerate(out_in_scores[i]):
                e["annotations"][j][args.method] = score

    with open(args.output, "w") as fout:
        json.dump(all_examples, fout)