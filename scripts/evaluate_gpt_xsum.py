import argparse
import itertools
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
    parser.add_argument("--prompt_filepath", type=str, default="data/gpt_prompt_v2.txt")
    parser.add_argument("--method", type=str, default="text-davinci-003", choices=["text-davinci-003"])
    parser.add_argument("--input", type=str, required=True) # default="data/xsum_reference.json")
    parser.add_argument("--output", type=str)
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
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

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

    # *
    for example, eval_example in tqdm(zip(examples, eval_examples)):
        out_in_scores = np.zeros((len(eval_example["output_sents"]), len(eval_example["input_sents"])))
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

        # * Different for CNN/DM
        assert len(example["annotations"]) == out_in_scores.shape[1]
        for idx, score in enumerate(out_in_scores[0]):
            example["annotations"][idx][args.method] = score

    with open(args.output, "w") as fout:
        json.dump(examples, fout)