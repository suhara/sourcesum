from typing import List

import bert_score
import numpy as np
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
import torch
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM)


# class USEBasedSentencePairScorer:
#     import tensorflow as tf
#     import tensorflow_hub as hub

#     def __init__(self):
#         self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

#     def calc_out_in_scores(self,
#                            output_sents: List[str],
#                            input_sents: List[str]) -> np.ndarray:
#         """Return out_in_scores."""
#         # L2-normalized embeddings (to calculate cosine similarity)
#         output_embeddings = tf.nn.l2_normalize(self.embed(output_sents), axis=1)
#         input_embeddings = tf.nn.l2_normalize(self.embed(input_sents), axis=1)

#         out_in_scores = tf.matmul(output_embeddings, tf.transpose(input_embeddings)).numpy()
#         assert out_in_scores.shape == (len(output_sents), len(input_sents))

#         return out_in_scores


class SimilarityBasedSentencePairScorer:
    def __init__(self, score_name: str = None):
        self.score_name = score_name
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=False)

    def calc_out_in_scores(self,
                           output_sents: List[str],
                           input_sents: List[str],
                           score_name: str = None) -> np.ndarray:
        """Return out_in_scores."""

        if not self.score_name:
            assert score_name is not None
        if score_name is None:
            score_name = self.score_name

        if score_name == "rouge1":
            score_func = self.rouge_scorer.score
            metric_func = lambda x: x["rouge1"].fmeasure
        elif score_name == "bertscore":
            score_func = self.bert_scorer.score  # The input is List, List (not str, str)
            metric_func = lambda x: x[2].item() # P, R, F1
        else:
            return ValueError("Undefined metric name.")

        out_in_scores = np.zeros((len(output_sents), len(input_sents)))
        for i, output_sent in enumerate(output_sents):
            for j, input_sent in enumerate(input_sents):
                if score_name == "bertscore":
                    out_in_scores[i, j] = metric_func(score_func([output_sent],
                                                                 [input_sent]))
                else:
                    out_in_scores[i, j] = metric_func(score_func(output_sent, input_sent))
        return out_in_scores


class SimCSEScorer:
    def __init__(self):
        # Import our models. The package will take care of downloading the models automatically
        self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    def calculate_score(self, summary_sent: str, source_sent: str) -> float:
        inputs = self.tokenizer([summary_sent, source_sent], padding=True, truncation=True, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            embeddings = self.model(**inputs,
                                    output_hidden_states=True,
                                    return_dict=True).pooler_output

        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        cosine_sim = 1 - cosine(embeddings[0], embeddings[1])
        return cosine_sim

    def calc_out_in_scores(self,
                           output_sents: List[str],
                           input_sents: List[str]) -> np.ndarray:
        out_in_scores = np.zeros((len(output_sents), len(input_sents)))
        for i, output_sent in enumerate(output_sents):
            for j, input_sent in enumerate(input_sents):
                out_in_scores[i, j] = self.calculate_score(output_sent, input_sent)
        return out_in_scores


class GPTPMIScorer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def calculate_langscore(self, sent: str) -> float:
        inputs = self.tokenizer(sent, return_tensors="pt")
        return - self.model(**inputs, labels=inputs["input_ids"][0]).loss.item()  # NLL -> LL

    def calculate_pmi(self, summary_sent: str, source_sent: str) -> float:
        """
        PMI(summary_sent; source_sent) = \log \frac{P_{LM}}(source_sent|summary_sent)}{P_{LM}}(source_sent)
        """
        inputs = self.tokenizer(" ".join([summary_sent, source_sent]), return_tensors="pt")
        prefix_ids_len = len(self.tokenizer.encode(summary_sent))
        output_ids = inputs.input_ids.clone()
        output_ids[:, :prefix_ids_len] = -100
        with torch.no_grad():
            outputs = self.model(**inputs, labels=output_ids)
            plm_summary_source = - outputs.loss.item()  # NLL -> LL
        plm_source = self.calculate_langscore(source_sent)
        return plm_summary_source - plm_source

    def calc_out_in_scores(self,
                           output_sents: List[str],
                           input_sents: List[str]) -> np.ndarray:
        out_in_scores = np.zeros((len(output_sents), len(input_sents)))
        for i, output_sent in enumerate(output_sents):
            for j, input_sent in enumerate(input_sents):
                out_in_scores[i, j] = self.calculate_pmi(output_sent, input_sent)
        return out_in_scores


if __name__ == "__main__":
    sentpair_scorer = SimilarityBasedSentencePairScorer()
    output_sents = ["The quick brown fox jumps over the lazy dog"]
    input_sents = ["The quick brown dog jumps on the log."]
    out_in_scores = sentpair_scorer.calc_out_in_scores(output_sents, input_sents, score_name="bertscore")
    print(out_in_scores)

    """
    P, R, F1 = bert_score.score(['The quick brown fox jumps over the lazy dog'],
                            ['The quick brown dog jumps on the log.'],
                            lang="en", verbose=True)
    print(P, R, F1)
    """