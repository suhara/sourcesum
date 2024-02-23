from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
from transformers import (
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
)
from transformers.generation_utils import BeamSearchEncoderDecoderOutput
from tqdm import tqdm


class PerplexityGainScorer:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 model: AutoModelForSeq2SeqLM):
        self.tokenizer = tokenizer
        self.model = model

    def calc_out_in_scores(self,
                           inputs: Dict[str, torch.Tensor],
                           outputs: Union[BeamSearchEncoderDecoderOutput, torch.Tensor],
                           input_sent_id_list: List[int],
                           output_sent_id_list: List[int]) -> np.ndarray:
        """! The interface is different from the similarity-based methods

        Args:
            inputs (Dict[str, torch.Tensor]): _description_
            outputs (BeamSearchEncoderDecoderOutput): _description_
            input_sent_id_list (List[int]): _description_
            output_sent_id_list (List[int]): _description_

        Returns:
            np.ndarray: _description_
        """

        #output_ids = outputs.sequences
        if type(outputs) == BeamSearchEncoderDecoderOutput:
            output_ids = outputs.sequences
        else:
            output_ids = outputs

        # DEBUG
        original_ppl = torch.exp(self.model(**inputs, labels=output_ids).loss).item()
        print(original_ppl)

        num_output_sents = max(output_sent_id_list)
        num_input_sents = max(input_sent_id_list)
        out_in_scores = np.zeros((num_output_sents, num_input_sents))

        for output_sent_id in range(1, num_output_sents + 1):
            # For each output sentence ID
            # Note: sent_id is 1 origin (0 is for padding)
            output_mask_array = (np.array(output_sent_id_list) == output_sent_id).astype("int")
            masked_output_ids = torch.masked_select(output_ids,
                                                    #torch.BoolTensor(output_mask_array)
                                                    torch.BoolTensor(output_mask_array).to(self.model.device)
                                                    ).unsqueeze(0)
            sent_ppl = torch.exp(self.model(**inputs, labels=masked_output_ids).loss).item()

            for input_sent_id in range(1, num_input_sents + 1):
                # For each input sentence ID
                # Note: sent_id is 1 origin (0 is for padding)
                input_mask_array = (np.array(input_sent_id_list) == input_sent_id).astype("int")
                attention_mask = (torch.LongTensor(1 - input_mask_array).to(self.model.device)) * inputs["attention_mask"]

                # Calculate "masked" perplexity
                masked_sent_ppl = torch.exp(
                                    self.model(input_ids=inputs["input_ids"],
                                               attention_mask=attention_mask,
                                               labels=masked_output_ids).loss)

                # Calculate the perplexity gain
                out_in_scores[output_sent_id - 1][input_sent_id - 1] = masked_sent_ppl - sent_ppl  # Larger is more important

        return out_in_scores