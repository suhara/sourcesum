from typing import Tuple, Dict, List, Any

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer
)
from transformers.generation_utils import BeamSearchEncoderDecoderOutput


class CrossAttentionScorer:
    def __init__(self,
                 model: AutoModelForSeq2SeqLM,
                 tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sent_delim_id = self.tokenizer.vocab["<n>"]  # * hard-coded for Pegasus + CNN/DailyMail

    def calc_out_in_scores(self,
                           inputs: Dict,
                           input_sent_id_list: List, # input_sent_ids_list?
                           orig_input_sents: List,
                           orig_output_sents: List,
                           doc: str = None,
                           sents: List[str] = None,
                           return_dict: bool = False):
        """
        Return:
                out_in_scores

                or

                {"output_sents": output_sents,
                 "input_sents": input_sents,
                 "output_sent_id_list": output_sent_id_list,
                 "input_sent_id_list": output_sent_id_list,
                 "out_in_scores": out_in_scores}
        """

        # * Use default decoding configuration
        #if doc is not None:
        if sents is not None:
            orig_inputs = inputs
            inputs = self.tokenizer([" ".join(sents)], return_tensors="pt").to(self.model.device)
            # TODO: Update input_sent_id_list as well
            # import pdb; pdb.set_trace()
            print(len(orig_inputs["input_ids"][0]), len(inputs["input_ids"][0]))
            input_sent_id_list += [0]

        outputs: BeamSearchEncoderDecoderOutput = self.model.generate(**inputs,
                                                                      output_attentions=True,
                                                                      return_dict_in_generate=True)
        assert hasattr(outputs, "sequences")
        assert hasattr(outputs, "cross_attentions")

        # Parse generated text
        output_token_ids = outputs.sequences[0]

        output_sent_id_list = []
        cur_sent_id = 1
        for id in output_token_ids:
            if id == self.sent_delim_id:
                cur_sent_id += 1
                output_sent_id_list.append(0)
            elif id == self.tokenizer.pad_token_id: # or id == self.tokenizer.eos_token_id: # e.g., </s>
                output_sent_id_list.append(0)
            else:
                output_sent_id_list.append(cur_sent_id)

        # Convert Tuple of Tuple of Tensor(_, _, _, _) into Tensor
        #
        # e.g.,
        # >>> cross_attentions.shape
        # torch.Size([78, 16, 8, 16, 1, 158])
        cross_attentions = torch.stack([torch.stack(crs_att) for crs_att in outputs.cross_attentions])

        # Simple average by all layers & all heads
        agg_cross_attentions = cross_attentions.mean(dim=(1, 3))  # -> torch.Size([78, 8, 1, 158])
        # The first beam is the one used for generation
        agg_cross_attentions = agg_cross_attentions[:len(outputs.sequences[0]), 0, 0, :]  # -> torch.Size([76, 158])

        num_output_sents = max(output_sent_id_list)
        num_input_sents = max(input_sent_id_list)
        out_in_scores = np.zeros((num_output_sents, num_input_sents))

        for output_sent_id in range(1, num_output_sents + 1):
            # For each output sentence ID
            # Note: sent_id is 1 origin (0 is for padding)
            output_mask_array = (np.array(output_sent_id_list) == output_sent_id).astype("int")
            output_mask_tensor = torch.LongTensor(np.tile(output_mask_array,
                                                        (agg_cross_attentions.shape[1], 1)).T) # (num_output_tokens, num_input_tokens)
            #out_masked_agg_cross_attentions = agg_cross_attentions * output_mask_tensor
            out_masked_agg_cross_attentions = agg_cross_attentions.detach().cpu() * output_mask_tensor

            for input_sent_id in range(1, num_input_sents + 1):
                # For each input sentence ID
                # Note: sent_id is 1 origin (0 is for padding)
                input_mask_array = (np.array(input_sent_id_list) == input_sent_id).astype("int")
                input_mask_tensor = torch.LongTensor(np.tile(input_mask_array,
                                                            (agg_cross_attentions.shape[0], 1))) # (num_output_tokens, num_input_tokens)
                out_in_masked_agg_cross_attentions = out_masked_agg_cross_attentions * input_mask_tensor

                # * Simple sum
                out_in_scores[output_sent_id - 1][input_sent_id - 1] = out_in_masked_agg_cross_attentions.sum()

        # Input/output sentence preparation
        output_sents = []
        input_sents = []
        for output_sent_id in range(1, num_output_sents + 1):
            output_sents.append(
                self.tokenizer.decode(
                    # * Assume the batch size == 0
                    outputs.sequences[0][np.where(np.array(output_sent_id_list) == output_sent_id)[0]],
                    skip_special_tokens=True))


        for input_sent_id in range(1, num_input_sents + 1):
            input_sents.append(
                self.tokenizer.decode(
                    inputs["input_ids"][0][np.where(np.array(input_sent_id_list) == input_sent_id)[0]],
                    skip_special_tokens=True))

        # * Check if the model generates the same summary
        if len(output_sents) != len(orig_output_sents):
            out_in_scores = np.empty((len(orig_output_sents), len(orig_input_sents)))
            out_in_scores[:] = np.nan

        for s1, s2 in zip(output_sents, orig_output_sents):
            if s1 != s2:
                print(s1)
                print(s2)
                print("---")
            # assert s1 == s2

        if return_dict:
            return {"output_sents": output_sents,
                    "input_sents": input_sents,
                    "output_sent_id_list": output_sent_id_list,
                    "input_sent_id_list": input_sent_id_list,
                    "out_in_scores": out_in_scores}
        else:
            return out_in_scores
