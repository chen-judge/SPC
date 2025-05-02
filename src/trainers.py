import torch
from torch import nn
from transformers import Trainer, TrainerCallback, EvalPrediction, PreTrainedModel, PreTrainedTokenizerBase
from torch.utils.data import Dataset
from utils import print_rank_0
from typing import Union, Optional, List, Tuple, Callable, Dict
from arguments import SFTWeightedWithKLTrainingArguments
from copy import deepcopy
import deepspeed
from utils import print_object_on_main_process

from base import BaseTrainer


def compute_lm_loglikeli(logits, labels):
    batch_size, seq_length, vocab_size = logits.shape
        
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    neg_logprobs = loss_fct(shift_logits, shift_labels).reshape(batch_size, -1) #  #[bs, seq_len]     # [bs * seq_len]
    ignore_mask = shift_labels != -100
    
    # mean_loss = loss.sum(dim=-1) / ignore_mask.sum(dim=-1)
    # print_rank_0(neg_logprobs)
    return -1* neg_logprobs, ignore_mask


class OfflineWeightedPolicyTrainer(BaseTrainer):

    def _is_create_ref_model(self) -> bool:
        return True

    @staticmethod
    def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor = None, gather: bool = True) -> torch.Tensor:
        """
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:] if labels is not None else None

        return BaseTrainer.logprobs_from_logits(shift_logits, shift_labels, gather)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")
            
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels']
        )

        with torch.no_grad():
            self.ref_model.eval()
            ref_model_outputs = self.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

        
        logprobs, mask = compute_lm_loglikeli(model_outputs.logits, inputs['labels'])
        ref_logprobs, _ = compute_lm_loglikeli(ref_model_outputs.logits, inputs['labels'])
        
        logprob = (logprobs * mask).sum(-1) / mask.sum(-1)
        ref_logprob = (ref_logprobs * mask).sum(-1) / mask.sum(-1)
                
        # This is a sentence-level importance ratio
        importance_ratio = (logprob - ref_logprob).exp()
        
        
        self.args.kl_penalty_mode = "kl"
        with torch.no_grad():
            kl_div = self.compute_kl_divergence(logprobs, ref_logprobs, kl_penalty=self.args.kl_penalty_mode)

        
        if self.args.debug_mode:
            print_object_on_main_process("importance_ratio", importance_ratio)
        importance_ratio_clipped = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
        if self.args.debug_mode:
            print_object_on_main_process("importance_ratio_clipped", importance_ratio_clipped)

        advantages = inputs['rewards'] #- self.args.lm_kl_coeff * kl_div_avg
        if self.args.debug_mode:
            print_object_on_main_process("advantages", advantages)
        ppo_loss = - torch.minimum(advantages * importance_ratio, advantages * importance_ratio_clipped)

        sample_size, sft_size = (1-inputs['sft_mask']).sum(), (inputs['sft_mask']).sum()
        
        sft_loss = (- logprob * inputs['sft_mask']).sum() / sft_size if sft_size > 0 else sft_size
        ppo_loss = (ppo_loss * (1 - inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else sample_size
        
        total_loss = self.args.lm_sft_coeff * sft_loss + ppo_loss                
        
        weighted_loss = (total_loss * inputs['weights']).mean() # [batch_size]
        
        self.store_metrics({"ppo_loss": ppo_loss}, 'train')
        self.store_metrics({"sft_loss": sft_loss}, 'train')
        self.store_metrics({"kl_div": (kl_div*mask).sum()/mask.sum()}, 'train')
        
        pos_mask = (inputs['rewards'] > 0) * (1-inputs['sft_mask']) 
        neg_mask = (inputs['rewards'] < 0) * (1-inputs['sft_mask']) 
        kl_div_sentence = (kl_div * mask).sum(-1) / mask.sum(-1)

        kl_div_pos = (kl_div_sentence*pos_mask).sum()/pos_mask.sum() if pos_mask.sum() != 0 else 0
        kl_div_neg = (kl_div_sentence*neg_mask).sum()/neg_mask.sum() if neg_mask.sum() != 0 else 0
        kl_div_sft = (kl_div_sentence*inputs['sft_mask']).sum()/sft_size.sum() if sft_size.sum() != 0 else 0
        importance_ratio = (importance_ratio * mask).sum(-1) / mask.sum(-1)
        importance_ratio = (importance_ratio*(1-inputs['sft_mask'])).sum()/sample_size.sum() if sample_size.sum() != 0 else 0
        

        self.store_metrics({"kl_div_pos": kl_div_pos}, 'train')
        self.store_metrics({"kl_div_neg": kl_div_neg}, 'train')
        self.store_metrics({"kl_div_sft": kl_div_sft}, 'train')
        
        self.store_metrics({"importance_ratio": importance_ratio}, 'train')

        if self.args.debug_mode:          
            print_rank_0(f"check loss : {total_loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")

        return (weighted_loss, model_outputs.logits) if return_outputs else weighted_loss

