# configure language model sequential

import torch
import torch.nn.functional as F
from torch import nn

from ironcore import get_tokenizer
from ironcore.config import MainConfig
from ironcore.layers import BaseModule, LanguageModelEmbedding
from ironcore.layers.layernorm import get_norm
from ironcore.layers.positional_embedding import RotaryPositionalEmbedding
from ironcore.models import get_model_provider_func
from ironcore.parallel import parallel_states
from ironcore.parallel.tensor_parallel import (
    ColumnParallelLinear,
    vocab_parallel_cross_entropy,
    copy_inputs_to_model_parallel_workers,
)


class LanguageModel(BaseModule):

    def __init__(
        self,
        config: MainConfig,
        loss_fn: torch.nn.modules.loss._Loss = F.cross_entropy,
    ):
        super().__init__(config)

        tokenizer = get_tokenizer()

        self.eod_mask_loss = config.model.eod_mask_loss
        self.reset_position_ids = config.model.reset_position_ids
        self.reset_attention_mask = config.model.reset_attention_mask
        self.fp16_lm_cross_entropy = config.model.fp16_lm_cross_entropy

        # model components initialization
        self.embedding = LanguageModelEmbedding(config)
        self.rotary_pos_emb = None
        if config.model.positional_embedding.type == "rope":
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                config.model.d_model // config.model.num_attention_heads,
                config.model.max_position_embeddings,
                base=config.model.positional_embedding.base,
                scale=config.model.positional_embedding.scaling_factor,
                offset=config.model.positional_embedding.offset,
            )

        model_provider_func = get_model_provider_func(config)
        self.model = model_provider_func(config)
        self.output_layernorm = get_norm(config)

        if config.model.untie_embed:
            self.output_layer = ColumnParallelLinear(
                config, config.model.d_model, tokenizer.padded_vocab_size, bias=False
            )

        self.loss_fn = loss_fn
        self.padding_start_idx = tokenizer.vocab_size

        self.init_weights()

        # Initialize VocabParallelEmbedding (zeros padding, registers hooks)
        if hasattr(self.embedding.word_embeddings, 'init_weight'):
            self.embedding.word_embeddings.init_weight()

    def forward(self, input_ids, labels=None):

        input_ids = input_ids.to(self.device, non_blocking=True)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        attention_mask, position_ids, loss_mask = self.get_masks_and_position_ids(input_ids, labels)

        # input_ids: [b s]
        # attention_mask: [b, 1, s, s]
        # position_ids: [b, s]
        # loss_mask: [b s]

        # pre process
        # x: [b, s, h]
        x = self.embedding(input_ids, position_ids)

        lm_output = self.model(x, attention_mask, self.rotary_pos_emb)

        # layer norm
        lm_output = self.output_layernorm(lm_output)

        # post process
        # lm_output: [b, s, h]
        outputs = self.post_lm_processing(
            lm_output,
            labels,
            loss_mask,
            self.fp16_lm_cross_entropy,
            padding_start_idx=self.padding_start_idx,
        )

        # outputs: logits[b, s, v] or loss[b, s]
        return outputs

    def get_masks_and_position_ids(self, input_ids, labels=None):

        # attention mask (lower triangular)
        if input_ids.dim() == 2:
            att_mask_batch = input_ids.size(0)  # micro_batch_size
        else:
            att_mask_batch = 1
        attention_mask = torch.tril(
            torch.ones(
                (att_mask_batch, input_ids.size(1), input_ids.size(1)),
                device=input_ids.device,
            )
        ).view(att_mask_batch, 1, input_ids.size(1), input_ids.size(1))

        # loss mask - CRITICAL: Must be based on labels, not input_ids
        # We're predicting the NEXT token, so mask positions where labels contain EOS/PAD
        loss_mask = torch.ones(
            input_ids.size(), dtype=torch.float, device=input_ids.device
        )
        # Only mask EOS/PAD tokens if eod_mask_loss is enabled
        # For nanoGPT-style training, we want to predict ALL tokens including across documents
        if self.eod_mask_loss and labels is not None:
            loss_mask[labels == get_tokenizer().eos_token_id] = 0
            loss_mask[labels == get_tokenizer().pad_token_id] = 0

        # position ids
        position_ids = torch.arange(
            input_ids.size(1), dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if self.reset_position_ids:
            position_ids = position_ids.clone()

        if self.reset_position_ids or self.reset_attention_mask:
            # loop through the batches
            for b in range(input_ids.size(0)):
                # find indices of EOD
                eod_index = position_ids[
                    b, input_ids[b] == get_tokenizer().eod_token_id
                ]
                # detach indices from position if going to modify them
                if self.reset_position_ids:
                    eod_index = eod_index.clone()

                # reset position ids along with EOD indices
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # reset attention mask
                    if self.reset_attention_mask:
                        attention_mask[b, 0, (i + 1):, : (i + 1)] = 0
                    # reset position
                    if self.reset_position_ids:
                        position_ids[b, (i + 1):] -= i + 1 - prev_index
                        prev_index = i + 1

        # convert attention mast to binary
        attention_mask = (attention_mask > 0.5).bool()

        return attention_mask, position_ids, loss_mask

    def compute_loss_from_logits(
        self, logits, labels, loss_mask, fp16_lm_cross_entropy=False, padding_start_idx: int = None
    ):
        """Compute loss from logits using vocab_parallel_cross_entropy.

        This method is reusable for both training and evaluation.
        It handles both TP=1 and TP>1 cases correctly.

        Args:
            logits: [batch, seq_len, vocab_size] or [batch, seq_len, vocab_size/tp]
            labels: [batch, seq_len] ground truth token IDs
            loss_mask: [batch, seq_len] valid token mask
            fp16_lm_cross_entropy: Whether to use fp16 for cross entropy
            padding_start_idx: Index where padding tokens start in vocab

        Returns:
            Scalar loss value
        """
        # Ensure labels are contiguous
        labels = labels.contiguous()

        # Convert to appropriate dtype
        if fp16_lm_cross_entropy:
            logits = logits.to(dtype=torch.half)
        else:
            logits = logits.float()

        # Compute per-token losses using vocab_parallel_cross_entropy
        # This handles both TP=1 and TP>1 cases correctly
        per_token_losses = vocab_parallel_cross_entropy(
            vocab_parallel_logits=logits,
            labels=labels,
            padding_start_idx=padding_start_idx,
        ).contiguous()

        # Apply loss function (e.g., loss_func_sft for per-sample averaging)
        loss = self.loss_fn(per_token_losses, loss_mask)

        return loss

    def post_lm_processing(
        self, lm_output, labels, loss_mask, fp16_lm_cross_entropy=False, padding_start_idx: int = None
    ):
        # b: batch size
        # s: sequence length
        # h: hidden_size
        # v: vocab_size
        # mp: tensor model parallel size

        # Compute logits from model output
        lm_output_parallel = lm_output
        if parallel_states.get_tensor_model_parallel_world_size() > 1:
            lm_output_parallel = copy_inputs_to_model_parallel_workers(lm_output)

        if self.config.model.untie_embed:
            logits = torch.matmul(lm_output_parallel, self.output_layer.weight)
        else:
            logits = torch.matmul(
                lm_output_parallel, self.embedding.word_embeddings.weight.transpose(0, 1)
            )

        if labels is None:
            # Return full logits for inference/evaluation
            # Shape: [b, s, v] where each position predicts the next token
            return logits

        # Compute loss from logits using shared method
        losses = self.compute_loss_from_logits(
            logits, labels, loss_mask, fp16_lm_cross_entropy, padding_start_idx
        )

        return losses
