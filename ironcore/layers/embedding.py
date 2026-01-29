from torch import nn

from ironcore import get_tokenizer
from ironcore.config import MainConfig
from ironcore.layers.module import BaseModule
from ironcore.parallel.tensor_parallel import VocabParallelEmbedding


class LanguageModelEmbedding(BaseModule):
    def __init__(self, config: MainConfig):
        super().__init__(config)

        tokenizer = get_tokenizer()

        self.add_position_embedding = (
            config.model.positional_embedding.type == "absolute"
        )
        self.fp32_residual_connection = config.model.fp32_residual_connection

        self.word_embeddings = VocabParallelEmbedding(
            config,
            tokenizer.padded_vocab_size,
            config.model.d_model,
            padding_start_idx=tokenizer.vocab_size,
            parallel_output=True,
        )
        if self.add_position_embedding:
            self.position_embedding = nn.Embedding(
                config.model.max_seq_len,
                config.model.d_model,
            )
        self.embedding_dropout = nn.Dropout(config.model.dropout_embd)

    def forward(self, input_ids, position_ids):
        # Embeddings.
        output = self.word_embeddings(input_ids)

        # apply position embedding
        if self.add_position_embedding:
            output = output + self.position_embedding(position_ids)

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            output = output.float()

        # Dropout.
        if self.config.model.dropout_embd > 0.0:
            output = self.embedding_dropout(output)

        return output
