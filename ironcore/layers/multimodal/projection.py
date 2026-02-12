# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from torch import nn

from ironcore.config import MainConfig
from ironcore.layers.module import BaseModule
from ironcore.parallel.tensor_parallel import ColumnParallelLinear


class VisionLanguageProjector(BaseModule):
    """Vision-Language projector for VLA models.

    Projects vision features from vision encoder dimension to language model dimension.
    Uses IronCore's ColumnParallelLinear and RowParallelLinear for tensor parallelism.
    """

    def __init__(self, config: MainConfig):
        super().__init__(config)

        vision_config = config.vla.vision
        projector_config = config.vla.projector
        model_config = config.model

        self.vision_hidden_size = vision_config.hidden_size
        self.projector_hidden_size = projector_config.hidden_size
        self.language_hidden_size = model_config.d_model
        self.num_layers = projector_config.num_layers

        # Choose activation
        if projector_config.activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Build projector based on type
        if projector_config.projector_type == "mlp" and self.num_layers == 2:
            # Two-layer MLP with intermediate expansion
            self.projector = nn.ModuleDict({
                "gate_proj": ColumnParallelLinear(
                    config,
                    self.vision_hidden_size,
                    self.projector_hidden_size,
                    bias=False,
                    gather_output=False,
                ),
                "up_proj": ColumnParallelLinear(
                    config,
                    self.projector_hidden_size,
                    self.language_hidden_size,
                    bias=False,
                    gather_output=True,  # Gather output for language model
                ),
            })
            self._is_two_layer = True
        elif projector_config.projector_type == "mlp":
            # Multi-layer MLP
            layers = []
            in_dim = self.vision_hidden_size
            for i in range(self.num_layers - 1):
                layers.append(
                    ColumnParallelLinear(
                        config,
                        in_dim,
                        self.projector_hidden_size,
                        bias=False,
                        gather_output=False,
                    )
                )
                layers.append(self.activation)
                in_dim = self.projector_hidden_size

            # Final projection to language dimension
            layers.append(
                ColumnParallelLinear(
                    config,
                    in_dim,
                    self.language_hidden_size,
                    bias=False,
                    gather_output=True,
                )
            )
            self.projector = nn.Sequential(*layers)
            self._is_two_layer = False
        else:
            # Simple linear projection
            self.projector = ColumnParallelLinear(
                config,
                self.vision_hidden_size,
                self.language_hidden_size,
                bias=False,
                gather_output=True,
            )
            self._is_two_layer = False

    def forward(self, vision_features):
        """Project vision features to language dimension.

        Args:
            vision_features: [batch, num_patches, vision_hidden_size]
                            Vision encoder output

        Returns:
            [batch, num_patches, language_hidden_size]
            Projected features ready for language model
        """
        if self._is_two_layer:
            # Two-layer MLP path
            hidden = self.projector["gate_proj"](vision_features)
            hidden = self.activation(hidden)
            output = self.projector["up_proj"](hidden)
            return output
        else:
            # Sequential or linear path
            return self.projector(vision_features)
