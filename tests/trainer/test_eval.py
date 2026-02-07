# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from pathlib import Path

import torch
from torch.distributed.fsdp.wrap import wrap
from torch.nn.parallel import DistributedDataParallel as DDP

from ironcore import get_logger, set_global_states
from ironcore.checkpointing import load_checkpoint
from ironcore.config import MainConfig, load_trainer_config
from ironcore.eval.tasks.hellaswag import HellaSwag
from ironcore.language_model import LanguageModel
from ironcore.parallel import initialize_parallelism, initialize_process
from ironcore.parallel.parallel_states import initialize_model_parallel
from ironcore.tokenizer import build_tokenizer
from ironcore.utils import get_device, get_model_dtype, is_first_rank


def main():
    """evaluation function"""
    config: MainConfig = load_trainer_config()
    set_global_states(config)
    initialize_process(config)

    logger = get_logger()

    # determine device and dtype
    device = get_device()
    dtype = get_model_dtype(config)

    initialize_model_parallel(
        config.trainer.tensor_model_parallel_size,
        timeout=config.parallel.timeout_minute,
    )

    tokenizer = build_tokenizer(config)

    # initialize model with parallelism design
    model = LanguageModel(config).to(device=device, dtype=dtype)
    model = initialize_parallelism(config, model)

    # load checkpoint
    last_step = load_checkpoint(config, model)
    if last_step > -1:
        logger.info(
            f"Successfuly loaded checkpoint: {config.trainer.model_path}")
    else:
        logger.info("Training start from scratch")
        last_step = 0

    # transfer model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize evaluator
    evaluator = HellaSwag(
        tokenizer=tokenizer,
        batch_size=int(config.trainer.eval_batch_size),
        num_samples=int(config.data.eval_datasets[0]["samples"]),
        cache_dir=Path("./cache"),
    )

    # load dataset and evaluate
    accuracy = evaluator.process(model=model)
    metrics = {"hellaswag": accuracy}
    logger.log_metrics(metrics)


if __name__ == "__main__":
    main()
