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

from ironcore import get_logger, set_global_states
from ironcore.checkpointing import load_checkpoint
from ironcore.config import MainConfig, load_trainer_config
from ironcore.eval import get_evaluators
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
        timeout_in_minutes=config.parallel.timeout_minute,
    )

    tokenizer = build_tokenizer(config)

    # initialize model with parallelism design
    model = LanguageModel(config).to(device=device, dtype=dtype)
    model = initialize_parallelism(config, model)

    # load checkpoint
    last_step = load_checkpoint(config, model)
    if last_step > -1:
        logger.info(
            f"Successfully loaded checkpoint: {config.trainer.model_path}")
    else:
        logger.error(f"Failed to load checkpoint from {config.trainer.model_path}. Aborting evaluation.")
        return

    # initialize evaluators
    if is_first_rank():
        logger.info("Loading evaluation datasets ...")
    if config.data.eval_datasets is not None:
        evaluators = get_evaluators(
            config.data.eval_datasets,
            batch_size=int(config.trainer.eval_batch_size),
            cache_dir=Path("./cache"),
        )

        metrics = {}
        logger.info("Evaluation Start >>")
        for evaluator in evaluators:
            metrics.update({evaluator.task_name: evaluator.process(model)})
    else:
        metrics = {}
        logger.info("No evaluation datasets specified.")
    logger.info("Evaluation Finished <<")

    # load dataset and evaluate
    logger.log_metrics(metrics)


if __name__ == "__main__":
    main()
