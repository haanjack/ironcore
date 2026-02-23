# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

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

    build_tokenizer(config)

    # initialize model with parallelism design
    model = LanguageModel(config).to(device=device, dtype=dtype)
    model = initialize_parallelism(config, model)

    # load checkpoint
    last_step = load_checkpoint(config, model)
    if last_step > -1:
        logger.info(f"Successfully loaded checkpoint: {config.trainer.model_path}")
    else:
        logger.error(
            f"Failed to load checkpoint from {config.trainer.model_path}. Aborting evaluation."
        )
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
