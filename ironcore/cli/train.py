"""Training CLI command."""

import sys
from pathlib import Path

from ironcore.config import MainConfig
from ironcore.config.config_data import DataConfig  # Old-style config for MainConfig
from ironcore.trainer import Trainer
from ironcore.training_utils import forward_step, get_loss_func


def run_train(args):
    """Run training command.

    Args:
        args: Command-line arguments from argparse
            - config: Path to training configuration YAML file
    """
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    print(f"Loading training configuration from: {config_path}")
    # Convert Path to string for config loading
    args.config_path = str(config_path)

    # Use the proper config loading function that handles nested configs
    from ironcore.config import load_trainer_config, _load_config_from_yaml
    from ironcore.config import ModelConfig, InitConfig, OptimConfig, ParallelConfig, TrainerConfig, OperationConfig, UtilsConfig
    from argparse import Namespace

    # Create a namespace with the config path
    config_args = Namespace(config_path=str(config_path))

    # Initialize default config
    config = MainConfig(
        model=ModelConfig(),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(),
        operation=OperationConfig(),
        utils=UtilsConfig(),
    )

    # Load config from YAML using the proper loader
    _load_config_from_yaml(config, config_args)

    # Set rank/world_size from environment
    import os
    config.parallel.rank = int(os.getenv("RANK", "0"))
    config.parallel.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    config.parallel.world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Validate config
    from ironcore.config import _config_validation
    _config_validation(config)

    # Select loss function based on task type
    task_type = getattr(config.data, 'task_type', 'pretrain')
    loss_fn = get_loss_func(task_type)
    print(f"Task type: {task_type}, using loss function: {loss_fn.__name__}")

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config, forward_step_func=forward_step, loss_fn=loss_fn)

    # Run training
    print("\nStarting training...")
    try:
        trainer.train()
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
