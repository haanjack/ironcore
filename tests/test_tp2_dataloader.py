import torch
import torch.distributed as dist
import sys
import os

sys.path.insert(0, os.getcwd())

from ironcore.config import load_trainer_config
from ironcore.parallel import parallel_states, initialize_process
from ironcore.dataloader import get_data_iterator
from ironcore.global_vars import set_global_states

def run_test():
    config = load_trainer_config()
    set_global_states(config)
    initialize_process(config)
    
    # Initialize TP
    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=config.trainer.tensor_model_parallel_size,
        timeout=config.parallel.timeout_minute
    )
    
    iterators = get_data_iterator(config)
    train_iter = iterators['train']
    
    rank = dist.get_rank()
    
    print(f"Rank {rank}: Initialized. TP={config.trainer.tensor_model_parallel_size}")
    
    for i in range(5):
        batch = next(train_iter)
        input_ids = batch['input_ids']
        s = input_ids.sum().item()
        print(f"Rank {rank}: Batch {i} sum: {s}")
        
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    run_test()