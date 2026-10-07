import sys,json,os
from pathlib import Path
sys.path.insert(0,'/home/hanjack/ironcore/ironcore')
import torch
from ironcore.train import load_full_config
from ironcore.trainers import LanguageModelTrainer
from ironcore.training_utils import forward_step,loss_func
cfg=load_full_config(sys.argv[1],overrides={'operation.train_steps':2,'trainer.log_interval':1})
with LanguageModelTrainer(cfg,forward_step,loss_func) as trainer:
 trainer.train()
 torch.cuda.synchronize()
 result={'status':'completed','config':sys.argv[1],'context':cfg.model.max_seq_len,'backend':cfg.model.moe.expert_backend,'steps':2,'weights_dtype':str(next(trainer.model.parameters()).dtype),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
 Path(sys.argv[2]+f'_rank{os.environ["RANK"]}.json').write_text(json.dumps(result,indent=2)+'\n')
