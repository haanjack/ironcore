import sys
sys.path.insert(0,"/home/hanjack/ironcore/ironcore")
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from ironcore.config import MainConfig,ModelConfig,TrainerConfig,InitConfig,OptimConfig,DataConfig,ParallelConfig,OperationConfig,UtilsConfig,ProfilerConfig,PEFTConfig
from ironcore.config.config_model import BiasConfig,PositionalEmbeddingConfig,KVCacheConfig
from ironcore.global_vars import set_global_states,global_states_cleanup
from ironcore.parallel.parallel_states import initialize_model_parallel,destroy_model_parallel
from ironcore.language_model import LanguageModel
from ironcore.checkpointing.hf_interop import load_from_huggingface
from ironcore.training_utils import loss_func_sft

torch.set_num_threads(4)
path='/tmp/ironcore-smollm135'
cfg=MainConfig(model=ModelConfig(d_model=576,d_ffn=1536,num_layers=30,num_attention_heads=9,num_attention_groups=3,head_dim=64,max_seq_len=1024,max_position_embeddings=8192,precision='float32',activation_type='swiglu',ln_type='rmsnorm',ln_eps=1e-5,positional_embedding=PositionalEmbeddingConfig(type='rope',base=100000),bias=BiasConfig.none(),untie_embed=False,reset_attention_mask=False,reset_position_ids=False,dropout_embd=0,dropout_attn=0,dropout_mlp=0,tokenizer_type='sentencepiece',vocab_name_or_path=path,kv_cache=KVCacheConfig(enabled=False)),trainer=TrainerConfig(vocab_padding_unit=128),init=InitConfig(),optim=OptimConfig(),data=DataConfig(),parallel=ParallelConfig(),operation=OperationConfig(),utils=UtilsConfig(),profiler=ProfilerConfig(),peft=PEFTConfig())
set_global_states(cfg);initialize_model_parallel(1,10)
model=LanguageModel(cfg,loss_func_sft).float().eval()
loaded=load_from_huggingface(path,model,architecture='llama')
tokenizer=AutoTokenizer.from_pretrained(path)
reference=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,attn_implementation='sdpa').eval()
inputs=tokenizer('Question: What is 2 plus 3? Answer:',return_tensors='pt')['input_ids']
with torch.no_grad():
 actual,_=model(inputs)
 expected=reference(inputs).logits
actual=actual[...,:expected.size(-1)]
error=(actual-expected).abs().max().item()
result={'max_logit_error':error,'weights':loaded,'parameters':sum(p.numel() for p in model.parameters()),'status':'passed' if error<2e-4 else 'failed'}
Path('/tmp/ironcore-smollm-native-oracle-final.json').write_text(json.dumps(result,indent=2)+'\n')
print('error',error,'missing',loaded['missing_keys'],'unexpected',loaded['unexpected_keys'])
torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)
global_states_cleanup();destroy_model_parallel()
