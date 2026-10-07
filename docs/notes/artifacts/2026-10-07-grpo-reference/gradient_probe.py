import sys
sys.path.insert(0, sys.argv[1])
from types import SimpleNamespace
import torch
from ironcore.trainers import GRPOTrainer
from ironcore.alignment.buffer import RolloutBuffer
from ironcore.parallel import parallel_states
parallel_states.initialize_model_parallel(1,10)
torch.manual_seed(47)
class Logits(torch.nn.Module):
 def __init__(self):
  super().__init__();self.logits=torch.nn.Parameter(torch.randn(4,6,32,dtype=torch.bfloat16))
 def forward(self,*args,**kwargs):return self.logits,None
model=Logits();trainer=object.__new__(GRPOTrainer);trainer.model=model
trainer.config=SimpleNamespace(alignment=SimpleNamespace(grpo_objective='grpo',moe_aux_loss='include',generation=SimpleNamespace(temperature=1.,top_p=1.,top_k=0)))
trainer.beta=.1;trainer.clip_eps=.2;trainer.entropy_coef=0
ids=torch.randint(0,32,(4,6));group=torch.zeros(4,dtype=torch.long)
rollout=RolloutBuffer(prompt_ids=ids[:1,:2],prompt_attention_mask=torch.ones(1,2),completion_ids=ids,response_ids=ids[:,2:],old_log_probs=torch.zeros(4),old_token_log_probs=torch.zeros(4,4),rewards=torch.zeros(4),advantages=torch.zeros(4),group_ids=group,metadata=[{}]*4,response_lengths=torch.tensor([4,3,2,4]))
labels,mask=trainer._prepare_labels_and_mask(rollout)
logp=model.logits.float().log_softmax(-1).gather(-1,labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)*mask
reference=logp.detach()+torch.randn_like(logp)*.7
old=logp.detach()+torch.randn_like(logp)*.3
rollout.old_token_log_probs=old[:,1:5].clone()
adv=torch.tensor([1.,-.5,.75,-1.5])
loss,_=trainer._compute_grpo_loss(rollout,adv,reference,old_log_probs=torch.zeros(4))
actual=torch.autograd.grad(loss,model.logits)[0]
fp=model.logits.detach().float().requires_grad_();p=fp.log_softmax(-1).gather(-1,labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
r=(p-old).exp();surrogate=torch.minimum(r*adv[:,None],r.clamp(.8,1.2)*adv[:,None]);delta=(reference-p).clamp(-6,6);k3=delta.exp()-delta-1
expected_loss=(((-surrogate+.1*k3)*mask).sum(-1)/mask.sum(-1)).mean()
expected=torch.autograd.grad(expected_loss,fp)[0].to(torch.bfloat16)
import json
print(json.dumps({'mismatched_gradient_elements': int((actual != expected).sum()), 'max_gradient_abs_error': float((actual - expected).abs().max()), 'loss_abs_error': float((loss - expected_loss).abs().detach())}, indent=2))
parallel_states.destroy_model_parallel()
