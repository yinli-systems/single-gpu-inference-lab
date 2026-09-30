from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
class PositionProbeProcessor(CustomLogitProcessor):
 def __call__(self,logits,custom_param_list):
  import json,os,time
  path=os.environ.get('SGI_FORCE_POSITION_LOG')
  for i,param in enumerate(custom_param_list):
   req=param.get('__req__');forced=param.get('forced_tokens')
   if req is None or not isinstance(forced,list):raise RuntimeError('position-probe state missing')
   pos=len(req.output_ids);before=int(logits[i].argmax());chosen=None
   if pos<len(forced):
    chosen=int(forced[pos]);logits[i,:]=-float('inf');logits[i,chosen]=0.0
   row=dict(time=time.time(),rid=req.rid,pos=pos,output_ids=list(req.output_ids),forced_len=len(forced),before_argmax=before,forced_token=chosen,logits_shape=list(logits.shape))
   if path:
    with open(path,'a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
  return logits
