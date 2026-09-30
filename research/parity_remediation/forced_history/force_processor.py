from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
class ForcedHistoryProcessor(CustomLogitProcessor):
 def __call__(self,logits,custom_param_list):
  if len(custom_param_list)!=logits.shape[0]:raise RuntimeError('processor row mismatch')
  for i,param in enumerate(custom_param_list):
   req=param.get('__req__');forced=param.get('forced_tokens')
   if req is None or not isinstance(forced,list):raise RuntimeError('forced-history state missing')
   pos=len(req.output_ids)
   if pos<len(forced):
    token=int(forced[pos])
    if token<0 or token>=logits.shape[-1]:raise RuntimeError('forced token out of range')
    logits[i,:]=-float('inf');logits[i,token]=0.0
  return logits
