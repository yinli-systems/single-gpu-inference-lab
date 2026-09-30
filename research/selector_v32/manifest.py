from pathlib import Path
import hashlib,json

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def load():
    p=Path(__file__).with_name("manifest.json")
    x=json.loads(p.read_text())
    assert digest(x["cases"])==x["case_hash"]
    assert x["schema"]==4 and x["primary"]=="guarded"
    return x

def strict_opposite(q,cached):
    if len(q)!=len(cached) or len(q)<5:return False
    concordant=discordant=0
    for i in range(len(q)):
        for j in range(i+1,len(q)):
            prod=(q[i]-q[j])*(cached[i]-cached[j])
            if prod>0:concordant+=1
            elif prod<0:discordant+=1
    return concordant==0 and discordant>=len(q)-1

def descriptor_threshold(num_qo_heads,num_kv_heads,num_sms):
    if num_kv_heads<=0:raise ValueError("num_kv_heads")
    return max(40,(2*num_sms)//num_kv_heads+1)

def plan_guard(q,cached,padded_batch_size,num_qo_heads,num_kv_heads,num_sms,split_kv):
    return (strict_opposite(q,cached) and max(cached,default=0)>=8192 and not split_kv and
            padded_batch_size>=descriptor_threshold(num_qo_heads,num_kv_heads,num_sms))

def tactic_identity(*,environment,operation):
    required_env=("gpu_uuid","gpu_name","num_sms","driver","cuda","torch","flashinfer","python","cudnn","nvcc","backend_sha256","official_overlay_sha256","source_archive_sha256","selector_version","measurement_policy","profiling_mode")
    required_op=("execution_mode","backend","causal","layout","dtype","requested_split","actual_split","num_qo_heads","num_kv_heads","head_dim_qk","head_dim_vo","page_size","q","cached","ordered_pairs_sha256","plan_signature","window_repeats","candidate_binary_sha256")
    if any(k not in environment for k in required_env):raise ValueError("incomplete environment identity")
    if any(k not in operation for k in required_op):raise ValueError("incomplete operation identity")
    return digest({"environment":{k:environment[k] for k in required_env},"operation":{k:operation[k] for k in required_op}})
