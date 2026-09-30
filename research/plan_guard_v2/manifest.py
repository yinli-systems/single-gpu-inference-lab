from pathlib import Path
import hashlib,json
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def load():
 p=Path(__file__).with_name("manifest.json");x=json.loads(p.read_text());assert digest(x["cases"])==x["case_hash"];return x
def workload_gate(q,cached):return max(cached)>=8192 and max(q)*len(q)*5>=sum(q)*6
def plan_guard(q,cached,padded_batch_size,num_kv_heads,num_sms,split_kv):return workload_gate(q,cached) and not split_kv and padded_batch_size*num_kv_heads>2*num_sms
