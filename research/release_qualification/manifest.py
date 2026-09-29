from pathlib import Path
import hashlib,json

def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def load():
 p=Path(__file__).with_name("manifest.json");x=json.loads(p.read_text());assert digest(x["cases"])==x["case_hash"];return x
def guarded(q,cached): return max(cached)>=8192 and max(q)*len(q)*5>=sum(q)*6
