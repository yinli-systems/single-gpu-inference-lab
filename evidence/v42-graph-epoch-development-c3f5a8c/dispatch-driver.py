from pathlib import Path
import datetime,hashlib,json,os,re,shutil,subprocess
R=Path('/ssd/scxi253/sgi-graph-epochs-c3f5a8c-20261002')
S=Path('/ssd/scxi253/sgi-official-wheel-validation-75544a17-20261001/wheel-overlay')
E='/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312/bin/python'
receipt=json.loads((R/'receipts/cpu-receipt.json').read_text());binding=json.loads((R/'binding.json').read_text())
assert receipt['pass'] and receipt['cpu_passed']==13 and receipt['gpu_skipped']==16
assert receipt['harness_commit']==binding['harness_commit']=='c3f5a8cba2b382b151a9908477bf3728f60ffea7'
assert hashlib.sha256((R/'source.tar.gz').read_bytes()).hexdigest()==binding['harness_archive_sha256']=='25d5b72c92d6a93221efeae09577bfe3f5a991cdb905eb03f34d6ed1f643d348'
assert not (R/'receipts/controller.json').exists() and not (R/'receipts/dispatch-intent.json').exists()
queue=subprocess.run(['squeue','--noheader','--jobs=1648991,1648992,1648993,1649001','--format=%N'],capture_output=True,text=True,check=True)
nodes=sorted(set(queue.stdout.split()))
assert nodes and all(re.fullmatch(r'[A-Za-z0-9_-]+',n) for n in nodes),queue.stdout
slurm_environment={'SBATCH_EXCLUDE':','.join(nodes)}
shutil.copy2(Path(__file__),R/'receipts/dispatch-driver.py')
argv=[E,str(R/'harness/research/selector_v4/exposed_diagnostics/graph_epoch_controller.py'),str(R),str(S)]
intent={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'harness_commit':binding['harness_commit'],'slurm_environment':slurm_environment,'dispatch_driver_sha256':hashlib.sha256((R/'receipts/dispatch-driver.py').read_bytes()).hexdigest(),'cpu_receipt_sha256':hashlib.sha256((R/'receipts/cpu-receipt.json').read_bytes()).hexdigest(),'qualification_authority':False,'default_promotion':False,'serving_promotion':False,'historical_token_divergence_resolved':False}
with (R/'receipts/dispatch-intent.json').open('x') as stream:
    stream.write(json.dumps(intent,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
with (R/'logs/controller.log').open('x') as log:
    child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,cwd=R,env={**os.environ,**slurm_environment,'PYTHONPATH':str(R/'harness'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True,close_fds=True)
with (R/'receipts/controller-pid.txt').open('x') as stream:stream.write(str(child.pid)+'\n')
print(json.dumps({'root':str(R),'controller_pid':child.pid,'source_commit':binding['harness_commit']}),flush=True)
