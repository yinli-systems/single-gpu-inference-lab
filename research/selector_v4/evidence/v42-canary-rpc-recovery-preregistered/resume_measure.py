"""Only absent, pre-process RPC-failed repetitions; exact original GPU and CPUs."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
r=Path(sys.argv[1]);old=sys.argv[2];job=os.environ['SLURM_JOB_ID']
info=json.loads((r/'recovery-preregistration.json').read_text())['failures'][old]
import torch
allocated=[str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]
allocated=[x if x.startswith('GPU-') else 'GPU-'+x for x in allocated]
cpus=set(os.sched_getaffinity(0));wanted=set(info['cpu_affinity'])
if info['gpu_uuid'] not in allocated or not wanted<=cpus:
    raise RuntimeError('exact original hardware unavailable within legal allocation')
pre={'original_job':old,'resume_job':job,'allocated_gpu_uuids':allocated,'allocated_cpu_affinity':sorted(cpus),'selected_gpu_uuid':info['gpu_uuid'],'selected_cpu_affinity':sorted(wanted),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fresh_cases_consumed':0}
(r/f'receipts/resume-allocation-{job}.json').write_text(json.dumps(pre,indent=2)+'\n')
os.sched_setaffinity(0,wanted)
base=os.environ.copy();base['CUDA_VISIBLE_DEVICES']=info['gpu_uuid']
monitor=subprocess.Popen(['nvidia-smi','--id='+info['gpu_uuid'],'--query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv','-l','2'],stdout=(r/f'logs/resume-telemetry-{job}.csv').open('w'))
rc=1
try:
    for rep,mode in info['missing_processes']:
        overlay='pristine' if mode=='pristine' else 'candidate'
        env=base.copy();env['PYTHONPATH']=f'{r}/source:{r}/overlays/{overlay}'
        env['FLASHINFER_WORKSPACE_BASE']=f'{r}/cache/gpu_4090/{overlay}'
        env['XDG_CACHE_HOME']=f'{r}/cache/gpu_4090/{overlay}/xdg-{mode}-r{rep}'
        out=r/f'runs/canary-gpu_4090-s{info["shard"]}-r{rep}-{mode}-{job}'
        if out.exists():raise RuntimeError('never overwrite/repeat a started measurement')
        subprocess.run([sys.executable,'-m','research.selector_v4.measure_v4','--mode',mode,'--stage','canary','--rep',str(rep),'--shard',str(info['shard']),'--shards','10','--refs',f'{r}/refs/gpu_4090/s{info["shard"]}','--out',str(out)],env=env,cwd=r/'source',check=True)
        (out/'source_commit.txt').write_bytes((r/'source/SOURCE_COMMIT.txt').read_bytes())
        (out/'launcher-complete.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
    rc=0
finally:
    monitor.terminate();monitor.wait(timeout=10)
    (r/f'receipts/resume-exit-{job}.txt').write_text(str(rc)+'\n')
