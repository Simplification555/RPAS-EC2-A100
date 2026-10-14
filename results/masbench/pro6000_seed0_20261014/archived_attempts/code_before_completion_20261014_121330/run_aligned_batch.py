"""Sequential native method/axis units sharing one capped Qwen service."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT/'repo';OUT=ROOT/'outputs/aflow_maas_9b_ctx16k_seed0_pro6000_20261013'
lock=(ROOT/'batch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=os.environ.copy();env.update(PYTHONPATH=f'{ROOT}/runtime_deps:{REPO}:{REPO}/experiments:{REPO}/external_comparison',
 RPAS_AFLOW_SOURCE=str(REPO/'external_baselines/AFlow'),RPAS_MAAS_SOURCE=str(REPO/'external_baselines/MaAS'),
 RPAS_MAAS_EMBEDDING_PATH=os.environ['RPAS_MAAS_EMBEDDING_PATH'],RPAS_TOKENIZER_PATH=os.environ['RPAS_MODEL_PATH'],
 RPAS_MAX_MODEL_LEN='16384',RPAS_MAX_NUM_SEQS='16',RPAS_TP_SIZE='1',RPAS_GPU_MEMORY_UTILIZATION=os.environ.get('GPU_MEMORY_UTILIZATION','0.85'),
 RPAS_AFLOW_REQUEST_TIMEOUT_S='600',RPAS_AFLOW_SAMPLE_TIMEOUT_S='1200',RPAS_MAAS_REQUEST_TIMEOUT_S='600',
 RPAS_MAAS_SAMPLE_TIMEOUT_S='1200',RPAS_MAAS_SAMPLE_RETRIES='3',RPAS_MAAS_CODE_TIMEOUT_S='60',
 CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENAI_API_KEY='EMPTY',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
PY=os.environ.get('RPAS_PYTHON',str(ROOT/'.venv-method/bin/python'))
state={'status':'preflight','started_at':time.time(),'units':[],'config':{'methods':['aflow','maas'],'axes':['breadth','depth','horizon','parallel','robustness'],'search_seed':0,'data_seed':2026,'concurrency':16,'context':16384,'max_output':6144,'thinking':False,'gpu_memory_utilization':float(os.environ.get('GPU_MEMORY_UTILIZATION','0.85')),'max_num_batched_tokens':int(os.environ.get('MAX_NUM_BATCHED_TOKENS','16384')),'chunked_prefill':True,'method_device':'cpu','hardware_profile':'RTX_PRO6000_Blackwell_96GB','temperatures':'native role defaults and overrides','scope':'new search-select-test, no reuse of 8K results'}}
def save():
 tmp=ROOT/'batch_status.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,ROOT/'batch_status.json')
def run(cmd,log):
 with (ROOT/log).open('ab',buffering=0) as f:return subprocess.run(cmd,env=env,cwd=REPO,stdout=f,stderr=f).returncode
save()
if (ROOT/'preflight_report.json').exists():
 os.replace(ROOT/'preflight_report.json',ROOT/f'preflight_report_previous_{time.time_ns()}.json')
if True:
 rc=run([PY,str(ROOT/'preflight_aligned.py')],'preflight_aligned.log')
 if rc:state.update(status='preflight_failed',stage='serving/context/16concurrency',exit_code=rc);save();raise SystemExit(rc)
rc=run([PY,str(ROOT/'check_native_imports.py')],'native_import_probe.log')
if rc:state.update(status='preflight_failed',stage='MaAS native imports',exit_code=rc);save();raise SystemExit(rc)
state['preflight_report_sha256']=hashlib.sha256((ROOT/'preflight_report.json').read_bytes()).hexdigest();save()
OUT.mkdir(parents=True,exist_ok=True)
for method in ['aflow','maas']:
 for axis in ['breadth','depth','horizon','parallel','robustness']:
  unit=OUT/method/axis/'seed_0';record={'method':method,'axis':axis,'started_at':time.time()}
  if unit.exists():
   gate=unit/'quality_gate.json'
   if gate.exists() and json.loads(gate.read_text()).get('passed'):
    rc=run([PY,str(REPO/'experiments/masbench_quality_gate.py'),'--run-dir',str(unit),'--rpas-root',str(REPO)],f'gate_{method}_{axis}.log')
    record['status']='reused_complete' if rc==0 else 'existing_gate_failed'
   else:record['status']='existing_partial_preserved'
   state['units'].append(record);save();continue
  state.update(status='running',current_unit=f'{method}/{axis}/seed_0');save()
  cmd=[PY,str(REPO/'experiments/native_masbench_external.py'),'--method',method,'--axis',axis,'--seed','0','--rpas-root',str(REPO),'--data-dir',str(REPO/'data/masbench'),'--run-dir',str(unit),'--endpoint',os.environ.get('RPAS_ENDPOINT','http://127.0.0.1:8000/v1'),'--data-seed','2026','--max-tokens','6144','--concurrency','16','--max-rounds','8','--maas-samples','1','--maas-batch-size','8']
  rc=run(cmd,f'run_{method}_{axis}.log');record['runner_exit_code']=rc
  if rc:record['status']='failed_preserved';record['diagnostic_tail']=(ROOT/f'run_{method}_{axis}.log').read_text()[-1800:]
  else:
   rc=run([PY,str(REPO/'experiments/masbench_quality_gate.py'),'--run-dir',str(unit),'--rpas-root',str(REPO)],f'gate_{method}_{axis}.log')
   record['gate_exit_code']=rc;record['status']='completed_pass' if rc==0 else 'completed_gate_failed'
  record['finished_at']=time.time();state['units'].append(record);save()
if all(x['status'] in ['completed_pass','reused_complete'] for x in state['units']):
 cmd=[PY,str(REPO/'experiments/aggregate_masbench_seed0.py'),'--root',str(OUT)]
 if (OUT/'seed0_summary').exists():cmd.append('--verify-existing')
 rc=run(cmd,'aggregate.log')
 state['status']='completed' if rc==0 else 'aggregation_failed'
else:state['status']='completed_with_failed_units'
state['finished_at']=time.time();save()
print(json.dumps(state,indent=2),flush=True)

if state['status'] != 'completed':raise SystemExit(2)
