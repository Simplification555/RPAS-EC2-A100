"""Shared serving-tokenizer boundary and RPAS alignment audit, opt-in for MASBench."""
import asyncio,collections,csv,hashlib,json,os,re,threading,time,uuid,weakref
from pathlib import Path
from shared_task_context import count_messages,fit_context,policy_for_dataset,ContextBudgetError
POLICY=policy_for_dataset('masbench')
ROWS={};EVENTS=[];METRICS_CURSOR=0;FAILURE_CURSOR=0;SAMPLE_FAILURES=[];LOCK=threading.Lock();INSTALLED=False

def register_rows(rows):
 for split,values in rows.items():
  for row in values:ROWS[str(row['id'])]=str(row['input'])

def identify(messages):
 text='\n'.join(str(m.get('content','')) for m in messages)
 hits=[(i,p) for i,p in ROWS.items() if p in text]
 return hits[0] if len(hits)==1 else (None,None)

def mark_evidence(messages):
 # Add only a structural marker after a complete registered original problem.
 # No early compaction of an unmarked problem or system text is permitted.
 result=[dict(m) for m in messages]
 for m in result:
  if m.get('role')!='user' or not isinstance(m.get('content'),str):continue
  text=m['content'];protected=[text.find(p)+len(p) for p in ROWS.values() if p in text]
  if not protected:continue
  end=max(protected)
  tail=text[end:]
  alias=re.search(r'(?m)(?:Several solutions have been generated[^\n]*|Other analysis:|Solution:)\s*',tail)
  if alias:
   at=end+alias.start();m['content']=text[:at]+'\n\nSubtask results:\n'+text[at:]
 return result

def append_event(event):
 EVENTS.append(event)
 directory=os.environ.get('ALIGN_RUN_DIR')
 if directory:
  path=Path(directory)/'request_ledger.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
  with LOCK:
   with path.open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

class TransportOwnedGuard:
 def __init__(self,*a,**k):pass
 def prepare(self,messages,requested):return messages,{'final_input_tokens':0,'counting_delegated_to':'actual serving /tokenize','policy':POLICY}
 def summary(self):
  budgets=[e['context_budget'] for e in EVENTS if 'context_budget' in e]
  return {'version':'shared_task_context_masbench_v3','policy':dict(POLICY),'context_limit':POLICY['limit'],
          'calls_seen':len(EVENTS),'count_source':'serving_tokenizer','early_4096_compaction':False,
          'compressed_calls':sum(b.get('compression_rounds',0)>0 for b in budgets)}

def install_transport():
 global INSTALLED
 if INSTALLED:return
 from openai.resources.chat.completions import AsyncCompletions
 original=AsyncCompletions.create
 limits=weakref.WeakKeyDictionary()
 async def aligned_create(self,*a,**kwargs):
  if a:raise TypeError('Expected keyword chat completions request')
  if kwargs.get('model')!='Qwen/Qwen3.5-9B':raise ValueError('Every inference/optimizer request must use Qwen3.5-9B')
  if kwargs.get('stream'):raise ValueError('Audited MASBench transport requires nonstreaming usage')
  kwargs=dict(kwargs);body=dict(kwargs.get('extra_body') or {})
  body['chat_template_kwargs']={'enable_thinking':False}
  kwargs['extra_body']=body
  requested=min(int(kwargs.get('max_tokens',6144)),6144)
  if requested<=0:raise ValueError('Output budget must be positive')
  messages=kwargs['messages'];sample_id,problem=identify(messages)
  meta=body.get('audit_metadata',{});phase=meta.get('phase','unattributed')
  event={'id':uuid.uuid4().hex,'phase':phase,'sample_id':sample_id,'started_at':time.time(),'model':kwargs['model'],'temperature':kwargs.get('temperature'),'top_p':kwargs.get('top_p',1),'thinking':False,'requested_max_tokens':requested}
  base=str(self._client.base_url)
  def prepare():
   return fit_context(mark_evidence(messages),requested,POLICY,lambda msgs:count_messages(msgs,kwargs['model'],{'api_base':base,'api_key':'EMPTY','extra_body':body}))
  try:
   guarded,effective,budget=await asyncio.to_thread(prepare)
   # Protected original text and all system messages must survive compaction.
   if problem and not any(problem in str(m.get('content','')) for m in guarded):raise RuntimeError('Protected original problem changed')
   if [m for m in guarded if m.get('role')=='system']!=[m for m in messages if m.get('role')=='system']:raise RuntimeError('System messages changed')
   kwargs['messages']=guarded;kwargs['max_tokens']=effective;kwargs['top_p']=1.0
   event['context_budget']=budget;event['effective_max_tokens']=effective
   event['messages_sha256']=hashlib.sha256(json.dumps(guarded,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
   body['audit_metadata']={**meta,'context_guard':budget}
   loop=asyncio.get_running_loop()
   if loop not in limits:limits[loop]=asyncio.Semaphore(16)
   async with limits[loop]:
    response=await original(self,**kwargs)
   if response.usage is None:raise RuntimeError('Serving response missing usage; token cost is unknown')
   event.update(status='completed',completed_at=time.time(),finish_reason=response.choices[0].finish_reason,content=response.choices[0].message.content,usage=response.usage.model_dump() if response.usage else None)
   append_event(event);return response
  except BaseException as exc:
   event.update(status='cancelled' if isinstance(exc,asyncio.CancelledError) else 'failed',completed_at=time.time(),error=f'{type(exc).__name__}: {exc}')
   if isinstance(exc,ContextBudgetError):event['context_budget']=exc.context_budget
   append_event(event);raise
 AsyncCompletions.create=aligned_create;INSTALLED=True

def request_events(phase=None):return [x for x in EVENTS if phase is None or x['phase']==phase]

def validity_metrics(log_dir,expected_rows,scores):
 global METRICS_CURSOR, FAILURE_CURSOR
 failed_ids={x["sample_id"] for x in SAMPLE_FAILURES[FAILURE_CURSOR:]}
 FAILURE_CURSOR=len(SAMPLE_FAILURES)
 current_events=EVENTS[METRICS_CURSOR:]
 METRICS_CURSOR=len(EVENTS)
 paths=sorted(Path(log_dir).glob('*.csv'),key=lambda p:p.stat().st_mtime)
 with paths[-1].open(newline='',encoding='utf-8-sig') as f:outputs=list(csv.DictReader(f))
 failures=0;bad_protocol=0;trunc_bad=0
 markers=('aflow sample timeout','maas sample failed','no code generated','maximum retries reached','traceback (most recent call last)','execution error:','code execution timed out','unknown error:','error:')
 for output,row,score in zip(outputs,expected_rows,scores):
  raw=str(output.get('prediction','') or '').strip()
  failed=str(row['id']) in failed_ids or not raw or any(raw.lower().startswith(m) for m in markers)
  failures+=int(failed);bad_protocol+=int(not failed and not score['parser_valid'])
  truncated=any(x.get('sample_id')==str(row['id']) and x.get('finish_reason')=='length' for x in current_events)
  trunc_bad+=int(truncated and not score['parser_valid'])
 n=len(expected_rows)
 return {'valid_execution_rate':1-failures/n,'error_example_rate':failures/n,'protocol_error_rate':bad_protocol/n,'truncated_unextractable_rate':trunc_bad/n}

def legal(metrics):
 return metrics['valid_execution_rate']>=.99 and metrics['error_example_rate']<=.01 and metrics['protocol_error_rate']<=.01 and metrics['truncated_unextractable_rate']<=.05

def validate_handoff(data_dir,include_test=False,axis=None):
 manifest=json.loads(Path(__file__).with_name('alignment_data_manifest.json').read_text());audit={}
 for group in manifest['groups']:
  if group['dataset']!='masbench' or (axis is not None and group['group']!=axis):continue
  axis=group['group'];audit[axis]={}
  for split,entry in group['splits'].items():
   path=Path(data_dir)/axis/Path(entry['path']).name;raw=path.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=entry['raw_sha256']:raise ValueError(f'Handoff data bytes differ: {axis}/{split}')
   record={'raw_sha256':entry['raw_sha256'],'matches':True,'content_check_deferred_until_lock':split=='test' and not include_test}
   if split!='test' or include_test:
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=entry['count']:raise ValueError('Data count differs')
    if hashlib.sha256('\n'.join(str(x['id']) for x in rows).encode()).hexdigest()!=entry['id_sha256']:raise ValueError('Data ID hash differs')
    if hashlib.sha256('\n'.join(str(x['content_sha256']) for x in rows).encode()).hexdigest()!=entry['content_sha256']:raise ValueError('Data content hash differs')
   audit[axis][split]=record
 return audit


def track_native_sample_failures(cls):
 original=cls._generate_output
 async def tracked(self,graph,input_text):
  try:return await original(self,graph,input_text)
  except BaseException as exc:
   sample_id,_=identify([{"role":"user","content":input_text}])
   SAMPLE_FAILURES.append({"sample_id":sample_id,"error":f"{type(exc).__name__}: {exc}"})
   raise
 cls._generate_output=tracked


def patch_aflow_optimizer_logs(cls):
 """Only model-generated output fields in sampled search logs are compressible."""
 original=cls.load_log
 def marked(self,*args,**kwargs):
  raw=original(self,*args,**kwargs)
  decoder=json.JSONDecoder();remaining=raw.strip();entries=[]
  while remaining:
   entry,end=decoder.raw_decode(remaining)
   if isinstance(entry,dict) and isinstance(entry.get('model_output'),str):
    entry=dict(entry)
    entry['model_output']='<intermediate_evidence>'+entry['model_output']+'</intermediate_evidence>'
   entries.append(json.dumps(entry,indent=4,ensure_ascii=False));remaining=remaining[end:].strip()
  return '\n\n'.join(entries)
 cls.load_log=marked
