import asyncio,hashlib,json,os,time
from pathlib import Path
import alignment_runtime as alignment
from shared_task_context import fit_context,policy_for_dataset,ContextBudgetError
from masbench_answer_protocol import score_masbench
from external_comparison.common.pareto import quality_operating_point,efficiency_operating_point
root=Path(__file__).resolve().parent;repo=root/'repo'
assert score_masbench('Reasoning gives 2','2')['score']==0
assert score_masbench('FINAL ANSWER: 2<<horizon>>3','2<<horizon>>4')['score_0_1_2']==1
assert not score_masbench('FINAL ANSWER: 2','2<<horizon>>4')['parser_valid']
records=[{'candidate_id':'b','valid':True,'score':.8,'avg_cost':10,'total_tokens':240}, {'candidate_id':'a','valid':True,'score':.8,'avg_cost':10,'total_tokens':240}, {'candidate_id':'c','valid':True,'score':.76,'avg_cost':9,'total_tokens':216}]
assert quality_operating_point(records)['candidate_id']=='a'
assert efficiency_operating_point(records)['candidate_id']=='c'
assert policy_for_dataset('masbench')['limit']==16384
report={'data_validation':alignment.validate_handoff(repo/'data/masbench'),'parser_and_QE_checks':'PASS','test_contents_not_opened':True}
rows=[json.loads(x) for x in (repo/'data/masbench/breadth/search_24.jsonl').read_text().splitlines() if x.strip()]
alignment.register_rows({'search':rows});alignment.install_transport();os.environ['ALIGN_RUN_DIR']=str(root/'preflight_requests')
from openai import AsyncOpenAI
from shared_scoring import MASBENCH_INSTRUCTION
async def main():
 async with AsyncOpenAI(base_url=os.environ.get('RPAS_ENDPOINT','http://127.0.0.1:8000/v1'),api_key='EMPTY',timeout=600,max_retries=0) as client:
  dev=MASBENCH_INSTRUCTION+rows[0]['input']
  for name,text,cap in [('normal_dev',dev,512),('compressed_dev',dev+'\n\nSubtask results:\n'+('Intermediate calculation 1 + 2 = 3. '*10000),128)]:
   response=await client.chat.completions.create(model='Qwen/Qwen3.5-9B',messages=[{'role':'system','content':'Follow the user task.'},{'role':'user','content':text}],temperature=0,max_tokens=cap,extra_body={'audit_metadata':{'phase':'preflight:'+name}})
   report[name]={'response':response.choices[0].message.content,'usage':response.usage.model_dump(),'finish_reason':response.choices[0].finish_reason,'context_budget':alignment.EVENTS[-1]['context_budget']}
  calls=[client.chat.completions.create(model='Qwen/Qwen3.5-9B',messages=[{'role':'user','content':f'Reply only FINAL ANSWER: {i}'}],temperature=0,max_tokens=32,extra_body={'audit_metadata':{'phase':'preflight:16concurrency'}}) for i in range(16)]
  started=time.time();responses=await asyncio.gather(*calls)
  assert all(r.usage is not None for r in responses)
  assert all(r.choices[0].finish_reason!='length' for r in responses), '16-request smoke output was truncated'
  assert all(score_masbench(r.choices[0].message.content,str(i))['correct'] for i,r in enumerate(responses)), '16-request smoke returned incorrect/protocol-invalid answers'
  report['concurrency_smoke']={'requests':16,'completed':len(responses),'failed':0,'wall_seconds':time.time()-started}
asyncio.run(main())
assert report['normal_dev']['context_budget']['compression_rounds']==0
assert report['compressed_dev']['context_budget']['compression_rounds']>0
(root/'preflight_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('ALIGNMENT_PREFLIGHT_PASS',json.dumps(report['concurrency_smoke']))
