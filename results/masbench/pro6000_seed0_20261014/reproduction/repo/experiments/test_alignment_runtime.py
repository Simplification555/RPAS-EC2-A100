import unittest
import alignment_runtime as runtime
from shared_task_context import fit_context, policy_for_dataset, ContextBudgetError

class AlignmentTests(unittest.TestCase):
 def test_summary_reflects_transport_requests(self):
  previous=runtime.EVENTS[:]
  try:
   runtime.EVENTS[:]=[{'context_budget':{'compression_rounds':0}},{'context_budget':{'compression_rounds':2}},{'status':'failed'}]
   summary=runtime.TransportOwnedGuard().summary()
   self.assertEqual(summary['context_limit'],16384)
   self.assertEqual(summary['calls_seen'],3)
   self.assertEqual(summary['compressed_calls'],1)
  finally:runtime.EVENTS[:]=previous
 def test_context_preserves_system_and_original_problem(self):
  messages=[{'role':'system','content':'protected system'},{'role':'user','content':'protected problem\n\nSubtask results:\n'+'x'*20000}]
  fitted,cap,budget=fit_context(messages,6144,policy_for_dataset('masbench'),lambda m:sum(len(x['content']) for x in m))
  self.assertEqual(fitted[0],messages[0])
  self.assertTrue(fitted[1]['content'].startswith('protected problem\n\nSubtask results:\n'))
  self.assertGreater(budget['compression_rounds'],0)
  self.assertLessEqual(budget['input_tokens']+cap+64,16384)
 def test_unmarked_problem_cannot_be_truncated(self):
  with self.assertRaises(ContextBudgetError):
   fit_context([{'role':'user','content':'x'*20000}],6144,policy_for_dataset('masbench'),lambda m:len(m[0]['content']))


class TransportTests(unittest.IsolatedAsyncioTestCase):
 async def test_parallel_workflow_calls_share_sixteen_request_limit(self):
  import asyncio,sys,types
  from unittest.mock import patch
  active=peak=0
  class FakeCompletions:
   _client=types.SimpleNamespace(base_url='http://127.0.0.1:18000/v1')
   async def create(self,**kwargs):
    nonlocal active,peak
    active+=1;peak=max(peak,active)
    self.assert_kwargs=kwargs
    await asyncio.sleep(0.002)
    active-=1
    return types.SimpleNamespace(usage=types.SimpleNamespace(model_dump=lambda:{'prompt_tokens':10,'completion_tokens':1,'total_tokens':11}),choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content='FINAL ANSWER: 1'))])
  module=types.ModuleType('openai.resources.chat.completions');module.AsyncCompletions=FakeCompletions
  previous=runtime.INSTALLED;events=[]
  runtime.INSTALLED=False
  try:
   with patch.dict(sys.modules,{'openai.resources.chat.completions':module}),patch.object(runtime,'fit_context',side_effect=lambda messages,requested,*args:(messages,requested,{'compression_rounds':0})),patch.object(runtime,'append_event',side_effect=events.append):
    runtime.install_transport()
    client=FakeCompletions()
    await asyncio.gather(*(client.create(model='Qwen/Qwen3.5-9B',messages=[{'role':'user','content':'test'}],max_tokens=32) for _ in range(64)))
    self.assertEqual(peak,16)
    self.assertEqual(len(events),64)
    self.assertEqual(client.assert_kwargs['extra_body']['chat_template_kwargs'],{'enable_thinking':False})
  finally:runtime.INSTALLED=previous
 async def test_missing_usage_is_recorded_as_failed(self):
  import sys,types
  from unittest.mock import patch
  class FakeCompletions:
   _client=types.SimpleNamespace(base_url='http://127.0.0.1:18000/v1')
   async def create(self,**kwargs):return types.SimpleNamespace(usage=None)
  module=types.ModuleType('openai.resources.chat.completions');module.AsyncCompletions=FakeCompletions
  previous=runtime.INSTALLED;events=[];runtime.INSTALLED=False
  try:
   with patch.dict(sys.modules,{'openai.resources.chat.completions':module}),patch.object(runtime,'fit_context',side_effect=lambda messages,requested,*args:(messages,requested,{'compression_rounds':0})),patch.object(runtime,'append_event',side_effect=events.append):
    runtime.install_transport()
    with self.assertRaisesRegex(RuntimeError,'usage'):
     await FakeCompletions().create(model='Qwen/Qwen3.5-9B',messages=[{'role':'user','content':'test'}],max_tokens=32)
    self.assertEqual(events[0]['status'],'failed')
  finally:runtime.INSTALLED=previous


class OptimizerEvidenceTests(unittest.TestCase):
 def test_optimizer_log_compression_preserves_questions_gold_and_tail_instructions(self):
  import json
  class Logs:
   def load_log(self):return json.dumps({'question':'protected original question','right_answer':'17','model_output':'x'*20000})
  runtime.patch_aflow_optimizer_logs(Logs)
  raw='protected workflow and graph\n'+Logs().load_log()+'\nprotected optimizer instructions'
  fitted,cap,budget=fit_context([{'role':'user','content':raw}],6144,policy_for_dataset('masbench'),lambda m:len(m[0]['content']))
  text=fitted[0]['content']
  self.assertIn('protected original question',text)
  self.assertIn('"right_answer": "17"',text)
  self.assertTrue(text.startswith('protected workflow and graph'))
  self.assertTrue(text.endswith('protected optimizer instructions'))
  self.assertGreater(budget['compression_rounds'],0)
  self.assertLessEqual(budget['input_tokens']+cap+64,16384)

if __name__=='__main__':unittest.main()
