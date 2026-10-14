from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_3.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name,
        llm_config,
        dataset,
    ):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem):
        prompts = [
            "Solve the math problem. Use Z_23 arithmetic. Return strictly in format: FINAL ANSWER: <values>. Do not show work.",
            "Analyze the problem carefully. Calculate all values modulo 23. Ensure order is preserved using <<horizon>>. Return strictly in format: FINAL ANSWER: <values>. Show steps first.",
            "Verify the previous calculation. Check modulo 23 reduction. Output strictly: FINAL ANSWER: <values>."
        ]
        responses_raw = []
            for p in prompts:
            res = await self.custom(input=f"{problem}@@@INST:{p}", instruction="")
            responses_raw.append(res['response'])
            solutions = [r.split('FINAL ANSWER:')[-1].strip() if 'FINAL ANSWER:' in r else r for r in responses_raw]
        best_sol = await self.sc_ensemble(input=solutions, problem=problem)
        final_text = best_sol['response']
        return self.extract_final_answer(final_text), dict(usage="calculated")

    def extract_final_answer(self, text):
        start = text.find("FINAL ANSWER:")
        if start != -1:
            raw_ans = text[start+14:].strip()
        else:
            pattern = r"\{\{(.*?)\}\}"
            matches = re.findall(pattern, text)
            if matches:
                raw_ans = "<<horizon>>".join(matches)
            else:
                raw_ans = text
        return raw_ans
