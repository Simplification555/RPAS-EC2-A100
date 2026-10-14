from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_8.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # 1. Parse problems and dependencies to identify subproblems
        subproblems = self._parse_dependencies(problem)
        
        answers = []
        
        # 2. Solve each subproblem independently with custom method + ensemble
        for i, sub_p in enumerate(subproblems):
            original_obj = sub_p['object'].split("'s")[0]  # Extract base object for clarity
            result = await self.sc_ensemble(
                solutions=[
                    await self.custom(input=sub_p['text'], instruction=prompt_custom.PROMPT_SOLVER),
                    await self.custom(input=sub_p['text'], instruction=prompt_custom.PROMPT_ALT)
                ],
                problem=f"Calculate value for {i+1}: {sub_p['text']}"
            )
            answers.append(result['response'])
        
        # 3. Format final output string exactly as required
        # Join with '<<horizon>>'
        final_str = "FINAL ANSWER: " + "<<horizon>>".join(answers)
        return final_str

    def _parse_dependencies(self, text):
        lines = text.split('\n')
        problems = []
        # Simple heuristic to split by "Problem X:"
        current_prob = ""
        in_prob = False
        for line in lines:
            if "Problem" in line and ":" in line:
                in_prob = True
                current_prob = line
            elif in_prob:
                current_prob += "\n" + line
                if "Problem" in line and not line.strip().startswith("Problem"): # Start of next
                     pass 
                if "Note:" in line or "Solve all problems" in line or not in_prob:
                     if current_prob.strip():
                         problems.append(current_prob.strip())
                     current_prob = ""
                     in_prob = False
        if current_prob.strip():
            problems.append(current_prob.strip())
        return problems
