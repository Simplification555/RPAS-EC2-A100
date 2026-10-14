from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_9.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        # 1. Parse problems and dependencies to identify subproblems
        text = problem.replace('Note:', ' ').replace('Solve all problems', ' ')
        raw_problems = text.split('Problem')
        subproblems = []
        for rp in raw_problems:
            if 'Problem' in rp:
                subproblems.append(rp.split('\n', 1)[-1].replace('\n', ' ').strip())
        
        answers = []
        
        # 2. Solve each subproblem independently
        for idx, sub_p in enumerate(subproblems):
            try:
                # Initial attempt with self-consistency
                original_obj = sub_p.split(': ')[1].split('.')[0] if ': ' in sub_p else sub_p
                result = await self.sc_ensemble(
                    solutions=[
                        await self.custom(input=sub_p.replace(f"Problem {idx+1}:", f"Question {idx+1}: "), instruction=prompt_custom.PROMPT_SOLVER),
                        await self.custom(input=sub_p.replace(f"Problem {idx+1}:", f"Question {idx+1}: "), instruction=prompt_custom.PROMPT_ALT)
                    ],
                    problem=f"Refine count for item {idx+1}: {original_obj}"
                )
                raw_ans = result['response'].lower()
                
                # Validate output format: must be an integer inside boxed{} or standalone
                import re
                if re.search(r'\\?\s*boxed\s*{?\s*(\d+)\s*}?|^(\d+)', raw_ans):
                    num = int(re.search(r'\\?(?:boxed)?\s*[\{\s]*(-?\d+)[\}\s]*', raw_ans, re.IGNORECASE).group(1))
                    answers.append(str(num))
                else:
                    raise ValueError("Invalid numeric format")
            
            except Exception as e:
                # Fallback to Programmer for complex logic if LLM fails
                try:
                    code_result = await self.programmer(problem=sub_p, analysis=f"Calculate {original_obj} modulo 23. {e}")
                    ans_val = int(code_result['output'].strip())
                    answers.append(str(ans_val))
                except Exception:
                    answers.append("0") # Ultimate fallback
        
        # 3. Format final output string exactly as required
        # Join with '<<horizon>>' and ensure boxed format
        final_items = [f"\\boxed{{{a}}}" for a in answers]
        final_str = "FINAL ANSWER: " + "<<horizon>>".join(final_items)
        return final_str

    def _parse_dependencies(self, text):
        lines = text.split('\n')
        problems = []
        current_prob = ""
        in_prob = False
        for line in lines:
            if "Problem" in line and ":" in line:
                in_prob = True
                current_prob = line
            elif in_prob:
                current_prob += "\n" + line
                if "Problem" in line and not line.strip().startswith("Problem X"): 
                     pass 
                if "Note:" in line or "Note" in line or "Solve all" in line or line.strip() == "":
                     if current_prob.strip():
                         problems.append(current_prob.strip())
                     current_prob = ""
                     in_prob = False
        if current_prob.strip():
            problems.append(current_prob.strip())
        return problems
