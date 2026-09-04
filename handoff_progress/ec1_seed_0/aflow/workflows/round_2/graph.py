from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_2.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Optimized workflow: Generate multiple solutions, select the best via ensemble, and validate with tests.
        """
        # Step 1: Generate multiple distinct solutions using the custom code generator
        # We generate 3 variations to increase the chance of finding a correct one
        solutions_list = []
        for i in range(3):
            sol = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=f"Generate a unique Python solution for the following problem. Variation {i+1}: {problem}")
            solutions_list.append(sol['response'])
        
        # Step 2: Use ScEnsemble to select the most consistent/best solution from the list
        best_solution = await self.sc_ensemble(solutions=solutions_list, problem=problem)
        
        # Step 3: Test the selected solution. If it fails, regenerate a new solution and test again (max 2 retries)
        final_solution = best_solution['response']
        for retry in range(2):
            test_result = await self.test(problem=problem, solution=final_solution, entry_point=entry_point)
            if test_result['result']:
                return final_solution, self.llm.get_usage_summary()["total_cost"]
            else:
                # If test fails, generate a new solution to replace the failed one
                new_sol = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="Generate a corrected Python solution based on the previous failure.")
                final_solution = new_sol['response']
        
        # Fallback if all retries fail
        return final_solution, self.llm.get_usage_summary()["total_cost"]
