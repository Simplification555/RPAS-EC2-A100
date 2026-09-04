from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_3.prompt as prompt_custom
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
        self.cost_manager = CostManage(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Optimized workflow: Generate solutions, ensemble, test, and refine based on specific errors.
        """
        # Step 1: Generate multiple distinct solutions
        solutions_list = []
        for i in range(3):
            sol = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=f"Generate a unique Python solution for the following problem. Variation {i+1}: {problem}")
            solutions_list.append(sol['response'])
        
        # Step 2: Use ScEnsemble to select the most consistent solution
        best_solution = await self.sc_ensemble(solutions=solutions_list, problem=problem)
        final_solution = best_solution['response']
        
        # Step 3: Test and Refine Loop
        max_retries = 2
        for retry in range(max_retries):
            test_result = await self.test(problem=problem, solution=final_solution, entry_point=entry_point)
            
            if test_result['result']:
                return final_solution, self.llm.get_usage_summary()["total_cost"]
            
            # If test fails, check cost before refining
            current_cost = self.llm.get_usage_summary()["total_cost"]
            if not self.cost_manager.is_within_budget(current_cost):
                return final_solution, current_cost
            
            # Refine: Pass the specific error message to generate a targeted fix
            error_msg = test_result.get('error', 'Test failed')
            refine_instruction = f"Analyze the following error: '{error_msg}'. Generate a corrected Python solution that specifically addresses this error."
            new_sol = await self.custom(input=problem, instruction=refine_instruction)
            final_solution = new_sol['response']
        
        # Fallback if all retries fail
        return final_solution, self.llm.get_usage_summary()["total_cost"]
