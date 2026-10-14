from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_2.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        """
        Traces dependency paths, generates code for precise calculation, verifies the logic, and extracts the final answer.
        """
        # Step 1: Use Custom to identify dependency paths and initial values, formatting the output for clarity.
        trace_init = await self.custom(input=problem, instruction=prompt_custom.DEPENDENCY_ANALYSIS)
        
        # Step 2: Use Programmer to automatically generate and execute Python code to solve the arithmetic chain.
        # This handles the modular arithmetic (Z_23) and sequential dependencies explicitly in code.
        code_result = await self.programmer(problem=problem, analysis=trace_init['response'])
        
        # Step 3: Use Custom to verify the calculated result and ensure it matches the required format.
        # If the code execution was successful, we use the code output as the primary source.
        solution = code_result['output']
        if "Final Answer" in solution or solution.strip().isdigit():
            final_check = await self.custom(input=solution, instruction=prompt_custom.FORMAT_VERIFICATION)
            return final_check['response'], self.llm.get_usage_summary()["total_cost"]
        else:
            # Fallback if code output parsing failed or format was wrong
            return solution, self.llm.get_usage_summary()["total_cost"]
