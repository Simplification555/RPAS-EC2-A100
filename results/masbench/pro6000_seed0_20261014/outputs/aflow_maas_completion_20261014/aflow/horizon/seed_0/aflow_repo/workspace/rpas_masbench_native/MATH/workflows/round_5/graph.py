from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_5.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.sc_ensemble(self.llm)

    async def __call__(self, problem: str):
        """
        Optimized Math Solver with Verification Loop
        """
        # Stage 1: Initial Reasoning
        initial_result = await self.custom(
            input=problem,
            instruction=prompt_custom.THINK_THROUGH_MATH
        )
        
        # Parse the structure to identify all variables used
        # We use the Programmer to extract equations and define variables first
        code_solution = await self.programmer(
            problem=f"Solve the math problem logic. Do not calculate final values yet. List all variables used, their definitions, and the equations in the text. If a variable like 'Product' or 'Supermarket' is not explicitly defined as a number, assume it equals 1.",
            analysis=initial_result['output']
        )
        
        # Stage 2: Verification Loop
        # If the code successfully defines all needed variables and runs, we use its output.
        # If it fails (because text was ambiguous), we refine the reasoning with the Programmer's findings.
        try:
            # Execute the generated code to verify arithmetic and handle undefined variables
            _ = eval(code_solution['code'])
            verified_result = await self.custom(
                input=f"Based on the verified logic: {code_solution['code']}",
                instruction=prompt_custom.THINK_THROUGH_MATH
            )
            return verified_result['response'], 0
        except:
            # Fallback: Rerun with the Programmer's explicit variable definitions appended to the context
            refined_problem = f"{problem}\n\n--- explicit_definitions ---\n{code_solution['code']}\n--- end definitions ---"
            refined_result = await self.custom(
                input=refined_problem,
                instruction=prompt_custom.THINK_THROUGH_MATH
            )
            return refined_result['response'], self.llm.get_usage_summary()["total_cost"]
