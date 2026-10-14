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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Step 1: Solve the problem (solve step)
        solution_step = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        
        # Step 2: Format the final answer to match the strict required format
        formatted_response = await self.custom(
            input=problem, 
            instruction=prompt_custom.FORMAT_PROMPT.format(solution=solution_step['response'])
        )
        
        return formatted_response['response'], self.llm.get_usage_summary()["total_cost"]
