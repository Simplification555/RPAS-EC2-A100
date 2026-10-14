from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_6.prompt as prompt_custom
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
        self.referrer = operator.Referrer(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Step 1: Extract raw numeric entities and dependencies first to ensure the model understands the numbers before reasoning.
        extracted_feat = await self.custom(input=problem, instruction="XX_PROMPT")
        
        # Step 2: Solve the problem using the programmer/operator for complex dependencies and arithmetic.
        # We pass the extracted dependencies and the original question to help with correct tracking of variables.
        solution_code, output = await self.programmer(problem=problem, analysis=f"Dependencies extracted: {extracted_feat['response']}")
        
        # Step 3: Refine the output and ensure it matches the strict benchmark format "FINAL ANSWER: X".
        # If the programm's output is clean, the referrer just checks it; if messy, it fixes the format.
        final_result = await self.referrer(input=output['output'], instruction="XXXX_PROMPT")
        
        return final_result, self.llm.get_usage_summary()["total_cost"]
