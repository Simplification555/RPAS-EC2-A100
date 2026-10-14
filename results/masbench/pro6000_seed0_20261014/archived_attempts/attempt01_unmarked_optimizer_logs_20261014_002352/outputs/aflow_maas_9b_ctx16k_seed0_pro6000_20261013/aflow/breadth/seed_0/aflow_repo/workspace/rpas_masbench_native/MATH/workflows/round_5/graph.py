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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow with strict formatting enforcement
        """
        # Construct a highly specific prompt that enforces the logic and output format
        # ensuring the final string is 'FINAL ANSWER: X' to match extraction patterns.
        prompt_instruction = """
        You are an expert solver for MASBench iGSM arithmetic problems.
        Rules:
        1. All quantities are in Z_23 (modulo 23).
        2. If a variable is undefined, treat it as 0.
        3. Step-by-step reasoning is allowed, but the final output MUST be EXACTLY one line at the very end: "FINAL ANSWER: <number>".
        4. No other text should appear after "FINAL ANSWER: <number>".
        
        Problem: {problem}
        
        Reasoning and Final Answer:
        """
        
        response = await self.custom(input=problem + prompt_instruction, instruction="")
        return response['response'], self.llm.get_usage_summary()["total_cost"]
