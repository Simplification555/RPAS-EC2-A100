from typing import Literal
import workspace.HotpotQA.workflows.template.operator as operator
import workspace.HotpotQA.workflows.round_2.prompt as prompt_custom
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
        # Step 1: Generate reasoning and a structured answer candidate
        reasoning_response = await self.custom(
            input=problem, 
            instruction=getattr(prompt_custom, 'REASONING_PROMPT', 'Use only the supplied context to answer the question. Reason briefly, then provide the concise final answer.

Input: {input}')
        )
        
        # Step 2: Extract the final answer from the reasoning response into a strict JSON format
        # This ensures the extraction logic can find the 'answer' key
        final_response = await self.custom(
            input=reasoning_response['response'], 
            instruction=getattr(prompt_custom, 'EXTRACT_PROMPT', 'Use only the supplied context to answer the question. Reason briefly, then provide the concise final answer.

Input: {input}')
        )
        
        return final_response['response'], self.llm.get_usage_summary()["total_cost"]
