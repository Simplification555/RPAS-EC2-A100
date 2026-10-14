from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_4.prompt as prompt_custom
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
        1. Extract final answer from problem text or standard format.
        2. Verify using reasoning.
        """
        # Step 1: Attempt to extract the 'FINAL ANSWER: X' from the raw problem text or implicit context to see if a direct value is hidden.
        raw_extraction = await self.custom(
            input=problem,
            instruction=prompt_custom.PROMPT_EXTRACT_ANSWER
        )
        extracted_val = raw_extraction.get('response', '').strip()
        
        # Step 2: Solve the problem to verify or correct the extraction.
        # The input includes the original problem and the extracted candidate to force alignment or re-derivation.
        reasoning_result = await self.custom(
            input=problem + f"\n\nExtracted candidate answer: {extracted_val}",
            instruction=prompt_custom.PROMPT_REASON_VERIFY
        )
        
        final_solution = reasoning_result.get('response', '').strip()
        
        # Step 3: Return the derived verification result.
        # This ensures the model engages the reasoning chain explicitly rather than guessing.
        return final_solution, self.llm.get_usage_summary()["total_cost"]
