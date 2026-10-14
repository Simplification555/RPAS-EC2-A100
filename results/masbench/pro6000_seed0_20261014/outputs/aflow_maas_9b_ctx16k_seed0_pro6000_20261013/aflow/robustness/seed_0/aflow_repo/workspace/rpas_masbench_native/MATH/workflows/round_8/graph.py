from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_8.prompt as prompt_custom
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
        self.ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # Step 1: Extract verified magic numbers and specific facts (Note: verify...)
        verified_data = await self.custom(
            input=problem, 
            instruction=prompt_custom.ROBUST_EXTRACTION_PROMPT
        )
        
        # Step 2: Attempt to derive algebraic values if needed (to compare/all)
        algebraic_hint = await self.custom(
            input=problem, 
            instruction=prompt_custom.ALGEBRAIC_HINT_PROMPT
        )
        
        # Step 3: Synthesis via Ensemble. We pass the raw verified string as the ground truth candidate
        # because the rules state to prioritize raw retrieval. The ensemble ensures if multiple notes exist,
        # the logic holds, but primarily it validates the formatting and ordering.
        # We feed the initial verification result and the question context.
        final_result = await self.ensemble(
            solutions=[verified_data['response']], 
            problem=f"{problem} - Target Format: 'FINAL ANSWER: NUM1<<horizon>>NUM2...'"
        )
        
        # Step 4: If ensemble returns the same result or a slightly formatted one, return it.
        # Ensure the final content is just the line "FINAL ANSWER: ..."
        return final_result['response'], self.llm.get_usage_summary()["total_cost"]
