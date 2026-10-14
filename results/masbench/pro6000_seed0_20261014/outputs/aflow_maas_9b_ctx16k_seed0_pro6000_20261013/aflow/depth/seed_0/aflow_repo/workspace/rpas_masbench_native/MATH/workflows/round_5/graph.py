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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow with Generate-Verify-Correct logic
        """
        # Step 1: Initial Generation
        solution_attempt = await self.custom(input=problem, instruction=prompt_custom.SOLUTION_GENERATION_PROMPT)
        response_text = solution_attempt['response']
        
        # Step 2: Verification
        verification = await self.custom(input=problem + f"    Current Output: {response_text}", instruction=prompt_custom.VERIFICATION_PROMPT)
        
        # Step 3: Logic Flow for Correction
        if verification['needs_correction'] == "True":
            corrected_attempt = await self.custom(input=problem + f"    Critique: {verification['critique']}\n    Previous Output: {response_text}", instruction=prompt_custom.REGENERATION_WITH_CRITIQUE_PROMPT)
            # Optional Ensemble if both attempts are available or just return corrected
            return corrected_attempt['response'], self.llm.get_usage_summary()["total_cost"]

        # Step 4: Return Final Answer
        # Ensure extraction handles the format strictly
        if "FINAL ANSWER:" in response_text:
            final_part = response_text.split("FINAL ANSWER:")[-1].strip()
            return clean_final_answer(final_part), self.llm.get_usage_summary()["total_cost"]
        else:
            # Fallback if no final tag found but we generated something
            return clean_final_answer(response_text), self.llm.get_usage_summary()["total_cost"]

def clean_final_answer(text: str) -> str:
    part = text.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in text else "".join([c for c in text if c.isdigit()])
    if " <<horizon>> " in part:
        return "FINAL ANSWER: " + part + " <<horizon>>"
    return "FINAL ANSWER: " + part
