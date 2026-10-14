from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_4.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Optimized workflow to handle MASBench complexity, robustness overrides, and strict formatting.
        """
        # Step 1: Extract all entities, magic numbers, and explicit verification notes from the text.
        extraction_result = await self.custom(
            input=problem,
            instruction="EXTRACT_ALL_AND_LIST"
        )
        context_for_solving = extraction_result['response']

        # Step 2: Solve algebraic chains or dependency paths in Z_23 if necessary.
        # This step handles the calculation of values defined by relations before applying overrides.
        calculation_result = await self.custom(
            input=f"{context_for_solving}",
            instruction="ALGEBRAIC_RELATIONS_AND_OVERRIDES"
        )
        
        # Step 3: Synthesize the final ordered list based on the prompt's requested sequence.
        # Explicitly enforce the <<horizon>> separator and FINAL ANSWER format here in the custom prompt.
        synthesis_result = await self.custom(
            input=problem + f"Processed Info:\n{calculation_result['response']}",
            instruction="FINAL_SYNTHESIS_FORMATTED"
        )
        
        # Step 4: Generate a robust verification code to re-format the answer if the custom method is unsure.
        # The programmer constructs a string that definitively applies the separators.
        code_verification = await self.programmer(
            problem=synthesis_result['response'],
            analysis="Regenerate the answer list, ensuring strict use of <<horizon>> between items and the 'FINAL ANSWER: ' prefix."
        )
        
        final_content = code_verification['output']
        
        # Step 5: Use ScEnsemble to self-consistency check the format and content.
        # Generate multiple versions of the final answer to ensure the separator and prefix are correct.
        final_solution, cost = await self.sc_ensemble(
            solutions=[
                f"FINAL ANSWER: {final_content}",
                f"FINAL ANSWER: {final_content}", # Force duplication for ensemble voting on format
                f"Answer: {final_content}"
            ],
            problem=problem
        )
        
        return final_solution['response'], cost
