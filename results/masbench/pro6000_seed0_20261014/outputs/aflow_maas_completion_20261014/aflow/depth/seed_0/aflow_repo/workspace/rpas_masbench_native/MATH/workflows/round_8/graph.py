from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_8.prompt as prompt_custom
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
        Implementation of the workflow with ensemble verification
        """
        # Step 1: Generate the final answer using a strict extraction prompt
        final_result = await self.custom(
            input=problem, 
            instruction=prompt_custom.STRICT_SOLVER_PROMPT
        )
        
        # Step 2: Verification via Ensemble (ScEnsemble)
        # Generate 3 independent reasoning attempts to ensure consensus
        attempts = [
            await self.custom(input=problem, instruction="Solve carefully following modulo arithmetic."),
            await self.custom(input=problem, instruction="Solve step-by-step, treating undefined variables as 0."),
            await self.custom(input=problem, instruction="Verify the logic chain and calculate the final result.")
        ]
        
        # Use ScEnsemble to pick the most frequent/correct result
        ensemble_result = await self.sc_ensemble(
            solutions=[a['response'] for a in attempts], 
            problem=problem
        )
        
        # Step 3: Return the formatted answer if it follows the strict format; otherwise the ensemble result
        if 'FINAL ANSWER:' in final_result['response'].strip():
            return final_result['response'], self.llm.get_usage_summary()["total_cost"]
        else:
            return ensemble_result['response'], self.llm.get_usage_summary()["total_cost"]
