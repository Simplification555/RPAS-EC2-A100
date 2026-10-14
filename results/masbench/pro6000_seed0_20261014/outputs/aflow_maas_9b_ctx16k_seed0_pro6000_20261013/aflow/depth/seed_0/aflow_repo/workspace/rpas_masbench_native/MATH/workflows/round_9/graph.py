from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_9.prompt as prompt_custom
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
        1. Generate multiple candidate solutions using self-consistency (sc_ensemble).
        2. Verify the best candidate using rigorous reasoning.
        """
        # Step 1: Use ScEnsemble to generate diverse candidate answers.
        # The problem is passed directly to the ensemble to trigger multiple reasoning paths.
        ensemble_results = await self.sc_ensemble(
            solutions=[problem],  # Passing the problem question sequence to trigger computations
            problem=problem
        )
        # Extract the top N generated candidates (ensuring they are non-empty strings)
        candidates = [r['response'] for r in ensemble_results['solutions'] if r['response']]
        
        if not candidates:
            # Fallback to empty list if ensemble fails completely
            candidates = []

        # Step 2: Perform rigorous reasoning on the original problem to verify the best candidate.
        # Aggregate all candidate responses into a single verification context.
        verification_context = "\n\n".join(f"Candidate Solution {i}: {c}" for i, c in enumerate(candidates))
        
        reasoning_result = await self.custom(
            input=problem + f"\n\n=== Generated Candidate Solutions Below ===\n{verification_context}\n\nYou must rigorously verify the math and dependencies in the candidates above.",
            instruction=prompt_custom.PROMPT_REASON_FINAL
        )
        
        final_solution = reasoning_result.get('response', '').strip()
        
        # Step 3: Return the derived verification result.
        return final_solution, self.llm.get_usage_summary()["total_cost"]
