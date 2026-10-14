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
        self.sc = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow with ensemble voting and strict formatting
        """
        # Interpret final instructions and enforce exact separators for the formatted content
        instruction = f"Solve the problem. Format the final answer strictly as <VAL1>><<<<horizon>><<<<horizon>>VAL2>. If multiple values exist, use the LAST verified injection. Output ONLY the formatted string."
        
        # Generate multiple candidate solutions with different seeds (controlled by critical thinking)
        candidates = []
        for seed in [100, 200, 300]:
            # Append random seed to input to force different reasoning paths per seed
            current_problem = f"{problem} (Seed: {seed})"
            resp = await self.custom(input=current_problem, instruction=instruction)
            if resp and 'response' in resp:
                candidates.append(resp['response'])
        
        if not candidates:
            return "", 0.0

        # Use ensemble to select the most robust answer
        final_result = await self.sc(solutions=candidates, problem=problem)
        return final_result['response'], self.llm.get_usage_summary()["total_cost"]
