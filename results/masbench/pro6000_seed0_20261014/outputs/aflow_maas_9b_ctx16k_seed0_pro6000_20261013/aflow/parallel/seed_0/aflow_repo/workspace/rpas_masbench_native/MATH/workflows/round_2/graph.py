from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_2.prompt as prompt_custom
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
        """
        Implementation of the workflow
        """
        derivations = await self.custom(input=problem, instruction=prompt_custom.DERIVATION_PROMPT)
        # Concatenate all derivation steps to form a robust context for the ensemble
        context = str(derivations['response'])
        
        # Use ensemble to vote and generate the formatted final answer
        formatted_result = await self.ensemble(input=context, instruction=prompt_custom.FINAL_ANSWER_PROMPT, problem=problem)
        
        return formatted_result['response'], self.llm.get_usage_summary()["total_cost"]
