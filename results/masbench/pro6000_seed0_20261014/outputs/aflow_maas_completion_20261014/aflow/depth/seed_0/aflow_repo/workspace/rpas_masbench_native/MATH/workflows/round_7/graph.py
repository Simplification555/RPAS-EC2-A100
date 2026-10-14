from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_7.prompt as prompt_custom
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
        # Generate multiple reasoning paths to ensure consistency
        solutions = await self.custom(
            input=problem, 
            instruction=prompt_custom.VERIFICATION_PROMPT
        )
        # Iterate through the list of solutions provided by the verifier to check for consistency
        # The Custom method above returns a list of [response] dictionaries in 'response' key when used with the specific prompt
        # Assuming the custom logic returns a list in the response structure based on the instruction
        raw_results = solutions['response']
        
        if isinstance(raw_results, list):
            # ScEnsemble takes the list of string solutions and returns the most frequent one
            final_response = await self.sc_ensemble(solutions=raw_results, problem=problem)
        else:
            final_response = raw_results

        # Ensure strict formatting parse
        return final_response['response'], self.llm.get_usage_summary()["total_cost"]
