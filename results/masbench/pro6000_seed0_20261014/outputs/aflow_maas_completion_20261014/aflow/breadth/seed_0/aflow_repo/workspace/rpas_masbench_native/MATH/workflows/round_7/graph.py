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
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # Path 1: Precise Arithmetic for Variables (Python)
        calc_result = await self.programmer(input=problem, analysis="Identify all variables requiring arithmetic operations in Z_23. Write Python code to parse the text dependency chain, compute values modulo 23, and print ONLY the final result as an integer.")
        
        # Path 2: Logical Mapping and Format Extraction (Custom)
        # Use the calculated result as context to ensure the LLM focuses on correct variable mapping (e.g., resolving 'Bone' to 'Distal Radials')
        logic_result = await self.custom(
            instruction=prompt_custom.FINAL_VERIFICATION_PROMPT,
            input=f"Calculated Value: {calc_result['output']}\n\n{problem}"
        )
        
        # Path 3: Ensemble (Simulated as single for graph complexity limit, but logic buffered)
        # If re-evaluation occurs, aggregate. For this linear graph, we trust the Python calculation + verification.
        final_response = logic_result['response']
        
        return final_response, self.llm.get_usage_summary()["total_cost"]
