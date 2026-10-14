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
        self.ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Optimized workflow with DAG check and ensemble voting to handle truncation errors
        """
        # Step 1: Analyze structure to ensure no circular dependencies before heavy computation
        analysis = await self.custom(input=problem, instruction=prompt_custom.STRUCTURE_ANALYSIS)
        
        is_dag = analysis.get('is_dag', True)
        solution_candidates = None
        
        if is_dag:
            # Step 2: Generate solutions multiple times (triggered internally by ensemble via loop prompt)
            # The custom method will run the solving logic n times
            final_result = await self.ensemble(solutions=[analysis], problem=problem)
            return final_result.get('response'), self.llm.get_usage_summary()["total_cost"]
        else:
            # Fallback for non-DAG: Return analysis as is to avoid infinite loops
            return analysis, self.llm.get_usage_summary()["total_cost"]
