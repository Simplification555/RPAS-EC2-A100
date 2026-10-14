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
        self.ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Sequential processing to handle independent subproblems and enforce formatting.
        """
        # Split or process problems. If input is a list of strings or delimited text, handle here.
        # Based on logs, input string contains "Problem 1: ... Problem 2: ..."
        # We will process them sequentially to manage context per problem.
        if not isinstance(problem, list):
            problem_list = [problem]
        else:
            problem_list = problem
            
        final_results = []
        
        for i, p in enumerate(problem_list):
            # Step 1: Solve current problem with strict instructions
            # We pass the specific problem text to ensure context isolation
            solution = await self.custom(solution)
            final_results.append(solution['response'])
        
        # Step 2: Aggregate results with strict separator
        # Format: ans1<<horizon>>ans2...
        # Ensure no hallucinations by validating length if possible, but custom output should handle it.
        response_text = "FINAL ANSWER: " + " <<horizon>> ".join(final_results)
        
        # Note: The `ScEnsemble` was attempted in previous logs but failed due to state aggregation.
        # We rely on the strict Custom instruction in the loop to generate the list correctly, 
        # then join externally. If the Custom method supports generating multiple outputs in one call, 
        # we could use it here, but the loop approach is safer for dependency resolution found in logs.
        
        return response_text, self.llm.get_usage_summary()["total_cost"]
