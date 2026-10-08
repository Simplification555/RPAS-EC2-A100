from typing import Literal
import workspace.rpas_aime_native.MATH.workflows.template.operator as operator
import workspace.rpas_aime_native.MATH.workflows.round_4.prompt as prompt_custom
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
        Optimized workflow:
        1. Use Custom to perform initial logical analysis and define constraints.
        2. Use Programmer to write and execute Python code to solve the derived equations.
        3. Use Custom to strictly format the result into \boxed{} and validate it.
        4. Use ScEnsemble to select the most frequent valid formatted answer.
        """
        # Step 1: Logical Analysis
        analysis = await self.custom(input=problem, instruction=prompt_custom.ANALYSIS_PROMPT)
        
        # Step 2: Code Execution for precise calculation
        code_result = await self.programmer(problem=problem, analysis=analysis['response'])
        
        # Step 3: Format & Verify (Critical Fix)
        # Ensure the output is strictly \boxed{number}
        formatted_result = await self.custom(
            input=code_result['output'], 
            instruction=prompt_custom.FORMAT_VERIFY_PROMPT
        )
        
        # Step 4: Ensemble for final answer selection
        # Pass the verified formatted string to the ensemble
        solutions = await self.sc_ensemble(
            solutions=[formatted_result['response']], 
            problem=f"{problem} | Analysis: {analysis['response']} | Code Output: {code_result['output']}"
        )
        
        return solutions['response'], self.llm.get_usage_summary()["total_cost"]
