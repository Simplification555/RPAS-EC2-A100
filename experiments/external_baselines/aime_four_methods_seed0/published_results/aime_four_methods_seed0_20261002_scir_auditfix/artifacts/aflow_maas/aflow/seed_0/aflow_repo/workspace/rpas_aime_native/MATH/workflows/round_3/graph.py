from typing import Literal
import workspace.rpas_aime_native.MATH.workflows.template.operator as operator
import workspace.rpas_aime_native.MATH.workflows.round_3.prompt as prompt_custom
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
        3. Use Custom to format the final answer strictly as \boxed{value} and verify validity.
        4. Use ScEnsemble to select the most consistent formatted answer.
        """
        # Step 1: Logical Analysis
        analysis = await self.custom(input=problem, instruction=prompt_custom.ANALYSIS_PROMPT)
        
        # Step 2: Code Execution for precise calculation
        code_result = await self.programmer(problem=problem, analysis=analysis['response'])
        
        # Step 3: Format and Verify the answer
        # We pass the code output and the original problem to ensure the final string is valid LaTeX
        formatted_result = await self.custom(
            input=f"Code Output: {code_result['output']}\nProblem: {problem}", 
            instruction=prompt_custom.FORMAT_VERIFY_PROMPT
        )
        
        # Step 4: Ensemble for final answer selection
        solutions = await self.sc_ensemble(
            solutions=[formatted_result['response']], 
            problem=f"{problem} | Analysis: {analysis['response']} | Formatted Output: {formatted_result['response']}"
        )
        
        return solutions['response'], self.llm.get_usage_summary()["total_cost"]
