from typing import Literal
import workspace.rpas_aime_native.MATH.workflows.template.operator as operator
import workspace.rpas_aime_native.MATH.workflows.round_8.prompt as prompt_custom
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
        Optimized workflow with Self-Correction:
        1. Use Custom to perform initial logical analysis and define constraints.
        2. Use Programmer to write and execute Python code to solve the derived equations.
        3. Use Custom (Reasoning) to derive the final numerical answer.
        4. Use Custom (Self-Correction) to critique the reasoning and recalculate if errors are found.
        5. Use Custom (Format Enforcement) to strictly force the output into a single \boxed{value} format.
        6. Use Custom (Validator) to verify the format is correct and contains a valid number.
        7. Use ScEnsemble to select the most frequent valid formatted answer.
        """
        # Step 1: Logical Analysis
        analysis = await self.custom(input=problem, instruction=prompt_custom.ANALYSIS_PROMPT)
        
        # Step 2: Code Execution for precise calculation
        code_result = await self.programmer(problem=problem, analysis=analysis['response'])
        
        # Step 3: Initial Reasoning
        reasoning_result = await self.custom(
            input=f"Problem: {problem}\nAnalysis: {analysis['response']}\nCode Output: {code_result['output']}", 
            instruction=prompt_custom.REASONING_PROMPT
        )
        
        # Step 4: Self-Correction (Critical Addition)
        # The model critiques its own previous reasoning and recalculates
        correction_result = await self.custom(
            input=f"Original Reasoning: {reasoning_result['response']}\nProblem: {problem}\nAnalysis: {analysis['response']}\nCode Output: {code_result['output']}", 
            instruction=prompt_custom.SELF_CORRECTION_PROMPT
        )
        
        # Step 5: Format Enforcement
        formatted_result = await self.custom(
            input=correction_result['response'], 
            instruction=prompt_custom.FORMAT_PROMPT
        )
        
        # Step 6: Validate Format
        validated_result = await self.custom(
            input=formatted_result['response'], 
            instruction=prompt_custom.VALIDATE_PROMPT
        )
        
        # Step 7: Ensemble for final answer selection
        solutions = await self.sc_ensemble(
            solutions=[validated_result['response']], 
            problem=f"{problem} | Analysis: {analysis['response']} | Code Output: {code_result['output']} | Corrected Reasoning: {correction_result['response']}"
        )
        
        return solutions['response'], self.llm.get_usage_summary()["total_cost"]
