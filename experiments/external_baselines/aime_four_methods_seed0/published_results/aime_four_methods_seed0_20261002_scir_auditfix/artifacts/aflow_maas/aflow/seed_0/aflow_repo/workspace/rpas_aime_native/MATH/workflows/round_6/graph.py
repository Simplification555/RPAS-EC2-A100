from typing import Literal
import workspace.rpas_aime_native.MATH.workflows.template.operator as operator
import workspace.rpas_aime_native.MATH.workflows.round_6.prompt as prompt_custom
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
        3. Use Custom (Reasoning) to explicitly state the final numerical answer in a standalone sentence.
        4. Use Custom (Sanitizer) to parse the reasoning output, clean floating-point noise, and wrap in \boxed{}.
        5. Use Custom (Verifier) to double-check the sanitized answer against the problem constraints.
        6. Use ScEnsemble to select the most frequent valid formatted answer.
        """
        # Step 1: Logical Analysis
        analysis = await self.custom(input=problem, instruction=prompt_custom.ANALYSIS_PROMPT)
        
        # Step 2: Code Execution for precise calculation
        code_result = await self.programmer(problem=problem, analysis=analysis['response'])
        
        # Step 3: Reasoning (Critical Addition)
        # Force the model to output the final answer as a clear sentence to aid extraction
        reasoning_result = await self.custom(
            input=f"Problem: {problem}\nAnalysis: {analysis['response']}\nCode Output: {code_result['output']}", 
            instruction=prompt_custom.REASONING_PROMPT
        )
        
        # Step 4: Sanitize Output (Enhanced)
        # Pass the reasoning text to a sanitizer that extracts the number and formats it
        sanitized_result = await self.custom(
            input=reasoning_result['response'], 
            instruction=prompt_custom.SANITIZE_PROMPT
        )
        
        # Step 5: Verify Logic Consistency
        verified_result = await self.custom(
            input=f"Analysis: {analysis['response']}, Sanitized Answer: {sanitized_result['response']}", 
            instruction=prompt_custom.VERIFY_PROMPT
        )
        
        # Step 6: Ensemble for final answer selection
        solutions = await self.sc_ensemble(
            solutions=[verified_result['response']], 
            problem=f"{problem} | Analysis: {analysis['response']} | Code Output: {code_result['output']}"
        )
        
        return solutions['response'], self.llm.get_usage_summary()["total_cost"]
