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
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow with Python code execution, strict formatting, and extraction.
        """
        # Step 1: Generate Python code to solve the problem using Z_23 arithmetic
        # Modified Instruction: Enforce strict "FINAL ANSWER: X" format at the very end, no code blocks.
        python_code, python_output = await self.programmer(
            problem=problem, 
            analysis="Translate the word problem into Python code that performs arithmetic modulo 23. The script MUST print ONLY one line at the end formatted exactly as: FINAL ANSWER: [numerical_answer]. Do not print code, no markdown ticks (```), no explanations, no variables. Just that one line."
        )
        
        # Step 2: Retry with slight variation to ensure robustness if format was missed
        python_code_retry, python_output_retry = await self.programmer(
            problem=problem, 
            analysis="Write a Python script to parse terms and calculate results strictly modulo 23. The final print statement must be exactly: print(f\"FINAL ANSWER: {result}\"). Ensure no other text or code blocks are printed."
        )
        
        potential_solutions_unformatted = [python_output, python_output_retry]
        
        # Step 3: Use ScEnsemble to select the most frequent valid answer string
        # The LLMs usually output the variable `x` or similar, we rely on strict formatting in Step 1/2 to get the string "FINAL ANSWER: X"
        final_result_unchecked = await self.sc_ensemble(solutions=potential_solutions_unformatted, problem=problem)
        
        # Step 4: Use Custom method to post-process and ensure the final output matches the rigid requirement "FINAL ANSWER: X"
        # The 'instruction' here forces the LLM to act as a formatter/extractor.
        # Input combines the original problem to ensure context, and the raw solution.
        instruction_extract = prompt_custom.ExtractAndFormat
        custom_output = await self.custom(
            input=problem + "\nGenerated Solution Candidates:\n" + final_result_unchecked['response'], 
            instruction=instruction_extract
        )
        
        # Return the processed string directly. 
        # If the custom step failed to add "FINAL ANSWER:", it will raise an error or return raw, 
        # but the prompt is designed to fail gracefully or force the format.
        return custom_output['response'], self.llm.get_usage_summary()["total_cost"]
