from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_5.prompt as prompt_custom
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
        Implementation of the workflow with Python code execution and ensemble voting
        """
        # Step 1: Generate Python code to solve the problem using Z_23 arithmetic
        python_code, python_output = await self.programmer(problem=problem, analysis="Translate the word problem into Python code that performs arithmetic modulo 23 (using `x % 23` for results) and prints the final integer answer.")
        
        # Step 2: Execute multiple independent code generations to ensure robustness
        potential_solutions = [python_output]
        
        # Retry with slight variation in instruction to generate a second potential solution
        python_code_retry, python_output_retry = await self.programmer(problem=problem, analysis="Write a Python script that parses the text, defines variables, performs all calculations modulo 23 carefully, and outputs the FINAL ANSWER as an integer.")
        potential_solutions.append(python_output_retry)
        
        # Step 3: Use ScEnsemble to select the most frequent answer among the generated solutions
        final_result = await self.sc_ensemble(solutions=potential_solutions, problem=problem)
        
        return final_result['response'], self.llm.get_usage_summary()["total_cost"]
