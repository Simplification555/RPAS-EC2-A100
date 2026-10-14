from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_4.prompt as prompt_custom
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
        self.custom = operator.Custom(self.llm)
        self.ensemble = operator.sc_ensemble(self.llm, "question")

    async def __call__(self, problem: str):
        solution = await self.programmer(problem=problem, analysis="Extract all equations and solve for the final variable. Return 'NULL' if any critical variable is undefined. Output must be pure Python code wrapped in triple backticks.")
        code = solution.get('code', "")
        
        result = None
        if "FAIL" in code or "NULL" in code:
            final_text = f"CRITICAL ERROR IN EQUATION RESOLUTION: {solution.get('output', 'Unknown')}."
        else:
            try:
                # Execute the generated code safely to get the answer
                exec_env = {'__builtins__': {k: v for k, v in __builtins__.items() if not k.startswith('_')}}
                exec(code, exec_env)
                if 'answer' in exec_env:
                    result = str(exec_env['answer'])
                else:
                    raise ValueError("Variable 'answer' not generated.")
            except Exception as e:
                final_text = f"CODE EXECUTION ERROR: {str(e)}. Attempting natural language resolution."
        
        # If code execution failed or returned error, force a natural language check
        if final_text != None and "FINAL ANSWER" not in final_text or (result is None):
            prompt_input = f"Problem: {problem}\nProgrammer Output: {solution.get('output', 'Auto-extraction failed')}\nReasoning:\nThe previous code execution failed or returned an error. You must identify if the variable is actually 0 or derived from a cycle. Re-evaluate the dependencies carefully.\n\nFinal Output Format:\nFINAL ANSWER: <number>"
            response = await self.custom(input=prompt_input, instruction="Solve the arithmetic carefully. All math in Z_23.")
            final_text = response['response']
        
        # Aggregate potential duplicates or format fixes
        final_output = await self.ensemble(solutions=[final_text], problem=problem)
        return final_output['response'], self.llm.get_usage_summary()["total_cost"]
