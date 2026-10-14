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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Optimized pipeline: Extract constants via code -> Solve logic -> Aggregate results.
        """
        # Step 1: Use Programmer to extract all numeric values and constants into a format safe for Z_23 calculation.
        # This prevents the model from hallucinating values for undefined variables by forcing code extraction.
        code_result = await self.llm.run_code(problem=input=problem, task="Extract all explicit numeric values and define variables for Z_23 modulo arithmetic. Return only the Python code to extract these values into a list of tuples (variable_name, value) where value is int(0-22).")
        
        # Step 2: Use Custom with a strict instruction to solve the dependency chain using the extracted values.
        # The prompt now explicitly references the output of the Python extraction to ground the reasoning.
        solution_list = []
        attempts = 3
        i = 0
        while i < attempts:
            # Inject the extracted variables into the prompt to ground the reasoning
            variable_str = f"Available constants for Z_23: {code_result['output']}".strip().replace('\n', ' ')
            prompt_inst = f"""
            Solve the following MASBench iGSM problem.
            CRITICAL INSTRUCTIONS:
            1. You MUST perform all arithmetic in Z_23 (mod 23).
            2. Use the following extracted constants as the ONLY source of truth: "{variable_str}". 
            3. If a variable is mentioned in a dependency chain but not in the constants, assume it resolves to 0 or trace the chain until a numeric base is found. Do NOT hallucinate new values.
            4. Output your step-by-step derivation briefly.
            5. END your response on a separate line with exactly: FINAL ANSWER: [integer], where [integer] is the final result modulo 23.
            
            Problem: {problem}
            """
            response = await self.custom(input=prompt_inst, instruction="")
            
            # Basic validation: ensure format before adding to list
            if "FINAL ANSWER:" in response['response']:
                solution_list.append(response['response'].split('FINAL ANSWER:')[-1].strip())
            
            # Simple pressure to switch reasoning path if first few are identical/ stuck (optional heuristic)
            i += 1
            
        # Step 3: Use sc_ensemble to pick the most frequent correct answer from the multiple attempts.
        final_result_dict = await self.sc_ensemble(solutions=solution_list, problem=problem)
        
        final_ans = final_result_dict['response']
        # Enforce final formatting in case ensemble returned an extra tag
        if "FINAL ANSWER:" in final_ans:
            final_ans = final_ans.split("FINAL ANSWER:")[-1].strip()
            
        return final_ans, self.llm.get_usage_summary()["total_cost"]
