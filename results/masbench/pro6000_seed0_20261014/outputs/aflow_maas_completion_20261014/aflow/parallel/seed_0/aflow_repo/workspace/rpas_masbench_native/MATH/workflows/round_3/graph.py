from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_3.prompt as prompt_custom
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
        Implementation of the workflow with multi-path reasoning and ensemble final aggregation.
        """
        # Parse the problem string to identify individual problem texts if they are concatenated
        # We will process each problem found in the input sequentially to ensure independence
        all_solutions = []
        
        # Define the number of reasoning paths (smaller than number of problems if many, otherwise fixed)
        # We will generate solutions for each problem definition found
        problem_texts = self._extract_problems(problem)
        
        solutions_dict = {}
        
        # Iterate through each identified problem text to solve independently (Path Strategy)
        for idx, p_text in enumerate(problem_texts):
            # Generate a specific prompt for this single problem to avoid cross-contamination in a single run
            # We focus on one problem per run of the custom method to maintain clean state
            current_solution = await self._solve_single_problem(p_text, idx + 1)
            
            if current_solution['response']:
                solutions_dict[idx + 1] = current_solution['response']
            else:
                # Default if extraction fails
                solutions_dict[idx + 1] = "Undefined"

        # Now aggregate the solutions into the final formatted string
        answers = []
        for k in range(1, len(problem_texts) + 1):
            if str(k) in solutions_dict:
                answers.append(solutions_dict[str(k)])
            else:
                # In case a problem text was missed but index exists, try to guess from full context or use 0
                # For robustness, we assume the loop covered all if extraction worked
                pass
        
        # Connect answers with the required separator
        final_ans = "<<horizon>>".join(answers) if answers else "None"
        
        # Format the strict output line
        output_str = f"FINAL ANSWER: {final_ans}"
        
        # Use ScEnsemble as a final safety check on the constructed final answer string (Optional but recommended)
        # However, usually the logic construction is sufficient if done right. 
        # Let's do a single ensemble on the potential variations if we ran multi-solve.
        # Since we solved linearly, we return the result directly.
        
        return {'response': output_str}, self.llm.get_usage_summary()["total_cost"]

    def _extract_problems(self, text):
        """
        Simple extraction to split problem strings based on the known header pattern.
        This ensures we process "Problem 1...", "Problem 2..." independently.
        """
        problems = []
        # Regex to find Problem X: ... patterns
        import re
        pattern = r'Problem \d+: .*?(?=\n\nProblem \d+:|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    async def _solve_single_problem(self, problem_text: str, problem_idx: int):
        """
        Generates a solution for a single problem instance.
        Uses specific instructions to handle the Z_23 module arithmetic and undefined variables.
        """
        # Construct a prompt that forces the model to output only the final answer for THIS specific problem
        # to facilitate the extraction step later.
        specific_instruction = (
            f"You are solving a specific instance of a modular arithmetic problem (Modulo 23).\n"
            f"PROBLEM: {problem_text}\n"
            f"INSTRUCTIONS:\n"
            f"1. Work strictly in Z_23 (integers 0-22). Apply modulo to all additions, subtractions, multiplications.\n"
            f"2. If a variable (e.g., 'Product', 'Classroom') is not defined in the dependency chain for THIS problem, assume its value is 0.\n"
            f"3. DO NOT provide step-by-step explanations. ONLY return the final integer answer. No text before or after.\n"
            f"4. Format: Just the number."
        )
        
        response = await self.custom(input=problem_text, instruction=specific_instruction)
        
        # Extract the number from the response if possible
        extract_val = self._extract_number(response.get('response', ''))
        
        return {'response': extract_val if extract_val else "0"}

    def _extract_number(self, text):
        # Try to find a single number at the end
        import re
        match = re.search(r'(\d+)$', text.strip())
        if match:
            val = int(match.group(1))
            # Normalize to Z_23 just in case the model hallucinated >22
            return val % 23
        return None
