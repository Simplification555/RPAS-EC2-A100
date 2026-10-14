from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_4.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem):
        """
        Solves a batch of subproblems. First solves them in parallel (waiting), 
        then iterates to extract and format the final sequence.
        """
        # Step 1: Wait for the model to generate independent solutions for all subproblems
        # The instruction in prompt_custom handles the parsing of all problems to a temporary JSON or marked format
        raw_response = await self.custom(
            input=problem,
            instruction=prompt_custom.MATH_BATCH_SOLUTION_PROMPT
        )
        response_text = raw_response['response']

        # Step 2: Extract individual answers using regex to find the boxed patterns in order
        import re
        pattern = r"Problem \d+:\s*\\boxed{([^}]+)}"
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if not matches:
            # Fallback if regex fails, though the prompt is designed to prevent this
            return "FINAL ANSWER: ERROR", self.llm.get_usage_summary()["total_cost"]

        # Clean up answers (remove spaces/newlines)
        cleaned_answers = [m.strip() for m in matches]
        
        # Step 3: Aggregate answers with the required separator and construct the final line
        joined_answers = "<<horizon>>".join(cleaned_answers)
        final_line = f"FINAL ANSWER: {joined_answers}\n"

        return final_line, self.llm.get_usage_summary()["total_cost"]
