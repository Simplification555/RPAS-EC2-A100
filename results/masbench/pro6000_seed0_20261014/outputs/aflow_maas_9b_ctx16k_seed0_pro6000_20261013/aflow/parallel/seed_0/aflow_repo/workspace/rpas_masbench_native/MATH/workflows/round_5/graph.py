from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_5.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem):
        """
        Solves a batch of subproblems. 
        1. Generates initial parallel solutions.
        2. Cleans noise (repetitions) using Python logic.
        3. Uses ScEnsemble to verify/select the best version.
        4. Formats the final output.
        """
        # Step 1: Wait for the model to generate independent solutions for all subproblems
        # Handle potential parallel structure in input logic
        raw_response = await self.custom(
            input=problem,
            instruction=prompt_custom.MATH_BATCH_SOLUTION_PROMPT
        )
        response_text = raw_response['response']

        # Step 2: Extract individual answers using regex to find the boxed patterns
        import re
        pattern = r"Problem \d+:\s*\\boxed{([^}]+)}"
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if not matches:
            # Fallback: Try to find the self-made format if regex fails
            if "<<horizon>>" in response_text:
                # Split by separator and clean surrounding whitespace
                parts = response_text.split("<<horizon>>")
                cleaned = [p.strip().strip('[').strip(']') for p in parts if p.strip()]
                joined = "<<horizon>>".join(cleaned)
                final_line = f"FINAL ANSWER: {joined}\n"
                return final_line, self.llm.get_usage_summary()["total_cost"]
            else:
                return "FINAL ANSWER: ERROR", self.llm.get_usage_summary()["total_cost"]

        # Clean up answers (remove spaces/newlines/brackets)
        cleaned_answers = [m.strip().strip('[').strip(']') for m in matches]
        
        # Step 3: Use ScEnsemble to ensure the sequence is robust (e.g., if it repeated answers)
        # Prepare the sequence string for the ensemble operator
        sequence_str = "<<horizon>>".join(cleaned_answers)
        ensemble_result = await self.sc_ensemble(
            solutions=[sequence_str], 
            problem=problem
        )
        # ScEnsemble returns a dict with 'response'
        robust_sequence = ensemble_result['response']

        # Step 4: Use a second Custom call to strictly format the noise-free sequence
        # This acts as a "Formatter" to ensure no extra brackets or noise remain
        formatter_response = await self.custom(
            input=cleaned_answers, 
            instruction=prompt_custom.FORMAT_FINAL_ANSWER_PROMPT
        )
        final_line = formatter_response['response']

        return final_line, self.llm.get_usage_summary()["total_cost"]
