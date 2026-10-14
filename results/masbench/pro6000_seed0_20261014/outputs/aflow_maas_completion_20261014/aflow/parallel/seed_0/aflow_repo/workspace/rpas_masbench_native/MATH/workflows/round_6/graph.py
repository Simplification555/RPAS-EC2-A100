from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_6.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem):
        """
        Solves a batch of subproblems. First parses, then verifies logic, 
        then iterates to extract and format the final sequence with high precision.
        """
        # Step 1: Instruct the model to output answers ONLY in the specific JSON-like format for regex extraction
        # We use a simpler template compared to the full detailed derivation to reduce hallucination in the extraction phase
        step1_response = await self.custom(
            input=problem,
            instruction=prompt_custom.BATCH_PARSING_PROMPT
        )
        parsed_text = step1_response['response']

        # Step 2: Extract individual answers using a stricter regex pattern
        import re
        # Updated pattern to handle potential newlines better but keep the structure tight
        pattern = r"Problem \d+ [Ww]as: (|\d+|\-[0-9]+)" 
        matches = re.findall(pattern, parsed_text, re.DOTALL)
        
        # Sanitize and convert to integers (handling empty strings or negative signs from negatives)
        cleaned_answers = []
        for m in matches:
            if m: 
                try:
                    val = int(m) % 23
                    if val < 0: val += 23 # Ensure 0-22 range for negative results
                    cleaned_answers.append(str(val))
                except ValueError:
                    cleaned_answers.append(0) # Default fallback 0 for parsing errors
            else:
                # Fallback to 0 if regex fails completely (though prompt prevents this)
                cleaned_answers.append(0)

        # Step 3: Use the Programmer operator to verify the arithmetic logic for the extracted answers
        # This reduces the chance of propagation errors by simulating a code check
        if cleaned_answers:
            verification_query = "Verify these calculations are correct in Z_23: " + " | ".join([f"{a}" for a in cleaned_answers])
            verification_result = await self.programmer(
                problem=verification_query,
                analysis="Output 'Correct' if all modular arithmetic is valid, or return specific errors."
            )
            
            # If verification indicates error, we re-run Step 1 with a slightly different instruction (retry logic)
            if "Error" in verification_result.get('output', '') or "incorrect" in verification_result.get('output', ''):
                # Minor retry with higher confidence instruction
                step1_response = await self.custom(
                    input=problem,
                    instruction=prompt_custom.BATCH_PARSING_PROMPT_RETRY
                )
                parsed_text = step1_response['response']
                matches = re.findall(pattern, parsed_text, re.DOTALL)
                cleaned_answers = []
                for m in matches:
                    if m: 
                        try:
                            val = int(m) % 23
                            if val < 0: val += 23
                            cleaned_answers.append(str(val))
                        except: cleaned_answers.append(0)
                    else: cleaned_answers.append(0)
            else:
                # Proceed with verified answers
                pass

        # Step 4: Aggregate answers with the required separator
        joined_answers = "<<horizon>>".join(cleaned_answers)
        final_line = f"FINAL ANSWER: {joined_answers}\n"

        return final_line, self.llm.get_usage_summary()["total_cost"]
