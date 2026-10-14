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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow with optimized logic for handling distractors and strict formatting
        """
        # Step 1: Robust parsing with explicit rule enforcement for conflicting values
        instruction = """
        You are an expert analyzer for the MASBench task. Follow these steps strictly:
        1. Identify the requested items from the text: [{items}].
        2. For each item, apply the 'Robustness-axis rule':
           - If the text contains multiple values for the same item (e.g., a definition followed by a 'Note: verify...'), the LAST value mentioned in a 'Note: verify...' statement is the CORRECT TRUE value.
           - Ignore the first unverified statement. Ignore calculated values (algebra) if a 'Note: verify...' exists for that specific item.
           - If no 'Note: verify...' exists, use the first explicit statement or the calculated value if necessary.
        3. Construct the final answer string. Join the verified values in the requested order using the separator '<<horizon>>'.
        4. Ensure the final output ends EXACTLY with one line formatted as: `FINAL ANSWER: [joined_values]`.
        Do not include \boxed{} or intermediate steps in the final line.
        """
        
        # Populate items and requested order from problem if dynamic, or pass raw problem for full context
        # Since placeholders are not supported, we inject the problem text directly but structure the instruction to prioritize the rule
        final_prompt = instruction.replace("[{items}]", "Analyze the provided text for all requested entities defined in the problem question.")
        
        response_data = await self.custom(input=problem, instruction=final_prompt)
        
        # The model is now instructed to output the FINAL ANSWER line directly
        if 'response' in response_data:
            content = response_data['response'].strip()
            # Extract the specific final line if the model adds extra text, though the prompt forbids it. 
            # Fallback to regex to ensure we get the 'FINAL ANSWER:' part even if formatted slightly differently.
            import re
            match = re.search(r'FINAL ANSWER:\s*(\S+)', content)
            if match:
                return match.group(1), self.llm.get_usage_summary()["total_cost"]
            else:
                return content, self.llm.get_usage_summary()["total_cost"]
        return "Error", 0.0
