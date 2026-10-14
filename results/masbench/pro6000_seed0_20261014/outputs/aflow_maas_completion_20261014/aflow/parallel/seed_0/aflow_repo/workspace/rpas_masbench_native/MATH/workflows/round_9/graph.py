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

async def __call__(self, problem: str):
"""
Implementation of the workflow
"""
# Step 1: Prepare the direct input string with embedded instructions
direct_problem = f"PROBLEMS:{problem}"

# Step 2: Solve sequentially for each problem to ensure independence and strict Z_23 handling
problem_results = []
for idx, sub_problem in enumerate(problem.split("Problem "))[1:]:
    if idx == 0:
        continue
    # Clean up the sub-problem text (remove leading newlines/indices if any)
    sub_task = sub_problem.strip().split("Note:")[0].strip()
    # Ensure we have text
    if not sub_task: continue
    
    # Construct a highly specific, placeholder-free prompt that enforces Z_23, 0-default, and regex-friendly output
    task_prompt = f"""Solve this specific math problem independently. All arithmetic is in Z_23 (modulo 23, results 0-22). 
        Undefined variables equal 0. If a 'Plural Category' has no listed members, its sum is 0.
        Problem: {sub_task}
        
        Instructions:
        1. Trace dependencies strictly.
        2. Perform all additions/subtractions/multiplications modulo 23.
        3. Output ONLY the final integer answer for this single problem using EXACTLY the format: [[ANSWER]]
        Example: [[20]]
        No other text, explanations, or markdown allowed.
        """
    
    # Solve the sub-problem
    response_text = await self.custom(input=f"{task_prompt}", instruction="")
    
    # Extract the answer using a robust regex look-alike (simulating the extraction logic in the graph)
    # We assume the custom method returns the raw response. We extract [[X]]
    import re
    match = re.search(r'\[\[(\d+)\]\]', response_text.get('response', ''))
    if match:
        ans_val = int(match.group(1))
        # Apply Z_23 reduction just in case the logic slipped, though prompt prevents it
        ans_val = ans_val % 23
        problem_results.append(str(ans_val))

# Step 3: Aggregate results with the required separator
final_answer_str = "<<horizon>>".join(problem_results)
if not final_answer_str:
    final_answer_str = "0"

# Step 4: Verify and Cast
# Construct the final standardized string
full_output = f"FINAL ANSWER: {final_answer_str}"

# Construct the specific final string required by the format
# The user requires: FINAL ANSWER: ans1<<horizon>>ans2...
# We have `final_answer_str` as "ans1<<horizon>>ans2..."
# We return the full line as the response for the main prompt's extraction logic
return {"response": full_output}, self.llm.get_usage_summary()["total_cost"]
