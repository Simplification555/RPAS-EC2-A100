# XXX_PROMPT = """
# You are an expert Python debugger and refiner.
# 
# Task: Fix the provided code based on a specific error message.
# 
# Problem Description:
# {problem}
# 
# Original Code:
# {original_code}
# 
# Specific Error Message:
# {error_message}
# 
# Constraints:
# 1. Analyze the error message carefully to identify the root cause.
# 2. Modify ONLY the necessary parts of the code to fix the error.
# 3. Ensure the function remains complete and runnable.
# 4. Do not include any input/output prompts or explanations outside the function code.
# 
# Output Format:
# Provide ONLY the corrected Python code for the function. Do not wrap it in markdown code blocks (```python ... ```).
# 
# Now, fix the code below:
# """