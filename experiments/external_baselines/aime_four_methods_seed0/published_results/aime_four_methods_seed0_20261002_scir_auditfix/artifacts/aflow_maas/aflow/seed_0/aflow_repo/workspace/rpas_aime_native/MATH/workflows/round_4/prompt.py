ANALYSIS_PROMPT = """
You are a mathematical expert. Analyze the following problem step-by-step.
1. Identify the core mathematical concepts and variables.
2. Derive the necessary equations or inequalities.
3. Determine the constraints on the variables.
4. The final answer must be a single rational number or integer.
5. Do not solve the final numerical value yet; focus on setting up the logic.
Output your analysis in a structured format.
"""
FORMAT_VERIFY_PROMPT = """
You are a strict output formatter. Your task is to take the provided calculation result and ensure it is a valid number.
1. If the input is "None", "Error", or empty, output "None".
2. If the input is a valid number (integer or fraction), wrap it strictly in LaTeX format: \\boxed{value}.
3. Do not add any other text, explanations, or markdown outside the boxed command.
Input: {input}
Output:
"""