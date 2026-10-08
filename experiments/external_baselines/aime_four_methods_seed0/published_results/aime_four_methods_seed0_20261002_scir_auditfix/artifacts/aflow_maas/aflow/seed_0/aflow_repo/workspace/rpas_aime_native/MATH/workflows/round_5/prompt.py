ANALYSIS_PROMPT = """
You are a mathematical expert. Analyze the following problem step-by-step.
1. Identify the core mathematical concepts and variables.
2. Derive the necessary equations or inequalities.
3. Determine the constraints on the variables.
4. The final answer must be a single rational number or integer.
5. Do not solve the final numerical value yet; focus on setting up the logic.
Output your analysis in a structured format.
"""
SANITIZE_PROMPT = """
You are a strict output sanitizer. Your task is to take the provided calculation result and ensure it is a valid, clean number wrapped in LaTeX.
1. If the input is "None", "Error", or empty, output "\\boxed{None}".
2. If the input is a valid number, remove all floating-point precision noise (e.g., convert 180.00000000000003 to 180).
3. If the input is a fraction, keep it as is.
4. Wrap the final clean value strictly in LaTeX format: \\boxed{value}.
5. Do not add any other text, explanations, or markdown outside the boxed command.
Input: {input}
Output:
"""
VERIFY_PROMPT = """
You are a strict logic verifier. You have an analysis of the problem and a sanitized candidate answer.
1. Check if the candidate answer is a valid number (integer or fraction).
2. If valid, output it strictly in LaTeX format: \\boxed{value}.
3. If invalid or "None", output "\\boxed{None}".
4. Do not add any other text, explanations, or markdown outside the boxed command.
Input: {input}
Output:
"""