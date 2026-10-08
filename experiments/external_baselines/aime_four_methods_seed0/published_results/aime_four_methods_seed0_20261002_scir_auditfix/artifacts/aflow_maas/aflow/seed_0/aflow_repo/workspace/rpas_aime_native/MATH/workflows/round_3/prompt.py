ANALYSIS_PROMPT = """
You are a mathematical expert. Analyze the following problem step-by-step.
1. Identify the core mathematical concepts and variables.
2. Derive the necessary equations or inequalities.
3. Determine the constraints on the variables.
4. Do not solve the final numerical value yet; focus on setting up the logic.
Output your analysis in a structured format.
"""
FORMAT_VERIFY_PROMPT = """
You are a strict output formatter for math problems.
Input: A code execution result and the original problem statement.
Task: 
1. Extract the final numerical answer from the provided code output.
2. If the answer is valid, wrap it strictly in LaTeX format: \boxed{answer}.
3. If the answer is invalid, None, or missing, output \boxed{None}.
4. Do not output any text, explanations, or markdown outside the boxed command.
5. Ensure the content inside the box is a clean number or simple expression.
"""