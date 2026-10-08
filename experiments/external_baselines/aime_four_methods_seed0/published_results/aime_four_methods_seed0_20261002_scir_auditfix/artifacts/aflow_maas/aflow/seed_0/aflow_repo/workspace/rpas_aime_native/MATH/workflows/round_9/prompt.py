ANALYSIS_PROMPT = """
You are a mathematical expert. Analyze the following problem step-by-step.
1. Identify the core mathematical concepts and variables.
2. Derive the necessary equations or inequalities.
3. Determine the constraints on the variables.
4. The final answer must be a single rational number or integer.
5. Do not solve the final numerical value yet; focus on setting up the logic.
Output your analysis in a structured format.
"""
REASONING_PROMPT = """
You are a reasoning engine. Based on the provided problem, analysis, and code output, derive the final numerical answer.
1. Synthesize the information from the analysis and code output.
2. Perform the final calculation if not already done.
3. State the final answer clearly in a standalone sentence at the very end of your response.
4. The sentence must follow the format: "The final answer is [number]." where [number] is the integer or fraction.
5. Do not include any other text after this sentence.
Input: {input}
Output:
"""
SANITIZE_PROMPT = """
You are a strict output sanitizer. Your task is to take the provided reasoning text and extract the final answer.
1. If the input does not contain a clear final answer sentence, output "\\boxed{None}".
2. If the input contains a sentence like "The final answer is X", extract X.
3. If the extracted value is a valid number, remove all floating-point precision noise (e.g., convert 180.00000000000003 to 180).
4. If the input is a fraction, keep it as is.
5. Wrap the final clean value strictly in LaTeX format: \\boxed{value}.
6. If the input is "\\boxed{None}" or contains no valid number, output "\\boxed{None}".
7. Do not add any other text, explanations, or markdown outside the boxed command.
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
FORMAT_ENFORCEMENT_PROMPT = """
You are a strict format enforcer. Your task is to ensure the input is a single LaTeX boxed value.
1. If the input is already in the format "\\boxed{value}" where value is a number or fraction, return it exactly as is.
2. If the input is "\\boxed{None}", return "\\boxed{None}".
3. If the input is a sentence like "The final answer is X", extract X and return "\\boxed{X}".
4. If the input contains any other text, explanations, or markdown, extract the number/fraction if present and return "\\boxed{number}". If no number is found, return "\\boxed{None}".
5. Do not add any other text, explanations, or markdown outside the boxed command.
Input: {input}
Output:
"""