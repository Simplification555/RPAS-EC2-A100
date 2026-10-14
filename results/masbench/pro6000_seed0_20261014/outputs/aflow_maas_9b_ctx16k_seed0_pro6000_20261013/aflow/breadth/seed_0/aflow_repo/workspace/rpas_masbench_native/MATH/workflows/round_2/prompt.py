DEPENDENCY_ANALYSIS_PROMPT = """
Analyze the provided math problem which contains a graph of entities and their quantitative relationships defined by arithmetic rules in Z_23 (modulo 23).

Your task is to:
1. Identify the target variable requested in the question.
2. Trace the dependency path backwards from the target to the known base constants.
3. List the exact logical steps required to calculate the target value, ensuring no circular dependencies.
4. Explicitly state any base values found (e.g., "X equals 5"). If a value is missing, assume it is 0 or note it as a dependency to be resolved in the calculation phase.
5. Do not calculate the final number yet; just outline the algebraic steps.

Output your analysis in a structured plain text format.

Problem: {input}
"""

FORMAT_VERIFICATION_PROMPT = """
You are given an intermediate calculation result or a raw string output. Your task is to extract the single definitive answer or answer sequence for the math problem.

Rules:
1. If the text contains modulo arithmetic or complex numbers, simplify them mentally or mentally check if the format is valid.
2. Identify the "FINAL ANSWER" line or a clear statement of the result.
3. If multiple answers exist, join them with `<<horizon>>`.
4. Output ONLY the final answer value or sequence. Do not include "FINAL ANSWER:" prefix or any explanatory text.
5. If the input is a code string with the answer at the end, extract just that number.

Input: {input}
"""