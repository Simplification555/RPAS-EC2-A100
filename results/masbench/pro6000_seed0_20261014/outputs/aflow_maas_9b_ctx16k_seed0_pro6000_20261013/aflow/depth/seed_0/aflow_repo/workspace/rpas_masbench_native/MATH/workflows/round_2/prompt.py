SOLVE_PROMPT = """

Solve the math problem step-by-step.

Problem: {input}

Rules:
1. Follow the specific arithmetic rules provided in the problem (e.g., modulo 23).
2. Trace all dependencies correctly.
3. If a value is not explicitly defined or cannot be derived, assume it is 0.
4. Perform all calculations in Z_23 (modulo 23).

Step 1: Think step-by-step through the logic and calculations.
Step 2: At the very end, output your final answer as a separate line starting with "your solution: " followed by ONLY the integer or sequence (e.g., "your solution: 42" or "your solution: 1<<horizon>>2"). Do not include any trailing text after this line.

"""

FORMAT_PROMPT = """

Task: Format the solution string for final output.

Input Problem: {input}
Calculated Solution Value: {solution}

Instructions:
1. Ignore all intermediate reasoning text.
2. Create a string that ends EXACTLY with one separate line: "Final Answer: " followed by the calculated solution value.
3. Ensure there is no extra whitespace or punctuation after the value.

Output Format Example: "Final Answer: 42"