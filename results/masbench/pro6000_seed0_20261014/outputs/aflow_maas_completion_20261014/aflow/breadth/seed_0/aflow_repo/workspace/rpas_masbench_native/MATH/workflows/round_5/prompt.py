prompt_custom = """

Analyze the mathematical problem provided below. The problem involves arithmetic in the ring Z_23 (integers modulo 23).

Instructions:
1. Translate the problem statement into a Python script.
2. In your Python script, perform all additions, subtractions, multiplications, and intermediate results using modulo 23 arithmetic (e.g., `result = (a + b) % 23` or `result = a % 23`).
3. Handle negative results in modulo arithmetic correctly (e.g., `-2 % 23` should be `21`).
4. Define any variables not explicitly given a number as 0 unless the problem context implies otherwise.
5. The script must strictly output only the final integer answer or the specific formatted string required (e.g., "FINAL ANSWER: X"). Do not include variable names, code blocks, or explanations in the output.
6. Read the question carefully to determine if the final output needs to be in a specific format (e.g., "FINAL ANSWER: ANSWER"). If so, ensure the Python print statement matches this format exactly, but the value extracted must be the answer itself.

Problem Statement:
{problem}

Output the resulting string from the Python execution.
"""