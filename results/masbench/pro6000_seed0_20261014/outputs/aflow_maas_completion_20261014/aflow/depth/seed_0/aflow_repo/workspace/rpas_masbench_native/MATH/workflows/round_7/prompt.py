VERIFICATION_PROMPT = """
Analyze the provided math word problem strictly using the rules:
1. Every quantity is in Z_23 (integers 0-22).
2. If a variable is mentioned in dependencies but never explicitly defined or derived by a formula, strictly set its value to 0.
3. Do not hallucinate values. Only use defined numbers and calculated results.

Solve the problem and provide the solution in exactly 4 distinct lines, one for each distinct result, following this exact format for LINE 1:
"Calculation: [Step-by-step logic]. Result: [Number]". Repeat this 4 times with different logical paths or verification points if possible, but ensure the final derived answer is consistent across all lines.
The 4 lines must be separated by a newline character.
End the entire output with the exact string: "FINAL ANSWER: [Final Number]".
"""