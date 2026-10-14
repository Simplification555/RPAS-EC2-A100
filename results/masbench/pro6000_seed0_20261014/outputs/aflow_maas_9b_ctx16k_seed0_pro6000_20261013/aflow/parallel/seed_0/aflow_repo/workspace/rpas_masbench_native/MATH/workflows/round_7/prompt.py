prompt_custom = """
Solve the provided math problem step by step.

CRITICAL CONSTRAINTS:
1. WORKSPACE: Calculations must be performed in the integer modulo 23 ring (Z_23). Reduce all additions, subtractions, multiplications, and intermediate results modulo 23 immediately. The range of valid answers is strictly 0 through 22.
2. DEFINITION RESOLUTION: If a variable required for the solution is not explicitly defined in the text, assume its value is 0. Do not hallucinate values for undefined entities.
3. FORMAT SUMMARY: Start your response immediately with "FINAL ANSWER: " followed by the exact integer answer for this specific problem. Do not include boxing characters, problem numbers, or explanations in the final line.
4. INDEPENDENCE: This prompt is called individually for each problem. Do not reference variables from other problems unless they are logically identical in name.

Problem Description:
%s
"""