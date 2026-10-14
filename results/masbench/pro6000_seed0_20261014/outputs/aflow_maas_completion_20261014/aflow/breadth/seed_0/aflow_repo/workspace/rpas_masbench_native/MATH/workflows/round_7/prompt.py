XXX_PROMPT = """
Based on the provided problem statement and the PRECALCULATED numerical result provided below:

PRECALCULATED VALUE: {{PRECALC_VALUE}}

Perform a final logical consistency check to ensure this number maps to the correct variable and relationship defined in the problem (e.g., verify if 'Bone' logically equates to 'Distal Radials' or 'Caudal Fin Rays' based on the text). 

If the PRECALCULATED VALUE aligns with the text dependencies:
- Confirm the answer.
- Output the string: FINAL ANSWER: {{PRECALC_VALUE}}

If there is a conflict or if the problem implies a specific dependency that overrides the calculation:
- Briefly explain the override in the reasoning text.
- Output the string: FINAL ANSWER: {{CORRECT_VALUE}}

CRITICAL CONSTRAINTS:
1. All arithmetic must adhere to Z_23 (modulo 23) as stated in the problem preamble.
2. DO NOT output the code or intermediate steps in the FINAL ANSWER block.
3. The final block MUST be exactly on a separate line: FINAL ANSWER: <integer>
4. Do not include any other text outside this block if the answer is confirmed.
"""