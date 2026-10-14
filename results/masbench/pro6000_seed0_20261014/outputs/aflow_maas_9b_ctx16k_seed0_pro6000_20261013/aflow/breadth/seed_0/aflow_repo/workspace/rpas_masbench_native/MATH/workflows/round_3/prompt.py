PART_1_PROMPT = """
Solve the following math problem. You are generating a solution trace. 
Follow the instructions strictly: Reduce all arithmetic modulo 23. Answer only the final number requested.
You MUST end your response with exactly one line: FINAL ANSWER: [NUMBER]
Format your output as: <intermediate_evidence>... thinking steps ...</intermediate_evidence>FINAL ANSWER: [NUMBER]

Problem: {problem}
"""

PART_2_PROMPT = """
Critique the previous solution attempt. Check for circular dependencies and modulo arithmetic errors.
Propose a verified calculation path.
Output: A revised <intermediate_evidence>...</intermediate_evidence> block ending with FINAL ANSWER: [NUMBER].
"""

PART_3_PROMPT = """
Extract the final numerical answer from the provided solution traces. 
Ignore all text except the specific line starting with "FINAL ANSWER: ".
Return ONLY the line "FINAL ANSWER: [NUMBER]".
Do not include <intermediate_evidence> blocks.
"""