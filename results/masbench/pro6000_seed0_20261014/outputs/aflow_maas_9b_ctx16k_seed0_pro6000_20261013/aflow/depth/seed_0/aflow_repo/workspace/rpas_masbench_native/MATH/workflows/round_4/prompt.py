PROMPT_EXTRACT_ANSWER = """
Your task is to read the provided text (which contains a math problem) and extract the explicit 'FINAL ANSWER' if it exists in the text. If no 'FINAL ANSWER' exists, extract the most plausible numeric answer mentioned or derived in the context if obvious. 
Output format: ONLY the final digit or number found. Do not include 'FINAL ANSWER:' text or any explanations.
Text: {INPUT}
"""

PROMPT_REASON_VERIFY = """
You are a rigorous math solver. Your goal is to solve the problem below and output the FINAL ANSWER in the exact format: `FINAL ANSWER: <number>`.
You must perform all calculations modulo 23 as implied by the problem context (`Z_23` arithmetic).
Input: {INPUT}
If your derived number is X, output exactly: FINAL ANSWER: X
Do not output any intermediate steps or explanations in the final output string provided in the 'response' key.
"""