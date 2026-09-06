# REASONING_PROMPT = """
# Analyze the input question carefully. Provide a step-by-step thought process to derive the answer. 
# After your reasoning, output the final answer strictly in this JSON format: {"answer": "YOUR_FINAL_ANSWER"}
# Do not include any other text outside the JSON block for the final answer line.
# """

# EXTRACT_PROMPT = """
# You are given a text containing reasoning and a final answer in JSON format.
# Your task is to extract the content of the 'answer' key from the JSON object.
# Return ONLY the string value of the 'answer' key.
# Input text: {input}
# Output the extracted answer string directly.
# """

EXTRACT_PROMPT = "Use only the supplied context to answer the question. Reason briefly, then provide the concise final answer.\n\nInput: {input}"

REASONING_PROMPT = "Use only the supplied context to answer the question. Reason briefly, then provide the concise final answer.\n\nInput: {input}"
