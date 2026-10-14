XX_PROMPT = """
Task: Extract quantitative information.
Input: A logic problem text.
Output Format: A list of key-value pairs of the form "Variable: Value" found in the text, including numbers and defined variables.
Do not reason. Just output the list.
"""
XXXX_PROMPT = """
Task: Post-process and Format.
Input: A potential solution string from a code execution or reasoning step.
Constraint: The output MUST end with a line strictly formatted as "FINAL ANSWER: <integer_value>".
If the input contains a final number, append the footer "FINAL ANSWER: " + that number.
If no number is clear, extract the last numeric token and append the footer.
Return ONLY the formatted string.
"""