DERIVATION_PROMPT = """
Act as a rigorous math solver for the MASBench task.
Your task is to solve the provided problem step-by-step using Z_23 arithmetic.
1. Identify all given equations.
2. Trace dependencies from the target variable.
3. Assume any undefined base quantities are 0 if no chain exists to define them.
4. Perform all operations (add, sub, mul, diff) modulo 23.
Output ONLY the detailed derivation steps in Markdown, ending with a line "### Extracted Logic" where you explicitly state the logic used to determine the final value or the sequence of values. Do not output the final answer yet.
"""

FINAL_ANSWER_PROMPT = """
Based on the derivation steps provided for the problem, synthesize the final answers.
The problem may have multiple sub-problems.
1. Extract the calculated value for each sub-problem. If a step was missing due to undefined variables, the result is the number derived from the assumption of 0.
2. Format the output strictly as a single line: "FINAL ANSWER: {ans1}<<horizon>>{ans2}..." where {ans1} corresponds to the first question, {ans2} to the second, etc.
3. Ensure all numbers are integers between 0 and 22.
4. If there is only one question, the output should be "FINAL ANSWER: {ans1}".
Output ONLY the formatted string starting with "FINAL ANSWER:".
"""