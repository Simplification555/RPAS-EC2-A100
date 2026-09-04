# XXX_PROMPT = """
# You are an expert Python programmer.
# 
# Task: Solve the following coding problem.
# 
# Problem Description:
# {problem}
# 
# Constraints:
# 1. The solution must be a complete, runnable Python function definition.
# 2. Do not include any input/output prompts or explanations outside the function code.
# 3. Ensure the function handles edge cases (empty inputs, None, etc.) as implied by the problem.
# 4. Return the result directly.
# 
# Output Format:
# Provide ONLY the Python code for the function. Do not wrap it in markdown code blocks (```python ... ```).
# 
# Example Input:
# def is_nested(string):
#     '''
#     Create a function that takes a string as input which contains only square brackets.
#     The function should return True if and only if there is a valid subsequence of brackets 
#     where at least one bracket in the subsequence is nested.
#     '''
# 
# Example Output:
# def is_nested(string):
#     opening_bracket_index = []
#     closing_bracket_index = []
#     for i in range(len(string)):
#         if string[i] == '[':
#             opening_bracket_index.append(i)
#         else:
#             closing_bracket_index.append(i)
#     closing_bracket_index.reverse()
#     cnt = 0
#     i = 0
#     l = len(closing_bracket_index)
#     for idx in opening_bracket_index:
#         if i < l and idx < closing_bracket_index[i]:
#             cnt += 1
#             i += 1
#     return cnt >= 2
# 
# Now, solve the problem below:
# """