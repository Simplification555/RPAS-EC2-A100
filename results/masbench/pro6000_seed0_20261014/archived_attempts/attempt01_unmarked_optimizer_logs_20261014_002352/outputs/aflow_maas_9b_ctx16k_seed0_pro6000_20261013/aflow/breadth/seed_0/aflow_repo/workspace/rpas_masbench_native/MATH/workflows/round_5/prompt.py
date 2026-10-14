<prompt_custom>Igsm_Strict_Format_PROMPT = """
System Instruction: You are solving a complex arithmetic puzzle in Z_23 (modulo 23).
Critical Constraints:
1. Reduce all additions, subtractions, multiplications, and intermediate results modulo 23 (range 0-22).
2. If any specific value requested by the dependencies is not explicitly defined in the text, assign it a value of 0.
3. Do not output any markdown, explanations, or intermediate code blocks after the final line.
4. The final response must end EXACTLY with the string: FINAL ANSWER: [integer].
5. If the calculation results in 0, output 0.

Input Problem: {}
"""