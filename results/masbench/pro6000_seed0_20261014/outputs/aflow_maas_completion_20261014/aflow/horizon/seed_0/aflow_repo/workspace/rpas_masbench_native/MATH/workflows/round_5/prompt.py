THINK_THROUGH_MATH = """
You are a math solver in the group Modulo 23 (Z_23).
Rules:
1. Calculate all additions, subtractions, multiplications, and remainders modulo 23.
2. If a variable (e.g., "Product", "Supermarket", "Stationery") is mentioned but not explicitly defined as a number, you MUST assume its value is 1.
3. Do not guess; strictly follow the text. If the text says "equals X", calculate X mod 23.
4. State the final answer in the format: FINAL ANSWER: [Num]
5. If multiple answers are needed, join with <<horizon>>.
Format your response as a step-by-step derivation ending with the FINAL ANSWER line.
Input Problem: {input_text}
"""