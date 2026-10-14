from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_4.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class MathSolver:
    def __init__(self, llm_config):
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem_text: str):
        # Step 1: Unify variables [answer1], etc. across all problems to a single variable `prev_result`
        # This helps the LLM track the chain of answers more clearly.
        unified_prompt = problem_text.replace("[answer1]", "参考答案_1").replace("[answer2]", "参考答案_2").replace("[answer3]", "参考答案_3").replace("[answer4]", "参考答案_4")
        unified_prompt = unified_prompt.replace("[answer5]", "参考答案_5").replace("[answer6]", "参考答案_6").replace("[variable]", "计算变量_")
        
        # Step 2: Construct a detailed instruction to force correct modulo arithmetic and strict formatting.
        # We guide the model to process problems sequentially and verify the separator.
        instruction = f"""
        You are a strict math solver for a benchmark set. Follow these rules EXACTLY:
        1. ARITHMETIC: All calculations are in Modulo 23 (Z_23). Results must be 0-22.
           - (a + b) mod 23
           - (a - b) mod 23 (if negative, add 23)
           - (a * b) mod 23
           - (a / b) mod 23 (calculate integer division then mod)
        2. CHAINING: [answer1] is the answer to Problem 1. [answer2] is the answer to Problem 2, derived using [answer1], etc. 
           If a problem says "Using result [answerX]", it refers to the numerical value calculated in Problem X.
        3. FORMAT: 
           - Output ALL intermediate steps in Chinese/English explaining the math.
           - For the final line, you MUST output exactly one line containing ONLY: FINAL ANSWER: <seq>
           - Where <seq> is the sequence of answers separated by `<<horizon>>`. Example: 5<<horizon>>18<<horizon>>7
           - Ensure you wrap this final line in \\boxed{ } if you are printing it as part of the solution block, but the LAST LINE of your text must be: FINAL ANSWER: ...
           - Do NOT output JSON. Do NOT output markdown code blocks for the final answer.
           - Check your separators carefully.
        """
        
        # Step 3: Execute the unified problem
        # Pass the unified prompt and the custom instruction which now expects the specific separator format.
        response = await self.custom(
            input=unified_prompt, 
            instruction=instruction
        )
        
        # Step 4: Validate and clean the output
        full_text = response['response']
        # Extract the final line if it exists
        lines = full_text.strip().split('\n')
        final_line = None
        for line in lines:
            if line.startswith("FINAL ANSWER:"):
                final_line = line
                break
        
        # If we couldn't find it perfectly, try to extract the boxed answer at the very end
        if not final_line:
            import re
            # Try to find the very last boxed value which should follow the format
            match = re.search(r'\\boxed{(\\d+<<horizon>>\\d+<<horizon>>\\d+)}', full_text)
            if match:
                final_line = f"FINAL ANSWER: {match.group(1)}"
            elif match and match.group(1):
                # Retry simpler single box match if anchoring fails, but prioritizing the explicit final stage format
                pass

        # Ensure we have a valid formatted string to return
        if final_line:
            return final_line, self.llm.get_usage_summary()["total_cost"]
        else:
            # Fallback: return the raw response costed, ensuring no None
            return full_text, 0.0

async def generate_math_solution(problem_str: str, llm_config: dict):
    solver = MathSolver(llm_config)
    solution_str, cost = await solver(problem_str)
    return {"solution": solution_str, "cost": cost}
