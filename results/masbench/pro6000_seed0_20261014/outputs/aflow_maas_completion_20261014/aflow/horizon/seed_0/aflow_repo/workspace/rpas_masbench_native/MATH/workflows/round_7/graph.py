from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_7.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class MathSolver:
    def __init__(
        self,
        name: str,
        llm_config,
    ) -> None:
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.prog = operator.Programmer(self.llm)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        step1 = await self.prog(problem=problem, analysis="-analyze dependencies and solve step-by-step")
        summary = await self.custom(
            input=step1['output'],
            instruction="Summarize the final answers found in the text above."
        )
        final_format = await self.custom(
            input=f"{step1['code']}\n\nRaw Answers: {summary['response']}",
            instruction="Given the derivation and raw answers, generate the FINAL ANSWER line in format 'FINAL ANSWER: A<<horizon>>B...' if multiple, or just the number. Do not output anything else."
        )
        return f"{final_format['response']}", self.llm.get_usage_summary()["total_cost"]
