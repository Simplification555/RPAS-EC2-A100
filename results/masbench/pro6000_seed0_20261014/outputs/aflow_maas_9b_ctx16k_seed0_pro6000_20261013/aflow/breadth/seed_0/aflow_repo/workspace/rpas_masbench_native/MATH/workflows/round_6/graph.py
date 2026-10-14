from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_6.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.prog = operator.Programmer(self.llm)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Step 1: Derive symbolic dependencies and calculate result using Python
        code_result = await self.prog(input=problem, instruction="")
        answer_num = code_result['output'].strip() if 'output' in code_result else ""
        
        # Step 2: Use Custom to validate logic and format the final string
        # Input includes the calculated number to ensure context, instruction enforces strict formatting
        validation = await self.custom(input=problem, instruction=prompt_custom.VALIDATION_PROMPT + answer_num)
        
        # Step 3: Aggregate potential variations if re-evaluation or alternative logic paths exist
        final_answer = await self.sc_ensemble(solutions=[validation['response']], problem=problem)
        
        return final_answer['response'], (sum(x['cost'] if isinstance(x.dr, dict) else 0 for x in self.llm.cost_summary.values()))
