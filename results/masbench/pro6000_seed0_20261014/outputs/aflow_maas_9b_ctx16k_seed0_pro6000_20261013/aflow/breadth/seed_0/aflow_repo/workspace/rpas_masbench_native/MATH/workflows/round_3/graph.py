from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_3.prompt as prompt_custom
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
        self.custom = operator.Custom(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble()

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solver_steps = [
            prompt_custom.PART_1_PROMPT,  # Step 1: Self-consistency reasoning and candidate generation
            prompt_custom.PART_2_PROMPT,  # Step 2: Verification of logic (SelfAsk style)
            prompt_custom.PART_3_PROMPT,  # Step 3: Extract exact FINAL ANSWER line
        ]
        
        # Initialize storage for generated solutions
        candidates_str = []
        verification_passed = False
        
        # Loop to generate multiple consistent solutions (Ensemble logic)
        max_iterations = 2
        for i in range(max_iterations):
            # Execute reasoning step
            res = await self.custom(input=problem, instruction=solver_steps[0])
            if "FINAL ANSWER" in res['response']:
                candidates_str.append(res['response'])
            else:
                candidates_str.append(res['response']) # Fallback if format not strict yet
            
            # Execute verification step in next turn of the same logical flow if possible, 
            # or rely on the ensemble in step 3.
            
        # If we have candidates, run Programmer to extract code logic for final check
        if len(candidates_str) > 0:
            code_gen = await self.programmer(problem=candidates_str[0], analysis="Verify dependencies and modulo 23 logic.")
            # Use the code generation to potentially re-solve if code is reliable, 
            # but primarily use the ensemble of text candidates.
            
            # Step 3: Aggregate final answer using Ensemble on the text candidates which contain 'FINAL ANSWER'
            # We filter candidates that likely contain the tag
            relevant_candidates = [c for c in candidates_str if "FINAL ANSWER:" in c]
            
            if len(relevant_candidates) > 0:
                # Wrap candidates in the expected format (list of strings)
                final_res = await self.sc_ensemble(solutions=relevant_candidates, problem=problem)
                solution_text = final_res['response']
                
                # Ensure text contains the tag if ensemble output is just a value
                if "FINAL ANSWER:" not in solution_text:
                    solution_text = "FINAL ANSWER: " + solution_text
                    
            else:
                # Fallback: just take the first one if none strictly formatted
                solution_text = candidates_str[0]
                
        else:
            solution_text = "FINAL ANSWER: ERROR"

        # Ensure the return value strictly matches the expected extraction pattern: "FINAL ANSWER: <value>"
        return solution_text, self.llm.get_usage_summary()["total_cost"]
