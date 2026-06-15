"""Vendorized into Test3 from Judge/Ori for local builder stability.

Behavior is kept source-compatible with the Ori implementation.
"""

from typing import Union, Dict, Any, List
import itertools
import os
from KVCOMM.prompt.prompt_set import PromptSet
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.prompt.common import get_combine_materials


def _console_debug_enabled() -> bool:
    value = os.environ.get("KVCOMM_PROGRESS_ONLY", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _paper_debug_import(message: str) -> None:
    path_value = os.environ.get("PAPER_REPAIR_DEBUG_LOG", "").strip()
    if not path_value:
        return
    line = f"[IMPORT CHAIN] {message}"
    if _console_debug_enabled():
        print(line, flush=True)
    try:
        with open(path_value, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


_paper_debug_import("imported mmlu_prompt_set")

roles = itertools.cycle(['Knowledgeable Expert',
                         'Wiki Searcher',
                         'Critic',
                         'Mathematician',
                         'Psychologist',
                         'Historian',
                         'Doctor',
                         'Lawyer',
                         'Economist',
                         'Programmer'])


ROLE_DESCRIPTION = {
"Knowledgeable Expert":
"""
You are a knowledgeable expert in question answering.
Please give at most six key entities that need to be searched in wikipedia to solve the problem. 
Key entities that need to be searched are included between two '@' when output, for example: @catfish effect@, @broken window effect@, @Shakespeare@.
If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
""",
"Wiki Searcher":
"""
You will be given a question and a wikipedia overview of the key entities within it.
Please refer to them step by step to give your answer.
And point out potential issues in other agent's analysis.
""",
"Critic":
"""
You are an excellent critic.
Please point out potential issues in other agent's analysis point by point.
""",
"Mathematician":
"""
You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
""",
"Psychologist":
"""
You are a psychologist.
You are good at psychology, sociology, and philosophy.
You give people scientific suggestions that will make them feel better.
""",
"Historian":
"""
You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
""",
"Doctor":
"""
You are a doctor and come up with creative treatments for illnesses or diseases.
You are able to recommend conventional medicines, herbal remedies and other natural alternatives. 
You also consider the patient's age, lifestyle and medical history when providing your recommendations.
""",
"Lawyer":
"""
You are good at law, politics, and history.
""",
"Economist":
"""
You are good at economics, finance, and business.
You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.
""",
"Programmer":
"""
You are good at computer science, engineering, and physics.
You have experience in designing and developing computer software and hardware.
""",
"MMLU Solver":
"""
Solve the MMLU multiple-choice question (A/B/C/D; only one is correct).

Output:
1) First line: ONLY one letter (A, B, C, or D).
2) Then a short justification (total < 100 words).

You may use other agents' answers as references if provided, but verify independently.
""",
"Fake":
"""
You are a liar who only tell lies.
""",
}


@PromptSetRegistry.register('mmlu')
class MMLUPromptSet(PromptSet):
    """Prompt set for 4-option multiple-choice QA (MMLU-style)."""
    @staticmethod
    def get_role():
        """Return the next cyclic role for multi-agent setups."""
        return next(roles)

    @staticmethod
    def get_decision_role():
        """Return the role used for the decision-maker agent."""
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."

    @staticmethod
    def get_constraint():
        """Return base task constraints for multiple-choice QA."""
        return """
I will ask you a question.
I will also give you 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
You must choose the correct answer to the question.
Your response must be one of the 4 letters: A, B, C or D,
corresponding to the correct answer.
Your answer can refer to the answers of other agents provided to you.
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C or D)
        """

    @staticmethod
    def get_analyze_constraint(role):
        """Return role-specific analysis constraints."""
        return ROLE_DESCRIPTION[role] if role in ROLE_DESCRIPTION.keys() else ""+ """
I will ask you a question and 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C or D)
"""

    @staticmethod
    def get_decision_constraint():
        """Return decision constraints for picking a single option."""
        return """
You will receive a question followed by four possible answers labeled A, B, C, and D. Only one answer is correct. 
Choose the correct option based on the analysis and recommendations provided by the output of other agents. 
Your response must be exactly one of the letters A, B, C, or D, with no additional characters or text.
        """

    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        """Return the raw question string."""
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        """Return an adversarial prompt to elicit intentionally wrong answers."""
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : A, B, C or D)
                """

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_few_shot():
        return ""

    @staticmethod
    def get_select_best_role() -> str:
        return (
            "You are a judge that selects which agent produced the best multiple-choice answer.\n"
            "You will be given a question and candidate answers from different agents.\n"
            "Your job is to choose the best agent and output its id and final choice."
        )

    @staticmethod
    def get_select_best_constraint(agent_ids: list[str]) -> str:
        allowed = ", ".join(agent_ids)
        return (
            "Selection criteria:\n"
            "- Prefer the agent with the most correct, reliable reasoning.\n"
            "- If multiple agents choose the same option, prefer the one with better justification.\n\n"
            "Output format requirements (MUST follow exactly):\n"
            f"1) First line: Selected agent id: <id> (id must be one of: {allowed})\n"
            "2) Second line: <choice> (must be exactly one of: A, B, C, D)\n"
            "3) Do NOT output anything else.\n"
        )

    @staticmethod
    def get_select_best_agent_prompt(
        question: str, candidates: list[tuple[str, str]], constraint: str
    ) -> str:
        formatted = "\n\n".join(
            f"Agent {agent_id}:\n{answer}" for agent_id, answer in candidates
        )
        return (
            "# Best Agent Selection Task (MMLU)\n\n"
            f"## Question:\n{question}\n\n"
            "## Candidate Agent Answers:\n"
            f"{formatted}\n\n"
            "## Instructions:\n"
            f"{constraint}"
        )

    async def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        """Normalize answer into a single string and strip prefix markers."""
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        answer = answer.strip()
        return answer
