from typing import List


def cot_sys_prompt(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " Explain your thinking process step-by-step. At the end, write down the number of the correct answer by strictly following this format: [[number_of_correct_answer]]."
    return sys_msg


def cot_sys_prompt_with_fallback_for_unknown_answers(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " Explain your thinking process step-by-step. At the end, if you are certain about the answer write down the number of the correct answer by strictly following this format: [[number_of_correct_answer]], otherwise return [[0]]."
    return sys_msg


def cot_sys_prompt_with_fallback_for_unknown_answers_alternative(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " Explain your thinking process step-by-step. At the end, if you know the answer write down the number of the correct answer by strictly following this format: [[number_of_correct_answer]], otherwise return [[0]]."
    return sys_msg


option_ids = [str(i + 1) for i in range(20)]


def cot_answer_prompt(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
    return user_prompt
