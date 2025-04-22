from reasoning_fine_tune.prompts.mmlu_single_token_answer import option_ids


def validate_mmlu_answer(answer: str):
    # 0 is a special exception for "do not know"
    return answer in option_ids or answer == "0"
