import gc

import pandas as pd
import torch
from tqdm import tqdm

import reasoning_fine_tune.prompts.mmlu as mmlu_prompts
from reasoning_fine_tune.entropy_estimation.logit_entropy import TokenwiseEntropy
from reasoning_fine_tune.utils.device import DEVICE


def estimate_dataset(
    in_filename,
    out_filename,
    model,
    tokenizer,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    verify_answer,
    dump_every=100,
):
    invalid_answers = 0

    df = pd.read_csv(
        in_filename,
        sep="\t",
        header=0,
    )

    model_name = model.config_class().model_type
    print(model_name)

    field_ans = f"entropy_ans_{model_name}"
    field_ans_correct = f"entropy_ans_correct_{model_name}"
    field_entropy_value = f"entropy_value_{model_name}"

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_entropy_value not in df.columns:
        df[field_entropy_value] = 0.0
    if field_ans not in df.columns:
        df[field_entropy_value] = ""

    entropy_estimator = TokenwiseEntropy(llm_model=model)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if df.at[index, field_entropy_value] != 0.0:
            continue

        # print(f"loop {index} -> start: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        sys_prompt = mmlu_prompts.get_sys_prompt(get_subject_from_row(row))
        user_prompt = mmlu_prompts.get_user_prompt(get_question_from_row(row), get_options_from_row(row))
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

        outputs = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        # print(f"loop {index} -> after generate: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        input_length = inputs.input_ids.shape[1]
        answer_raw = outputs[0, input_length:]
        answer = tokenizer.decode(answer_raw, skip_special_tokens=True)
        if answer in mmlu_prompts.option_ids:
            entropy = entropy_estimator.calculate(outputs)
            # print(f"loop {index} -> after entropy: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")
            df.at[index, field_entropy_value] = entropy
            df.at[index, field_ans] = answer
            df.at[index, field_ans_correct] = verify_answer(row, answer)
        else:
            invalid_answers += 1

        # print(f"Answer: {answer}\nEntropy: {entropy}\nis_correct: {df.at[index, field_ans_correct]}\n\n")

        if index % dump_every == 0:
            df.to_csv(out_filename, sep="\t", index=False)

        gc.collect()
        if DEVICE == torch.device("cuda"):
            torch.cuda.empty_cache()

    df.to_csv(out_filename, sep="\t", index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
