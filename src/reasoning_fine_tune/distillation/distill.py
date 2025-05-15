import os

import pandas as pd
from tqdm import tqdm

from reasoning_fine_tune.prompts.mmlu_cot_answer import answer_marker, cot_answer_prompt, cot_sys_prompt
from reasoning_fine_tune.utils import openrouter
from reasoning_fine_tune.utils.validation import validate_mmlu_answer


def distill_on_dataset(
    in_filename,
    out_filename,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    dump_every=1000,
    max_tokens=1024,
    model="deepseek/deepseek-chat-v3-0324",
    get_sys_prompt=cot_sys_prompt,
    get_user_prompt=cot_answer_prompt,
):
    invalid_answers = 0

    if os.path.exists(out_filename):
        in_filename = out_filename

    df = pd.read_csv(
        in_filename,
        sep="\t",
        header=0,
    )

    field_response = "distill_response"
    field_ans = "distill_answer"
    field_ans_correct = "distill_ans_correct"

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_response not in df.columns:
        df[field_response] = ""
    if field_response not in df.columns:
        df[field_ans] = ""

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if df.at[index, field_response] != "":
            continue

        # print(f"loop {index} -> start: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        sys_prompt = get_sys_prompt(get_subject_from_row(row))
        user_prompt = get_user_prompt(get_question_from_row(row), get_options_from_row(row))
        # print(user_prompt)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = openrouter.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)

        response = completion.choices[0].message.content
        df.at[index, field_response] = response

        answer_marker_start = response.find(answer_marker[0])
        answer_marker_end = response.find(answer_marker[1])

        extracted_answer = ""
        if answer_marker_end != -1 and answer_marker_start != -1:
            extracted_answer = response[answer_marker_start:answer_marker_end]

        if validate_mmlu_answer(extracted_answer):
            df.at[index, field_ans] = extracted_answer
            df.at[index, field_ans_correct] = check_answer_correct(row, extracted_answer)
        else:
            invalid_answers += 1

        if index % dump_every == 0:
            df.to_csv(out_filename, sep="\t", index=False)

    df.to_csv(out_filename, sep="\t", index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
