import gc
import os

import pandas as pd
import torch
from tqdm import tqdm

from reasoning_fine_tune.entropy_estimation.logit_entropy import compute_entropy_from_logits
from reasoning_fine_tune.prompts.mmlu_cot_answer import answer_marker, cot_answer_prompt, cot_sys_prompt
from reasoning_fine_tune.utils.device import DEVICE
from reasoning_fine_tune.utils.validation import validate_mmlu_answer


def estimate_dataset(
    in_filename,
    out_filename,
    model,
    tokenizer,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    dump_every=10,
    max_new_tokens=1024,
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

    model_name = model.config_class().model_type
    print(model_name)

    field_ans = f"entropy_ans_{model_name}"
    field_ans_correct = f"entropy_ans_correct_{model_name}"
    field_entropy_value = f"entropy_value_{model_name}"
    field_entropy_formatted_ans_token_index = f"entropy_formatted_ans_token_index_{model_name}"

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_entropy_value not in df.columns:
        df[field_entropy_value] = ""
    if field_entropy_formatted_ans_token_index not in df.columns:
        df[field_entropy_formatted_ans_token_index] = ""
    if field_ans not in df.columns:
        df[field_ans] = ""

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if validate_mmlu_answer(df.at[index, field_ans]):
            continue

        # print(f"loop {index} -> start: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        sys_prompt = get_sys_prompt(get_subject_from_row(row))
        user_prompt = get_user_prompt(get_question_from_row(row), get_options_from_row(row))
        # print(user_prompt)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=None,
            top_p=None,
            top_k=None,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        # print(f"loop {index} -> after generate: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        input_length = inputs.input_ids.shape[1]
        answer_raw = outputs.sequences[0, input_length:]
        answer = tokenizer.decode(answer_raw, skip_special_tokens=True)

        df.at[index, field_ans] = answer

        output_str: str = ""
        extracted_answer: str = ""
        answer_marker_start = -1
        answer_marker_end = -1
        output_entropy = []
        for i in range(len(outputs.scores)):
            # generated token position, batch_dim
            token_logits = outputs.scores[i][0]
            token_entropy = compute_entropy_from_logits(token_logits)
            output_entropy.append(token_entropy)

            token = token_logits.argmax(dim=-1)
            token_str = tokenizer.decode(token, skip_special_tokens=True)
            output_str += token_str

            if answer_marker_start == -1:
                if answer_marker[0] in output_str:
                    answer_marker_start = i

                    # Start accumulating extracted answer
                    extracted_answer += token_str
            elif answer_marker_end == -1:
                # Accumulate extracted answer until we see the "enf of answer" marker
                extracted_answer += token_str

                if answer_marker[1] in output_str:
                    answer_marker_end = i

                    # Extract option id by removing answer markers
                    extracted_answer = extracted_answer.split(answer_marker[0])[-1].split(answer_marker[1])[0]

        df.at[index, field_entropy_value] = ",".join(
            [f"{single_token_entropy:.4f}" for single_token_entropy in output_entropy]
        )

        if answer_marker_start != -1 and answer_marker_end != -1:
            df.at[index, field_entropy_formatted_ans_token_index] = f"{answer_marker_start},{answer_marker_end}"

        if validate_mmlu_answer(extracted_answer):
            # print(f"loop {index} -> after entropy: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")
            df.at[index, field_ans_correct] = check_answer_correct(row, extracted_answer)
        else:
            invalid_answers += 1

        print(
            f"CoT: {answer}\nExtracted answer: {extracted_answer}\nAnswer token indecies: {df.at[index, field_entropy_formatted_ans_token_index]}\nEntropy: {df.at[index, field_entropy_value]}\nis_correct: {df.at[index, field_ans_correct]}\ndims:{input_length}, {outputs.sequences.shape}\n\n"
        )

        if index % dump_every == 0:
            df.to_csv(out_filename, sep="\t", index=False)

        gc.collect()
        if DEVICE == torch.device("cuda"):
            torch.cuda.empty_cache()

    df.to_csv(out_filename, sep="\t", index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
