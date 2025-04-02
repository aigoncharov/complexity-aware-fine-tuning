import ast
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from reasoning_fine_tune.entropy_estimation.estimate_dataset import estimate_dataset
from reasoning_fine_tune.utils.device import DEVICE

print(f"Using device: {DEVICE}")

MODEL_NAME = "microsoft/phi-4"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map=DEVICE,
)


def verify_model_answer(row, model_answer):
    try:
        return int(row["answer_index"]) + 1 == int(model_answer)
    except:
        return False


estimate_dataset(
    in_filename=Path(__file__).joinpath("../../../data/source/mmlu_pro_stem.tsv").resolve(),
    out_filename=Path(__file__).joinpath("../../../data/out/mmlu_phi4_single_token.tsv").resolve(),
    model=model,
    tokenizer=tokenizer,
    get_subject_from_row=lambda row: row["base_cluster"],
    get_question_from_row=lambda row: row["question"],
    get_options_from_row=lambda row: ast.literal_eval(row["options"]),
    verify_answer=verify_model_answer,
)
