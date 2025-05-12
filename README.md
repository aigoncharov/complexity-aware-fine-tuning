# reasoning-fine-tune

## Prerequisites

- [uv](https://docs.astral.sh/uv/)

## Data

- Download CoT entropy data for MMLU ([Qwen 3B](https://disk.yandex.ru/d/A99rxeAx63CMsQ) [Qwen 3B with fallback if unknown](https://disk.yandex.ru/d/LowMkpNfbTcrXQ), [Phi4-mini](https://disk.yandex.ru/d/Z9NMNqJrDjchOg), [Phi4-mini with fallback if unknown](https://disk.yandex.ru/d/GAsFliSAiaAPAg)) to `data/out/cot_entropy`

## Running experiments

`uv run src/experiments/REPLACE_ME.py`