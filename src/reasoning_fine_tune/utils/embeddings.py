import torch


def get_embeddings(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    batch_hidden_states = outputs.last_hidden_state[0, :-1].squeeze()
    print("batch_hidden_states - pre", inputs.shape, batch_hidden_states.shape)
    pool_embeddings = {
        "min": batch_hidden_states.min(dim=0).values.cpu().numpy(),
        "max": batch_hidden_states.max(dim=0).values.cpu().numpy(),
        "mean": batch_hidden_states.mean(dim=0).values.cpu().numpy(),
    }
    print(
        "batch_hidden_states - post",
        pool_embeddings["min"].shape,
        pool_embeddings["max"].shape,
        pool_embeddings["mean"].shape,
    )
    return pool_embeddings
