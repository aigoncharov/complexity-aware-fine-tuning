def pool_embeddings(embeddings):
    pooled_embeddings = {
        "min": embeddings.min(dim=0).values.cpu().numpy(),
        "max": embeddings.max(dim=0).values.cpu().numpy(),
        "mean": embeddings.mean(dim=0).values.cpu().numpy(),
    }
    print(
        "batch_hidden_states - post",
        pooled_embeddings["min"].shape,
        pooled_embeddings["max"].shape,
        pooled_embeddings["mean"].shape,
    )
    return pooled_embeddings
