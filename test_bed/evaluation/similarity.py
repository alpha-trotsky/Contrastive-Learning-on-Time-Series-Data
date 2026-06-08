''' Computes the similarity between a predicted vector v_pred and original v from the same field.
After, X fields and vectors v from those fields are sampled. The mean similarity of the sampled "random" vectors v is compared to the original.
The higher the ratio of the original vector similarity to the sampled ones, the better the trained model. '''

import torch


def cosine_similarity(v_i, v_j):
    return (v_i @ v_j) / (torch.dot(v_i, v_i).sqrt() * torch.dot(v_j, v_j).sqrt())


@torch.no_grad()
def compute_similarities(model, modality, n_samples=25):
    device = next(model.parameters()).device
    u, v = modality.sample_pair(1)
    u_features = model.encode_u(u.float().to(device)).squeeze(0)   # [embed_dim]
    H = model.v_encoder.weight
    v_pred = u_features @ H                                         # [v_dim]
    v = v.squeeze(0).float().to(device)                             # [v_dim]

    self_sim = cosine_similarity(v_pred, v)

    _, v_random = modality.sample_pair(n_samples)
    v_random = v_random.float().to(device)
    random_sims = torch.stack([
        cosine_similarity(v_pred, v_random[i])
        for i in range(n_samples)
    ])
    
    return self_sim, random_sims.mean()