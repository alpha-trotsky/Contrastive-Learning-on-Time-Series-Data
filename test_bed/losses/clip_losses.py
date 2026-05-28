# losses/clip_losses.py

import torch
import torch.nn as nn
from torch.nn import functional as F


class CLIPLoss(nn.Module):
    """Base class providing logit computation for CLIP-style losses.

    Parameters
    ----------
    inner_product : bool — if True use cosine (dot-product) logits;
                          if False use L²-distance tilting
    """

    def __init__(self, inner_product: bool = True):
        super().__init__()
        self.inner_product = inner_product

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        return torch.arange(num_logits, device=device, dtype=torch.long)

    def get_logits(self, image_features, text_features, logit_scale):
        if self.inner_product:
            logits_per_image = logit_scale * image_features @ text_features.T
        else:
            logits_per_image = (
                -0.5 * (image_features * image_features).sum(-1, keepdim=True)
                + image_features @ text_features.T
                - 0.5 * (text_features * text_features).sum(-1).unsqueeze(0)
            )
            logits_per_image = logits_per_image * logit_scale
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text


class CLIPConditionalLoss(CLIPLoss):
    """Symmetric (or one-sided) cross-entropy CLIP loss.

    Loss = lambda_u * CE(logits_per_u, labels)
         + lambda_v * CE(logits_per_v, labels)

    Parameters
    ----------
    lambda_u      : float — weight on the u-side cross-entropy
    lambda_v      : float — weight on the v-side cross-entropy
    inner_product : bool  — passed to CLIPLoss base
    """

    def __init__(
        self,
        lambda_u: float = 0.5,
        lambda_v: float = 0.5,
        inner_product: bool = True,
    ):
        super().__init__(inner_product=inner_product)
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v

    def forward(self, image_features, text_features, logit_scale) -> torch.Tensor:
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        return (
            self.lambda_u * F.cross_entropy(logits_per_image, labels)
            + self.lambda_v * F.cross_entropy(logits_per_text, labels)
        )


class CLIPJointLoss(CLIPLoss):
    """Joint (InfoNCE-style) loss using only the positive diagonal.

    Loss = -( mean(pos) - logsumexp(neg) - log(1/N) )

    Parameters
    ----------
    inner_product : bool — passed to CLIPLoss base
    """

    def __init__(self, inner_product: bool = True):
        super().__init__(inner_product=inner_product)

    def forward(self, image_features, text_features, logit_scale) -> torch.Tensor:
        logits_per_image, _ = self.get_logits(image_features, text_features, logit_scale)
        n = logits_per_image.shape[0]
        positive_mask = torch.eye(n, dtype=torch.bool, device=logits_per_image.device)
        negative_mask = ~positive_mask
        total_loss = (
            logits_per_image[positive_mask].mean()
            - torch.logsumexp(logits_per_image[negative_mask], dim=0)
            - torch.log(torch.tensor(1.0 / n, device=logits_per_image.device))
        )
        return -total_loss
