"""Recompute E_dir / scale on saved checkpoints (not part of the pipeline; scratch check)."""
import yaml
import torch

from signals.gaussian_process import GaussianProcess1D
from modalities.past_future import PastFutureModality
from models.linear_clip import LinearCLIP
from evaluation.theory_match import theory_match_error, theory_match_dir_error
from experiment import build_signal, build_modality, build_model, resolve_theory_target

RUNS = [
    ("conditional_dot [OLDER RUN, embed=20] (cosine)", r"..\results\phase1_gp_conditional_baseline",
     r"..\results\phase1_gp_conditional_baseline\model.pt"),
    ("one_sided_v_dot (cosine)", r"results\fill_one_sided_v_dot",
     r"results\fill_one_sided_v_dot\model.pt"),
    ("one_sided_v_l2 (L2, sanity check)", r"results\fill_one_sided_v_l2",
     r"results\fill_one_sided_v_l2\model.pt"),
    ("mse (sanity check)", r"results\fill_mse",
     r"results\fill_mse\model.pt"),
]

for label, cfg_dir, model_path in RUNS:
    cfg_path = f"{cfg_dir}\\config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    gen = build_signal(cfg["signal"])
    modality = build_modality(cfg["modality"], gen)
    model = build_model(cfg["model"], modality)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    target = resolve_theory_target(cfg["loss"])
    raw_err = theory_match_error(model, modality, target)
    dir_err, scale = theory_match_dir_error(model, modality, target)
    logit_scale = model.logit_scale.exp().item()

    raw_v = raw_err.item() if raw_err is not None else float("nan")
    dir_v = dir_err.item() if dir_err is not None else float("nan")
    scale_v = scale.item() if scale is not None else float("nan")

    print(f"{label:38s} steps={cfg['training']['num_steps']:6d} "
          f"raw_err={raw_v:.4f}  E_dir={dir_v:.4f}  scale={scale_v:.4f}  "
          f"logit_scale={logit_scale:.3f}")
