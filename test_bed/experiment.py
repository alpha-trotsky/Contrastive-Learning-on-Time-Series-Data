"""Run a single contrastive-learning experiment per the config.

Usage:
    python experiment.py --config config.yaml
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from modalities.past_future import PastFutureModality
from signals.gaussian_process import GaussianProcess1D
from modalities.field_coeff import FieldCoeffModality
from models.linear_clip import LinearCLIP
from losses.clip_losses import CLIPConditionalLoss, CLIPJointLoss, MSEloss
from evaluation.theory_match import theory_match_error, theory_match_dir_error
from evaluation.forecast import forecast_mse, residual_cov_error, model_cov_error, debug_Ahat
from evaluation.similarity import compute_similarities
from evaluation.retrieval import evaluate_encoders
import wandb


# ---------------------------------------------------------------------------
# Factories — translate config dicts into objects
# ---------------------------------------------------------------------------

def build_signal(cfg):
    if cfg['type'] == 'gaussian_process_1d':
        return GaussianProcess1D(
            alpha=cfg['alpha'],
            tau=cfg['tau'],
            dim_true=cfg['dim_true'],
            zero_mean=cfg['zero_mean'],
        )
    #sinusoidal generator once thats built to be tested
    raise ValueError(f"Unknown signal type: {cfg['type']}")


def build_modality(cfg, gen):
    if cfg['type'] == 'field_coeff':
        return FieldCoeffModality(
            gen=gen,
            u_index=cfg['u_index'],
            dim_coeff=cfg['dim_coeff'],
            sigma=cfg['sigma'],
        )
    if cfg['type'] == 'PastFuture':
        start = cfg.get('start', 0)
        past_index = list(range(start, start + cfg['past_len'])) # init of the windows for the past future prediction - if that even makes sense?
        future_index = list(range(start + cfg['past_len'], start + cfg['past_len'] + cfg['future_len'])) 
        return PastFutureModality(
            gen=gen,
            past_index=past_index,
            future_index = future_index,
            sigma_u = cfg.get('sigma_u', 0.0),
            sigma_v = cfg.get('sigma_v', 0.0),)
    raise ValueError(f"Unknown modality type: {cfg['type']}")


def build_model(cfg, modality):
    if cfg['type'] == 'linear_clip':
        return LinearCLIP(
            embed_dim=cfg['embed_dim'],
            u_dimension=modality.dim_u,
            v_dimension=modality.dim_v,
            bias=cfg['bias'],
            init_logit_scale=cfg['init_logit_scale'],
        )
    raise ValueError(f"Unknown model type: {cfg['type']}")


def resolve_mean_family(loss_cfg):
    """Which model conditional-mean formula applies, given the loss/tilting.

    The forecast prediction head is set by the *tilting* (the model form ν),
    not by conditional-vs-joint:
      - cosine tilting  (inner_product=True)  -> eq (41): E[v|u] = C_vv A^T u
      - L2 tilting      (inner_product=False) -> eq (48): E[v|u] = (C+C_vv^-1)^-1 A^T u
      - mse baseline    -> plain regression head  E[v|u] = (W_u^T W_v)^T u
    """
    if loss_cfg['type'] == 'mse':
        return 'mse'
    return 'cosine' if loss_cfg.get('inner_product', True) else 'quadratic'


def resolve_theory_target(loss_cfg):
    """Which analytical A* `theory_match_error` should compare against.

    Mirrors `resolve_mean_family` but at the level of the closed-form optimum:
      - cosine conditional / one-sided -> Thm 5.1   ('cosine_conditional')
      - cosine joint                   -> Thm 5.6   ('cosine_joint')
      - L2 one-sided v / u             -> Thm 5.3   ('quadratic_v' / 'quadratic_u')
      - mse                            -> K          ('mse')
      - two-sided L2 / L2 joint        -> no closed form ('none')
    """
    t = loss_cfg['type']
    if t == 'mse':
        return 'mse'
    inner_product = loss_cfg.get('inner_product', True)
    if t == 'joint':
        return 'cosine_joint' if inner_product else 'none'   # L2 joint: no formula
    if inner_product:
        return 'cosine_conditional'                          # cosine: any lambda -> Thm 5.1
    # L2 (quadratic) tilting: closed form only for the one-sided cases.
    # NB: here lambda_u weights the p(v|u) cross-entropy (CE on logits_per_image),
    # which constrains C and matches the v|u conditional; lambda_v weights p(u|v),
    # constrains B and matches u|v.  A *pure* one-sided match needs the OTHER
    # weight to be 0.  (This is the opposite of the paper's lambda subscripts.)
    lambda_u = loss_cfg.get('lambda_u', 0.5)
    lambda_v = loss_cfg.get('lambda_v', 0.5)
    if t == 'one_sided_u' or lambda_v == 0.0:                 # only p(v|u): matches v|u
        return 'quadratic_v'
    if t == 'one_sided_v' or lambda_u == 0.0:                 # only p(u|v): matches u|v
        return 'quadratic_u'
    return 'none'                                             # two-sided L2: no formula


def build_loss(cfg, model=None):
    if cfg['type'] == 'conditional':
        return CLIPConditionalLoss(
            lambda_u=cfg['lambda_u'],
            lambda_v=cfg['lambda_v'],
            inner_product=cfg['inner_product'],
        )
    elif cfg['type'] == 'joint':
        return CLIPJointLoss(inner_product=cfg['inner_product'])
    elif cfg['type'] == 'one_sided_u':
        return CLIPConditionalLoss(lambda_u=1.0, lambda_v=0.0, inner_product=cfg['inner_product'])
    elif cfg['type'] == 'one_sided_v':
        return CLIPConditionalLoss(lambda_u=0.0, lambda_v=1.0, inner_product=cfg['inner_product'])
    elif cfg['type'] == 'mse':
        return MSEloss(v_encoder=model.v_encoder)
    else:
        raise ValueError(f"Unknown loss type: {cfg['type']}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg):
    torch.manual_seed(cfg['experiment']['seed'])

    wandb.init(
        project='contrastive-ts',
        name=cfg['experiment']['name'],
        config=cfg,
    )

    # Build everything
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    gen = build_signal(cfg['signal'])
    modality = build_modality(cfg['modality'], gen)
    model = build_model(cfg['model'], modality).to(device)
    loss_fn = build_loss(cfg['loss'], model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    # Logs
    history = {
        'step': [],
        'loss': [],
        'theory_err_steps': [],
        'theory_err': [],
        'forecast_mse': [],
        'res_cov_error': [],
        #'A_hat' : [], 
        #A_entry_00' : [],
    }

    # Resolve which analytical A* theory_match compares against
    theory_target = resolve_theory_target(cfg['loss'])
    # Resolve which model conditional-mean head the forecast metrics should use
    mean_family = resolve_mean_family(cfg['loss'])

    # Pre-training forecast baseline
    pre_mse = forecast_mse(model, modality, mean_family, n_samples=500)
    pre_cov_err = residual_cov_error(model, modality, mean_family, n_samples=1000)
    wandb.log({'forecast_mse_pre': pre_mse.item(), 'cov_error_pre': pre_cov_err.item()}, step=0)
    print(f"pre-training | forecast_mse {pre_mse.item():.4f} | cov_error {pre_cov_err.item():.4f}")

    # Training
    for step in range(cfg['training']['num_steps']):
        u, v = modality.sample_pair(cfg['training']['batch_size'])
        u, v = u.to(device), v.to(device)
        u_features, v_features, logit_scale = model(u.float(), v.float())
        loss = loss_fn(u_features, v_features, logit_scale, u, v)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss
        if step % cfg['logging']['log_every'] == 0:
            history['step'].append(step)
            history['loss'].append(loss.item())
            #result = debug_Ahat(model, modality)
            #history['A_hat'].append(result['A_norm'])
            #history['A_entry_00'].append(result['A_entry_00'])
            wandb.log({'loss': loss.item()}, step=step)
            #print(f"A_hat = {result['A_norm']:.4f} | A_entry_00 = {result['A_entry_00']:.4f} | A - K = {result['A_minus_K']:.4f} ")

        # Log theory match error
        if step % cfg['logging']['theory_match_every'] == 0:
            err = theory_match_error(model, modality, theory_target)
            err_val = err.item() if err is not None else float('nan')
            dir_err, scale = theory_match_dir_error(model, modality, theory_target)
            dir_err_val = dir_err.item() if dir_err is not None else float('nan')
            scale_val = scale.item() if scale is not None else float('nan')
            history['theory_err_steps'].append(step)
            history['theory_err'].append(err_val)
            self_sim, rand_sim = compute_similarities(model, modality)
            res_cov_error = residual_cov_error(model, modality, mean_family, n_samples = 1000)
            fore_mse = forecast_mse(model, modality, mean_family, n_samples=500)
            mdl_cov_err = model_cov_error(model, modality, mean_family)  # None for mse
            history['forecast_mse'].append(fore_mse)
            history['res_cov_error'].append(res_cov_error)
            logit_scale = model.logit_scale.exp().item()
            wp_norm = (model.u_encoder.weight.T @ model.v_encoder.weight).norm().item()
            H = model.v_encoder.weight
            v_pred = u_features @ H
            v_pred_mean = v_pred.mean(dim=0).norm()
            v_pred_std = v_pred.std(dim=0).mean()
            v_mean = v.mean(dim=0).norm()
            v_std = v.std(dim=0).mean()
            wandb.log({
                'theory_err': err_val,
                'theory_err_dir': dir_err_val,
                'theory_err_scale': scale_val,
                'similarity_ratio': (self_sim / rand_sim).item(),
                'self_sim': self_sim.item(),
                'rand_sim': rand_sim.item(),
                'logit_scale': logit_scale,
                'weight_product_norm': wp_norm,
                'forecast_mse': fore_mse,
                'res_cov_error': res_cov_error,
                'model_cov_error': mdl_cov_err.item() if mdl_cov_err is not None else float('nan'),
                #'v_mean': v_mean,
                #v_std': v_std,
                #'v_pred_mean': v_pred_mean,
                #'v_pred_std': v_pred_std,
            }, step=step)
            print(f"step {step:6d} | loss {loss.item():.4f} | theory_err {err_val:.4f} | E_dir {dir_err_val:.4f} | scale {scale_val:.4f}")
            print(f"  logit_scale={logit_scale:.4f} | weight_product_norm={wp_norm:.4f} ")

    # Post-training forecast
    post_mse = forecast_mse(model, modality, mean_family, n_samples=500)
    post_cov_err = residual_cov_error(model, modality, mean_family, n_samples=1000)
    final_step = cfg['training']['num_steps'] - 1
    wandb.log({'forecast_mse_post': post_mse.item(), 'cov_error_post': post_cov_err.item()}, step=final_step)
    print(f"post-training | forecast_mse {post_mse.item():.4f} | cov_error {post_cov_err.item():.4f}")

    # Post-training retrieval / similarity eval on a fixed test batch
    eval_metrics = evaluate_encoders(model, modality, n_eval=512, seed=1234)
    wandb.log(eval_metrics, step=final_step)
    history['eval'] = eval_metrics
    print("eval | " + " | ".join(
        f"{k.split('/')[-1]} {v:.4f}" for k, v in eval_metrics.items()
    ))

    wandb.finish()
    return model, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# WILL BE REPLACED BY WANDB ONCE I GET MY HANDS TO THAT 
'''
def plot_results(history, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['step'], history['loss'])
    axes[0].set_xlabel('step'); axes[0].set_ylabel('CLIP loss')
    axes[0].set_title('Training loss')

    axes[1].plot(history['theory_err_steps'], history['theory_err'])
    axes[1].set_xlabel('step'); axes[1].set_ylabel('rel. Frobenius error')
    axes[1].set_title('Theory match: ||G^T H - A*|| / ||A*||')
    axes[1].set_yscale('log')

    axes[2].plot(history['forecast_steps'], history['forecast_mse'])
    axes[2].set_xlabel('step'); axes[2].set_ylabel('MSE')
    axes[2].set_title('Forecast MSE vs. analytical conditional mean')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_curves.png', dpi=150)
    plt.close() '''


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the config alongside results (for reproducibility)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    model, history = run(cfg)
    #plot_results(history, output_dir)

    # Save model + history
    torch.save(model.state_dict(), output_dir / 'model.pt')
    torch.save(history, output_dir / 'history.pt')

    print(f"\nDone. Results in {output_dir}/")


if __name__ == '__main__':
    main()