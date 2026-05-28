"""Run a single contrastive-learning experiment per the config.

Usage:
    python experiment.py --config config.yaml
"""
from pathlib import Path
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from modalities.past_future import PastFutureModality
from signals.gaussian_process import GaussianProcess1D
from modalities.field_coeff import FieldCoeffModality
from models.linear_clip import LinearCLIP
from losses.clip_losses import CLIPConditionalLoss, CLIPJointLoss
from evaluation.theory_match import theory_match_error
from evaluation.forecast import forecast_mse


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


def build_loss(cfg):
    inner_product = cfg['inner_product']
    if cfg['type'] == 'conditional':
        return CLIPConditionalLoss(
            lambda_u=cfg['lambda_u'],
            lambda_v=cfg['lambda_v'],
            inner_product=inner_product,
        )
    elif cfg['type'] == 'joint':
        return CLIPJointLoss(inner_product=inner_product)
    elif cfg['type'] == 'one_sided_u':
        return CLIPConditionalLoss(lambda_u=1.0, lambda_v=0.0, inner_product=inner_product)
    elif cfg['type'] == 'one_sided_v':
        return CLIPConditionalLoss(lambda_u=0.0, lambda_v=1.0, inner_product=inner_product)
    else:
        raise ValueError(f"Unknown loss type: {cfg['type']}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg):
    torch.manual_seed(cfg['experiment']['seed'])

    # Build everything
    gen = build_signal(cfg['signal'])
    modality = build_modality(cfg['modality'], gen)
    model = build_model(cfg['model'], modality)
    loss_fn = build_loss(cfg['loss'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    # Logs
    history = {
        'step': [],
        'loss': [],
        'theory_err_steps': [],
        'theory_err': [],
        'forecast_steps': [],
        'forecast_mse': [],
    }

    # Resolve loss type label for theory_match
    theory_loss_type = (
        'joint' if cfg['loss']['type'] == 'joint' else 'conditional'
    )

    # Training
    for step in range(cfg['training']['num_steps']):
        u, v = modality.sample_pair(cfg['training']['batch_size'])
        u_features, v_features, logit_scale = model(u.float(), v.float())
        loss = loss_fn(u_features, v_features, logit_scale)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss
        if step % cfg['logging']['log_every'] == 0:
            history['step'].append(step)
            history['loss'].append(loss.item())

        # Log theory match error
        if step % cfg['logging']['theory_match_every'] == 0:
            err = theory_match_error(model, modality, theory_loss_type)
            history['theory_err_steps'].append(step)
            history['theory_err'].append(err.item())
            print(f"step {step:6d} | loss {loss.item():.4f} | theory_err {err.item():.4f}")

        # Log forecast MSE
        if step % cfg['logging']['forecast_every'] == 0:
            mse = forecast_mse(model, modality, n_samples=500)
            history['forecast_steps'].append(step)
            history['forecast_mse'].append(mse.item())

    return model, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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
    plt.close()


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
    plot_results(history, output_dir)

    # Save model + history
    torch.save(model.state_dict(), output_dir / 'model.pt')
    torch.save(history, output_dir / 'history.pt')

    print(f"\nDone. Results in {output_dir}/")


if __name__ == '__main__':
    main()