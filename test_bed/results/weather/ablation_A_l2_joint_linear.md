# Modality ablation, setup A, L² joint (linear)

Each row retrains with one group dropped from the past. Δ = change against the full model (negative skill change = the group helped). Test period, mean over seeds.

| dropped | n | skill vs clim | Δ skill | CRPS 1h / 6h | Δ CRPS 6h | cov90 | width90 | cov90 by past-vol |
|---|---|---|---|---|---|---|---|---|
| (none) | 2 | 0.816 | — | 0.38 / 1.22 | — | 0.88 | 4.25 | 0.94 / 0.85 / 0.83 |
| temperature | 2 | 0.360 | -0.456 | 3.06 / 3.15 | +1.932 | 0.54 | 9.53 | 0.45 / 0.64 / 0.52 |
| moisture | 2 | 0.806 | -0.010 | 0.41 / 1.28 | +0.065 | 0.88 | 4.50 | 0.94 / 0.86 / 0.83 |
| pressure | 2 | 0.807 | -0.009 | 0.41 / 1.27 | +0.049 | 0.86 | 4.28 | 0.92 / 0.84 / 0.81 |
| wind | 2 | 0.812 | -0.004 | 0.38 / 1.26 | +0.038 | 0.88 | 4.31 | 0.94 / 0.87 / 0.83 |
| radiation | 2 | 0.803 | -0.013 | 0.41 / 1.33 | +0.107 | 0.88 | 4.57 | 0.93 / 0.86 / 0.84 |
| rain | 2 | 0.813 | -0.004 | 0.41 / 1.22 | -0.001 | 0.87 | 4.31 | 0.94 / 0.85 / 0.82 |
