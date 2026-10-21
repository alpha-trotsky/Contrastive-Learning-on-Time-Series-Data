# Modality ablation, setup A, Gauss NLL (linear)

Each row retrains with one group dropped from the past. Δ = change against the full model (negative skill change = the group helped). Test period, mean over seeds.

| dropped | n | skill vs clim | Δ skill | CRPS 1h / 6h | Δ CRPS 6h | cov90 | width90 | cov90 by past-vol |
|---|---|---|---|---|---|---|---|---|
| (none) | 2 | 0.839 | — | 0.34 / 1.04 | — | 0.90 | 4.26 | 0.95 / 0.90 / 0.85 |
| temperature | 2 | 0.416 | -0.423 | 2.72 / 2.87 | +1.830 | 0.82 | 13.33 | 0.84 / 0.83 / 0.80 |
| moisture | 2 | 0.834 | -0.005 | 0.34 / 1.10 | +0.063 | 0.92 | 4.60 | 0.95 / 0.92 / 0.89 |
| pressure | 2 | 0.837 | -0.002 | 0.35 / 1.10 | +0.060 | 0.91 | 4.35 | 0.95 / 0.90 / 0.87 |
| wind | 2 | 0.839 | +0.000 | 0.33 / 1.05 | +0.016 | 0.91 | 4.37 | 0.96 / 0.91 / 0.87 |
| radiation | 2 | 0.825 | -0.014 | 0.35 / 1.18 | +0.142 | 0.91 | 4.71 | 0.94 / 0.90 / 0.89 |
| rain | 2 | 0.837 | -0.001 | 0.33 / 1.06 | +0.021 | 0.90 | 4.18 | 0.95 / 0.90 / 0.84 |
