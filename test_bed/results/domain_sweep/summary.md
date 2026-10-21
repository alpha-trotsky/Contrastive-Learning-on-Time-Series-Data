
### Heavy tails (Student-t), dial = df (inf = GP)

| dial | model | energy vs oracle | skill | avg err vs oracle | band90 | band90 easy / mid / hard | width90 / oracle | seeds |
|---|---|---|---|---|---|---|---|---|
| inf | cosine | +5.6% | 0.248 | 1.01x | 0.980 | 0.98 / 0.98 / 0.98 | 1.47 | 2 |
| inf | l2 | +0.0% | 0.288 | 1.00x | 0.891 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| inf | mse | +0.1% | 0.287 | 1.00x | 0.892 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| 10 | cosine | +12.9% | 0.198 | 1.16x | 0.978 | 1.00 / 0.99 / 0.95 | 1.65 | 2 |
| 10 | l2 | +1.2% | 0.281 | 1.00x | 0.888 | 0.95 / 0.91 / 0.80 | 0.99 | 2 |
| 10 | mse | +1.6% | 0.278 | 1.00x | 0.894 | 0.96 / 0.91 / 0.81 | 1.02 | 2 |
| 3 | cosine | +999.0% | -6.505 | 212.62x | 0.827 | 0.95 / 0.89 / 0.64 | 3.58 | 2 |
| 3 | l2 | +4.6% | 0.287 | 1.02x | 0.888 | 0.98 / 0.92 / 0.76 | 1.00 | 2 |
| 3 | mse | +8.6% | 0.260 | 1.01x | 0.929 | 1.00 / 0.98 / 0.81 | 1.24 | 2 |

SIR diagnostics: inf/cosine: ESS median 31183, min 4.8, whitened theory err 0.138; inf/l2: ESS median 11193, min 20.8, whitened theory err 0.013; 10/cosine: ESS median 37455, min 1.7, whitened theory err 0.246; 10/l2: ESS median 11356, min 90.3, whitened theory err 0.018; 3/cosine: ESS median 29666, min 1.0, whitened theory err 0.482; 3/l2: ESS median 14260, min 3.4, whitened theory err 0.166

### Bent GP (warp), dial = strength (0 = GP)

| dial | model | energy vs oracle | skill | avg err vs oracle | band90 | band90 easy / mid / hard | width90 / oracle | seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | cosine | +5.6% | 0.250 | 1.01x | 0.981 | 0.98 / 0.98 / 0.98 | 1.48 | 2 |
| 0 | l2 | -0.0% | 0.290 | 1.00x | 0.893 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| 0 | mse | +0.0% | 0.290 | 1.00x | 0.894 | 0.89 / 0.90 / 0.89 | 1.00 | 2 |
| 0.5 | cosine | +17.1% | 0.156 | 1.46x | 0.964 | 0.95 / 0.98 / 0.96 | 1.59 | 2 |
| 0.5 | l2 | +0.2% | 0.278 | 1.00x | 0.893 | 0.90 / 0.89 / 0.89 | 1.00 | 2 |
| 0.5 | mse | +1.7% | 0.266 | 1.00x | 0.905 | 0.97 / 0.93 / 0.82 | 1.06 | 2 |
| 1 | cosine | +40.0% | -0.017 | 2.80x | 0.926 | 0.88 / 0.98 / 0.92 | 1.77 | 2 |
| 1 | l2 | +1.6% | 0.261 | 1.02x | 0.905 | 0.91 / 0.91 / 0.89 | 1.02 | 2 |
| 1 | mse | +12.7% | 0.180 | 1.01x | 0.920 | 0.99 / 0.96 / 0.80 | 1.39 | 2 |

SIR diagnostics: 0/cosine: ESS median 31440, min 5.2; 0/l2: ESS median 10763, min 36.2; 0.5/cosine: ESS median 40139, min 1.5; 0.5/l2: ESS median 12705, min 10.9; 1/cosine: ESS median 48264, min 1.1; 1/l2: ESS median 21512, min 5.1

### Double-well, dial = lambda (0 = OU)

| dial | model | energy vs oracle | skill | avg err vs oracle | band90 | band90 easy / mid / hard | width90 / oracle | seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | cosine | +0.4% | 0.091 | 1.00x | 0.919 | 0.92 / 0.92 / 0.91 | 1.13 | 2 |
| 0 | l2 | +0.1% | 0.094 | 1.00x | 0.882 | 0.89 / 0.88 / 0.88 | 0.99 | 2 |
| 0 | mse | -0.1% | 0.096 | 1.00x | 0.889 | 0.90 / 0.88 / 0.89 | 1.00 | 2 |
| 1 | cosine | +1.5% | 0.476 | 1.02x | 0.913 | 0.91 / 0.92 / 0.92 | 1.13 | 2 |
| 1 | l2 | +0.7% | 0.480 | 1.02x | 0.889 | 0.89 / 0.89 / 0.88 | 0.99 | 2 |
| 1 | mse | +10.9% | 0.427 | 1.06x | 0.906 | 0.93 / 0.94 / 0.85 | 1.05 | 2 |
| 1.5 | cosine | +2.6% | 0.674 | 1.09x | 0.904 | 0.90 / 0.90 / 0.91 | 1.11 | 2 |
| 1.5 | l2 | +0.4% | 0.681 | 1.01x | 0.894 | 0.89 / 0.89 / 0.89 | 0.99 | 2 |
| 1.5 | mse | +14.1% | 0.638 | 1.13x | 0.939 | 0.96 / 0.95 / 0.91 | 1.37 | 2 |

SIR diagnostics: 0/cosine: ESS median 31491, min 23.5; 0/l2: ESS median 18188, min 19.3; 1/cosine: ESS median 21377, min 10374.9; 1/l2: ESS median 16514, min 2121.4; 1.5/cosine: ESS median 23028, min 39.6; 1.5/l2: ESS median 20017, min 732.4
