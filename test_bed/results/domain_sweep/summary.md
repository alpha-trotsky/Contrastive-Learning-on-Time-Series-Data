
### Energy vs perfect forecaster (0% = perfect)

| domain | dial | cosine one-sided | cosine two-sided | cosine joint | L2 one-sided | L2 two-sided | L2 joint | MSE + noise |
|---|---|---|---|---|---|---|---|---|
| t | inf | +5.6% | +5.6% | +9.8% | +0.0% | +0.0% | +0.0% | +0.1% |
| t | 10 | +12.9% | +12.9% | +19.5% | +1.2% | +1.3% | +1.2% | +1.6% |
| t | 3 | +940.9% | +999.0% | +69.5% | +4.6% | +4.5% | +4.5% | +8.6% |
| warp | 0 | +5.5% | +5.6% | +12.5% | -0.0% | -0.0% | +0.0% | +0.0% |
| warp | 0.5 | +16.2% | +17.1% | +23.1% | +0.2% | +0.3% | +0.3% | +1.7% |
| warp | 1 | +37.4% | +40.0% | +25.2% | +1.6% | +1.5% | +1.6% | +12.7% |
| dwell | 0 | +0.3% | +0.4% | +1.5% | +0.1% | -0.1% | +0.1% | -0.1% |
| dwell | 1 | +1.5% | +1.5% | +3.3% | +0.7% | +0.5% | +0.6% | +10.9% |
| dwell | 1.5 | +1.3% | +2.6% | +2.7% | +0.4% | +0.1% | +0.3% | +14.1% |

### 90% band coverage on the HARD third of cases (target 0.90)

| domain | dial | cosine one-sided | cosine two-sided | cosine joint | L2 one-sided | L2 two-sided | L2 joint | MSE + noise |
|---|---|---|---|---|---|---|---|---|
| t | inf | 0.98 | 0.98 | 0.97 | 0.89 | 0.89 | 0.89 | 0.89 |
| t | 10 | 0.95 | 0.95 | 0.92 | 0.80 | 0.80 | 0.81 | 0.81 |
| t | 3 | 0.65 | 0.64 | 0.78 | 0.76 | 0.76 | 0.77 | 0.81 |
| warp | 0 | 0.98 | 0.98 | 0.97 | 0.89 | 0.89 | 0.89 | 0.89 |
| warp | 0.5 | 0.97 | 0.96 | 0.94 | 0.89 | 0.89 | 0.89 | 0.82 |
| warp | 1 | 0.93 | 0.92 | 0.93 | 0.89 | 0.89 | 0.89 | 0.80 |
| dwell | 0 | 0.91 | 0.91 | 0.91 | 0.88 | 0.87 | 0.88 | 0.89 |
| dwell | 1 | 0.91 | 0.92 | 0.92 | 0.88 | 0.89 | 0.89 | 0.85 |
| dwell | 1.5 | 0.91 | 0.91 | 0.92 | 0.89 | 0.89 | 0.90 | 0.91 |

### 90% band coverage overall (target 0.90)

| domain | dial | cosine one-sided | cosine two-sided | cosine joint | L2 one-sided | L2 two-sided | L2 joint | MSE + noise |
|---|---|---|---|---|---|---|---|---|
| t | inf | 0.98 | 0.98 | 0.97 | 0.89 | 0.89 | 0.89 | 0.89 |
| t | 10 | 0.98 | 0.98 | 0.96 | 0.89 | 0.89 | 0.89 | 0.89 |
| t | 3 | 0.84 | 0.83 | 0.91 | 0.89 | 0.89 | 0.89 | 0.93 |
| warp | 0 | 0.98 | 0.98 | 0.97 | 0.89 | 0.89 | 0.89 | 0.89 |
| warp | 0.5 | 0.97 | 0.96 | 0.94 | 0.89 | 0.89 | 0.89 | 0.90 |
| warp | 1 | 0.93 | 0.93 | 0.92 | 0.90 | 0.90 | 0.91 | 0.92 |
| dwell | 0 | 0.92 | 0.92 | 0.92 | 0.88 | 0.88 | 0.88 | 0.89 |
| dwell | 1 | 0.91 | 0.91 | 0.92 | 0.89 | 0.89 | 0.90 | 0.91 |
| dwell | 1.5 | 0.90 | 0.90 | 0.92 | 0.89 | 0.89 | 0.90 | 0.94 |

### Heavy tails (Student-t), dial = df (inf = GP)

| dial | model | energy vs oracle | skill | avg err vs oracle | band90 | band90 easy / mid / hard | width90 / oracle | seeds |
|---|---|---|---|---|---|---|---|---|
| inf | cosine one-sided | +5.6% | 0.248 | 1.01x | 0.980 | 0.98 / 0.98 / 0.98 | 1.47 | 2 |
| inf | cosine two-sided | +5.6% | 0.248 | 1.01x | 0.980 | 0.98 / 0.98 / 0.98 | 1.47 | 2 |
| inf | cosine joint | +9.8% | 0.218 | 1.14x | 0.971 | 0.97 / 0.97 / 0.97 | 1.47 | 2 |
| inf | L2 one-sided | +0.0% | 0.288 | 1.00x | 0.891 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| inf | L2 two-sided | +0.0% | 0.288 | 1.00x | 0.891 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| inf | L2 joint | +0.0% | 0.288 | 1.00x | 0.892 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| inf | MSE + noise | +0.1% | 0.287 | 1.00x | 0.892 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| 10 | cosine one-sided | +12.9% | 0.198 | 1.17x | 0.979 | 1.00 / 0.99 / 0.95 | 1.65 | 2 |
| 10 | cosine two-sided | +12.9% | 0.198 | 1.16x | 0.978 | 1.00 / 0.99 / 0.95 | 1.65 | 2 |
| 10 | cosine joint | +19.5% | 0.151 | 1.36x | 0.960 | 0.99 / 0.97 / 0.92 | 1.51 | 2 |
| 10 | L2 one-sided | +1.2% | 0.281 | 1.00x | 0.888 | 0.95 / 0.91 / 0.80 | 0.99 | 2 |
| 10 | L2 two-sided | +1.3% | 0.280 | 1.00x | 0.886 | 0.95 / 0.90 / 0.80 | 0.99 | 2 |
| 10 | L2 joint | +1.2% | 0.281 | 1.00x | 0.891 | 0.95 / 0.91 / 0.81 | 1.00 | 2 |
| 10 | MSE + noise | +1.6% | 0.278 | 1.00x | 0.894 | 0.96 / 0.91 / 0.81 | 1.02 | 2 |
| 3 | cosine one-sided | +940.9% | -6.113 | 200.46x | 0.837 | 0.96 / 0.90 / 0.65 | 3.47 | 2 |
| 3 | cosine two-sided | +999.0% | -6.505 | 212.62x | 0.827 | 0.95 / 0.89 / 0.64 | 3.58 | 2 |
| 3 | cosine joint | +69.5% | -0.157 | 8.67x | 0.914 | 1.00 / 0.97 / 0.78 | 1.73 | 2 |
| 3 | L2 one-sided | +4.6% | 0.287 | 1.02x | 0.888 | 0.98 / 0.92 / 0.76 | 1.00 | 2 |
| 3 | L2 two-sided | +4.5% | 0.288 | 1.00x | 0.888 | 0.98 / 0.92 / 0.76 | 1.01 | 2 |
| 3 | L2 joint | +4.5% | 0.288 | 1.00x | 0.893 | 0.98 / 0.93 / 0.77 | 1.03 | 2 |
| 3 | MSE + noise | +8.6% | 0.260 | 1.01x | 0.929 | 1.00 / 0.98 / 0.81 | 1.24 | 2 |

SIR diagnostics: inf/cosine_1s: ESS median 31318, min 4.5, whitened theory err 0.135; inf/cosine: ESS median 31183, min 4.8, whitened theory err 0.138; inf/cosine_joint: ESS median 41412, min 306.8, whitened theory err 0.179; inf/l2: ESS median 11193, min 20.8, whitened theory err 0.013; inf/l2_2s: ESS median 11164, min 24.6; inf/l2_joint: ESS median 11281, min 24.3; 10/cosine_1s: ESS median 37346, min 1.7, whitened theory err 0.241; 10/cosine: ESS median 37455, min 1.7, whitened theory err 0.246; 10/cosine_joint: ESS median 46729, min 218.7, whitened theory err 0.364; 10/l2: ESS median 11356, min 90.3, whitened theory err 0.018; 10/l2_2s: ESS median 11385, min 108.6; 10/l2_joint: ESS median 11609, min 109.3; 3/cosine_1s: ESS median 32506, min 1.0, whitened theory err 0.497; 3/cosine: ESS median 29666, min 1.0, whitened theory err 0.482; 3/cosine_joint: ESS median 49889, min 1.1, whitened theory err 0.885; 3/l2: ESS median 14260, min 3.4, whitened theory err 0.166; 3/l2_2s: ESS median 14513, min 12.2; 3/l2_joint: ESS median 15986, min 21.1

### Bent GP (warp), dial = strength (0 = GP)

| dial | model | energy vs oracle | skill | avg err vs oracle | band90 | band90 easy / mid / hard | width90 / oracle | seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | cosine one-sided | +5.5% | 0.251 | 1.01x | 0.981 | 0.98 / 0.98 / 0.98 | 1.47 | 2 |
| 0 | cosine two-sided | +5.6% | 0.250 | 1.01x | 0.981 | 0.98 / 0.98 / 0.98 | 1.48 | 2 |
| 0 | cosine joint | +12.5% | 0.201 | 1.23x | 0.967 | 0.97 / 0.97 / 0.97 | 1.47 | 2 |
| 0 | L2 one-sided | -0.0% | 0.290 | 1.00x | 0.893 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| 0 | L2 two-sided | -0.0% | 0.290 | 1.00x | 0.892 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| 0 | L2 joint | +0.0% | 0.290 | 1.00x | 0.891 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| 0 | MSE + noise | +0.0% | 0.290 | 1.00x | 0.894 | 0.89 / 0.90 / 0.89 | 1.00 | 2 |
| 0.5 | cosine one-sided | +16.2% | 0.162 | 1.42x | 0.966 | 0.95 / 0.98 / 0.97 | 1.59 | 2 |
| 0.5 | cosine two-sided | +17.1% | 0.156 | 1.46x | 0.964 | 0.95 / 0.98 / 0.96 | 1.59 | 2 |
| 0.5 | cosine joint | +23.1% | 0.112 | 1.53x | 0.940 | 0.90 / 0.98 / 0.94 | 1.52 | 2 |
| 0.5 | L2 one-sided | +0.2% | 0.278 | 1.00x | 0.893 | 0.90 / 0.89 / 0.89 | 1.00 | 2 |
| 0.5 | L2 two-sided | +0.3% | 0.277 | 1.00x | 0.892 | 0.89 / 0.89 / 0.89 | 1.00 | 2 |
| 0.5 | L2 joint | +0.3% | 0.277 | 1.00x | 0.893 | 0.90 / 0.89 / 0.89 | 1.00 | 2 |
| 0.5 | MSE + noise | +1.7% | 0.266 | 1.00x | 0.905 | 0.97 / 0.93 / 0.82 | 1.06 | 2 |
| 1 | cosine one-sided | +37.4% | 0.002 | 2.62x | 0.927 | 0.88 / 0.98 / 0.93 | 1.76 | 2 |
| 1 | cosine two-sided | +40.0% | -0.017 | 2.80x | 0.926 | 0.88 / 0.98 / 0.92 | 1.77 | 2 |
| 1 | cosine joint | +25.2% | 0.089 | 1.39x | 0.924 | 0.86 / 0.98 / 0.93 | 1.58 | 2 |
| 1 | L2 one-sided | +1.6% | 0.261 | 1.02x | 0.905 | 0.91 / 0.91 / 0.89 | 1.02 | 2 |
| 1 | L2 two-sided | +1.5% | 0.261 | 1.01x | 0.905 | 0.91 / 0.91 / 0.89 | 1.02 | 2 |
| 1 | L2 joint | +1.6% | 0.261 | 1.01x | 0.905 | 0.91 / 0.91 / 0.89 | 1.03 | 2 |
| 1 | MSE + noise | +12.7% | 0.180 | 1.01x | 0.920 | 0.99 / 0.96 / 0.80 | 1.39 | 2 |

SIR diagnostics: 0/cosine_1s: ESS median 31209, min 4.7; 0/cosine: ESS median 31440, min 5.2; 0/cosine_joint: ESS median 42385, min 1007.8; 0/l2: ESS median 10763, min 36.2; 0/l2_2s: ESS median 10734, min 44.2; 0/l2_joint: ESS median 10859, min 44.9; 0.5/cosine_1s: ESS median 40121, min 2.0; 0.5/cosine: ESS median 40139, min 1.5; 0.5/cosine_joint: ESS median 48442, min 1027.6; 0.5/l2: ESS median 12705, min 10.9; 0.5/l2_2s: ESS median 12660, min 13.9; 0.5/l2_joint: ESS median 12923, min 15.0; 1/cosine_1s: ESS median 48307, min 1.4; 1/cosine: ESS median 48264, min 1.1; 1/cosine_joint: ESS median 49640, min 3.5; 1/l2: ESS median 21512, min 5.1; 1/l2_2s: ESS median 21505, min 10.9; 1/l2_joint: ESS median 22900, min 12.0

### Double-well, dial = lambda (0 = OU)

| dial | model | energy vs oracle | skill | avg err vs oracle | band90 | band90 easy / mid / hard | width90 / oracle | seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | cosine one-sided | +0.3% | 0.092 | 1.00x | 0.919 | 0.92 / 0.92 / 0.91 | 1.13 | 2 |
| 0 | cosine two-sided | +0.4% | 0.091 | 1.00x | 0.919 | 0.92 / 0.92 / 0.91 | 1.13 | 2 |
| 0 | cosine joint | +1.5% | 0.081 | 1.02x | 0.917 | 0.92 / 0.92 / 0.91 | 1.13 | 2 |
| 0 | L2 one-sided | +0.1% | 0.094 | 1.00x | 0.882 | 0.89 / 0.88 / 0.88 | 0.99 | 2 |
| 0 | L2 two-sided | -0.1% | 0.096 | 1.00x | 0.879 | 0.89 / 0.88 / 0.87 | 0.98 | 2 |
| 0 | L2 joint | +0.1% | 0.094 | 1.00x | 0.882 | 0.89 / 0.88 / 0.88 | 0.99 | 2 |
| 0 | MSE + noise | -0.1% | 0.096 | 1.00x | 0.889 | 0.90 / 0.88 / 0.89 | 1.00 | 2 |
| 1 | cosine one-sided | +1.5% | 0.476 | 1.02x | 0.906 | 0.90 / 0.91 / 0.91 | 1.06 | 2 |
| 1 | cosine two-sided | +1.5% | 0.476 | 1.02x | 0.913 | 0.91 / 0.92 / 0.92 | 1.13 | 2 |
| 1 | cosine joint | +3.3% | 0.467 | 1.06x | 0.922 | 0.92 / 0.92 / 0.92 | 1.28 | 2 |
| 1 | L2 one-sided | +0.7% | 0.480 | 1.02x | 0.889 | 0.89 / 0.89 / 0.88 | 0.99 | 2 |
| 1 | L2 two-sided | +0.5% | 0.481 | 1.01x | 0.890 | 0.89 / 0.90 / 0.89 | 0.98 | 2 |
| 1 | L2 joint | +0.6% | 0.480 | 1.01x | 0.897 | 0.90 / 0.90 / 0.89 | 1.04 | 2 |
| 1 | MSE + noise | +10.9% | 0.427 | 1.06x | 0.906 | 0.93 / 0.94 / 0.85 | 1.05 | 2 |
| 1.5 | cosine one-sided | +1.3% | 0.678 | 1.03x | 0.899 | 0.90 / 0.89 / 0.91 | 1.04 | 2 |
| 1.5 | cosine two-sided | +2.6% | 0.674 | 1.09x | 0.904 | 0.90 / 0.90 / 0.91 | 1.11 | 2 |
| 1.5 | cosine joint | +2.7% | 0.674 | 1.09x | 0.915 | 0.91 / 0.91 / 0.92 | 1.34 | 2 |
| 1.5 | L2 one-sided | +0.4% | 0.681 | 1.01x | 0.894 | 0.89 / 0.89 / 0.89 | 0.99 | 2 |
| 1.5 | L2 two-sided | +0.1% | 0.682 | 1.01x | 0.893 | 0.89 / 0.89 / 0.89 | 0.98 | 2 |
| 1.5 | L2 joint | +0.3% | 0.681 | 1.01x | 0.897 | 0.90 / 0.90 / 0.90 | 1.00 | 2 |
| 1.5 | MSE + noise | +14.1% | 0.638 | 1.13x | 0.939 | 0.96 / 0.95 / 0.91 | 1.37 | 2 |

SIR diagnostics: 0/cosine_1s: ESS median 31443, min 24.6; 0/cosine: ESS median 31491, min 23.5; 0/cosine_joint: ESS median 40016, min 425.2; 0/l2: ESS median 18188, min 19.3; 0/l2_2s: ESS median 18037, min 38.5; 0/l2_joint: ESS median 18527, min 44.3; 1/cosine_1s: ESS median 21430, min 10046.6; 1/cosine: ESS median 21377, min 10374.9; 1/cosine_joint: ESS median 24934, min 10116.2; 1/l2: ESS median 16514, min 2121.4; 1/l2_2s: ESS median 16575, min 2448.2; 1/l2_joint: ESS median 17033, min 3258.1; 1.5/cosine_1s: ESS median 22467, min 63.3; 1.5/cosine: ESS median 23028, min 39.6; 1.5/cosine_joint: ESS median 24501, min 88.3; 1.5/l2: ESS median 20017, min 732.4; 1.5/l2_2s: ESS median 20007, min 1072.5; 1.5/l2_joint: ESS median 20885, min 1601.7
