# Weather sweep, setup A+trainbank: temperature + moisture + pressure + wind + radiation + rain → future temperature

Test period 2020-10-21T01:30 – 2021-01-01T00:00, 1000 test queries, bank = 36708 held-out futures (train). Mean over seeds (n = number of seeds). CRPS and width in the units of `T (degC)`.

## Main table

| model | enc | n | skill vs clim | skill vs persist | CRPS 10m / 1h / 3h / 6h | cov50 | cov90 | width90 | bias | ESS med / min | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L² joint | mlp | 2 | 0.817 | 0.427 | 0.261 / 0.427 / 0.768 / 1.09 | 0.42 | 0.83 | 3.56 | +0.221 | 491 / 15.1 | 3500 |
| L² joint | linear | 2 | 0.823 | 0.447 | 0.161 / 0.355 / 0.737 / 1.09 | 0.56 | 0.91 | 4.29 | +0.266 | 657 / 24.3 | 7250 |
| *climatology* | — | — | 0.000 | -2.124 | 4.55 / 4.54 / 4.52 / 4.49 | — | 0.84 | — | — | — | — |
| *persistence* | — | — | 0.680 | 0.000 | 0.121 / 0.535 / 1.29 / 2.14 | — | — | — | — | — | — |
| *seasonal-naive* | — | — | 0.491 | -0.589 | 2.22 / 2.2 / 2.2 / 2.18 | — | — | — | — | — | — |

## Calibration detail

cov90 per query, by tercile of a difficulty proxy (calm / mid / hard), coverage by horizon, and the PIT histogram (ideal: 0.20 of values in the two outer bins, 0.50 in the lower half).

| model | enc | by past-vol | by NLL width | cov90 at 10m / 1h / 3h / 6h | PIT outer bins | PIT lower half |
|---|---|---|---|---|---|---|
| L² joint | mlp | 0.86 / 0.82 / 0.81 | — / — / — | 0.93 / 0.84 / 0.82 / 0.80 | 0.28 | 0.57 |
| L² joint | linear | 0.95 / 0.89 / 0.88 | — / — / — | 0.99 / 0.93 / 0.90 / 0.89 | 0.17 | 0.58 |
