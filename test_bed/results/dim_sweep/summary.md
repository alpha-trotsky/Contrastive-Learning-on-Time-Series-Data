# Dimension sweep

Seed 0. info = nats captured on fresh pairs; ES = energy score above oracle (%) at the largest bank; OF = over-forecast factor.

## t3: by embedding dimension

| model | quantity | 1 | 2 | 4 | 8 | 16 | 32 | 50 |
|---|---|---|---|---|---|---|---|---|
| inner product, $\tanh$ future | info | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 |
| inner product, $\tanh$ future | es | 20.9 | 20.7 | 20.9 | 23.3 | 27.2 | 28.5 | 33.0 |
| inner product, $\tanh$ future | of4 | 2.3 | 2.7 | 7.4 | 17.5 | 24.7 | 34.8 | 47.3 |
| $L^2$ one-way | info | 1.46 | 1.45 | 1.45 | 1.45 | 1.44 | 1.44 | 1.43 |
| $L^2$ one-way | es | 5.0 | 5.1 | 5.1 | 5.1 | 5.0 | 4.9 | 6.6 |
| $L^2$ one-way | of4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Student-$t$ kernel, one-way | info | 1.61 | 1.64 | 1.69 | 1.72 | 1.76 | 1.81 | 1.83 |
| Student-$t$ kernel, one-way | es | 5.1 | 4.9 | 5.0 | 4.9 | 4.9 | 4.9 | 4.9 |
| Student-$t$ kernel, one-way | of4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MLP, Student-$t$ kernel, one-way | info | 1.57 | 2.64 | 2.73 | 2.68 |  | 2.60 |  |
| MLP, Student-$t$ kernel, one-way | es | 5.2 | 1.4 | 0.4 | 0.4 |  | 0.6 |  |
| MLP, Student-$t$ kernel, one-way | of4 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 |  |

## ar0.6: by embedding dimension

| model | quantity | 1 | 2 | 4 | 8 | 16 | 32 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|
| inner product, $\tanh$ future | info | 0.11 | 0.11 | 0.11 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.08 |
| inner product, $\tanh$ future | es | 5.0 | 5.0 | 5.1 | 5.0 | 5.1 | 4.8 | 4.8 | 4.5 | 4.6 |
| inner product, $\tanh$ future | of4 | 1.9 | 1.9 | 2.1 | 2.9 | 9.1 | 7.1 | 10.6 | 5.1 | 3.2 |
| normalized cosine | info | -0.00 | 0.10 | 0.12 | 0.14 | 0.15 | 0.16 | 0.16 | 0.15 | 0.15 |
| normalized cosine | es | 5.5 | 5.3 | 5.3 | 5.2 | 5.3 | 5.2 | 5.3 | 5.2 | 5.2 |
| normalized cosine | of4 | 1.3 | 1.3 | 1.3 | 1.3 | 1.2 | 1.2 | 1.2 | 1.2 | 1.2 |
| $L^2$ one-way | info | 0.06 | 0.06 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | -0.01 |
| $L^2$ one-way | es | 4.5 | 4.7 | 4.7 | 4.7 | 4.8 | 4.6 | 4.6 | 4.6 | 4.3 |
| $L^2$ one-way | of4 | 0.4 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Student-$t$ kernel, one-way | info | 0.21 | 0.22 | 0.23 | 0.24 | 0.28 | 0.32 | 0.33 | 0.34 | 0.34 |
| Student-$t$ kernel, one-way | es | 4.4 | 4.5 | 4.7 | 4.6 | 4.5 | 4.4 | 4.6 | 4.5 | 4.5 |
| Student-$t$ kernel, one-way | of4 | 0.3 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MLP, inner product, $\tanh$ future | info | 0.54 | 1.12 | 1.27 | 1.25 |  | 1.18 |  | 1.11 |  |
| MLP, inner product, $\tanh$ future | es | 2.0 | 0.6 | 0.1 | 0.5 |  | 3.8 |  | 4.2 |  |
| MLP, inner product, $\tanh$ future | of4 | 2.3 | 3.0 | 3.2 | 4.4 |  | 18.2 |  | 12.3 |  |
| MLP, Student-$t$ kernel, one-way | info | 1.06 | 1.18 | 1.13 | 1.12 |  | 1.06 |  | 1.06 |  |
| MLP, Student-$t$ kernel, one-way | es | 0.5 | 0.2 | -0.0 | 0.0 |  | 0.4 |  | 0.4 |  |
| MLP, Student-$t$ kernel, one-way | of4 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 |  | 0.0 |  |

## AR(1) Student-t: by lag-one correlation (default embedding 100)

| model | ar0 | ar0.3 | ar0.6 | ar0.9 | ar0.99 |
|---|---|---|---|---|---|
| available (I_corr + I_scale) | 1.20 | 1.25 | 1.43 | 2.03 | 3.16 |
| inner product, $\tanh$ future | -0.05 | 0.01 | 0.10 | 0.21 | 0.28 |
| normalized cosine | -0.01 | 0.03 | 0.15 | 0.44 | 0.62 |
| $L^2$ one-way | -0.47 | -0.11 | 0.03 | 0.54 | 1.80 |
| Student-$t$ kernel, one-way | 0.14 | 0.17 | 0.34 | 0.98 | 2.16 |
| log-quadratic, joint | 1.09 | 1.11 | 1.24 | 1.65 | 2.51 |
| MLP, inner product, $\tanh$ future | 0.74 | 0.83 | 1.11 | 1.74 | 2.52 |
| MLP, Student-$t$ kernel, one-way | 0.77 | 0.82 | 1.06 | 1.79 | 2.74 |
