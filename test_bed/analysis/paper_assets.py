"""Figures and generated table rows for notes/score_tails/main.tex.

Reads results/theory/saturation.json, results/tail_sweep/*.json and
results/amplification/*.json.  Writes PDF figures to notes/score_tails/figs/
and rewrites the blocks of main.tex between  %%BEGIN:<name>  and  %%END:<name>.

Usage (from test_bed/):  python -m analysis.paper_assets
"""
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.summarize_tail_sweep import load, ms, row_at, of_factor, of_oracle, MODELS

NOTE = Path('../notes/score_tails')
FIGS = NOTE / 'figs'
# categorical slots (validated order), light surface
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
     'green': '#008300', 'violet': '#6250d6', 'red': '#e34948'}
SEQ = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']        # ordinal blue ramp, light -> dark
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 1.6, 'lines.markersize': 3.5,
    'legend.frameon': False, 'pdf.fonttype': 42, 'figure.dpi': 150,
})


def end_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords='offset points', va='center', fontsize=7, color=INK)


# ---------------------------------------------------------------------------
# fixed scores
# ---------------------------------------------------------------------------

def fig_plateau():
    B = json.load(open('results/theory/saturation.json'))['B']
    M = B['models']
    keys = [('plateau A = 30', C['blue'], r'plateau, $A=30$'), ('plateau A = 1000', C['orange'], r'plateau, $A=1000$'),
            ('unbounded s = v', C['aqua'], r'unbounded, $s=v$')]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    N = [r['N'] for r in M['exact (A = 0)']]
    ax[0].plot(N, N, color=MUTED, lw=0.8, zorder=1)
    ax[0].annotate(r'ceiling $N$', (300, 300), xytext=(-4, 5), textcoords='offset points', ha='right', fontsize=7, color=MUTED)
    for A in (30, 1000):
        ax[0].axhline(A, color=MUTED, lw=0.8, zorder=1)
    for name, col, lab in keys:
        ax[0].plot(N, [r['of'] for r in M[name]], color=col, marker='o', label=lab, zorder=3)
    ax[0].set(xscale='log', yscale='log', xlabel=r'bank size $N$', ylim=(5, 3e4),
              title=r'over-forecast factor of $T$')
    ax[0].annotate(r'$A=1000$', (10, 1000), xytext=(0, 3), textcoords='offset points', fontsize=7, color=MUTED)
    ax[0].annotate(r'$A=30$', (1e6, 30), xytext=(0, 3), textcoords='offset points', ha='right', fontsize=7, color=MUTED)
    handles = []
    for name, col, lab in keys + [('exact (A = 0)', C['yellow'], 'exact score')]:
        handles += ax[1].plot(N, [max(r['excess'], 1e-7) for r in M[name]], color=col, marker='o', label=lab, zorder=3)
    ax[1].set(xscale='log', yscale='log', xlabel=r'bank size $N$', title='expected excess energy score')
    fig.legend(handles=handles, loc='outside lower center', ncol=4)
    fig.savefig(FIGS / 'plateau.pdf')
    plt.close(fig)

    def f(x):
        if x == 0:
            return '0'
        if x >= 100:
            return f'{x:,.0f}'.replace(',', '{,}')
        if x >= 10:
            return f'{x:.1f}'
        if x >= 0.1:
            return f'{x:.2f}'
        e = int(math.floor(math.log10(abs(x))))
        return f'${x / 10 ** e:.1f}\\!\\cdot\\!10^{{{e}}}$'
    ex, a30, a1k, un = [M[k] for k in ('exact (A = 0)', 'plateau A = 30', 'plateau A = 1000', 'unbounded s = v')]
    rows = []
    for i, r in enumerate(ex):
        Ns = f"{r['N']:,}".replace(',', '{,}')
        rows.append(f"  ${Ns}$ & {f(ex[i]['of'])} & {f(ex[i]['excess'])} & {f(a30[i]['of'])} & {f(a30[i]['of_lower'])} & {f(a30[i]['excess'])} & "
                    f"{f(a1k[i]['of'])} & {f(a1k[i]['of_lower'])} & {f(a1k[i]['excess'])} & {f(un[i]['of'])} & {f(un[i]['excess'])} \\\\")
    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# trained models
# ---------------------------------------------------------------------------

CURVES = [(('dot', 'twoway', 'kl'), C['red'], 'inner product'),
          (('tanh', 'twoway', 'kl'), C['orange'], r'$\mathrm{tanh}$ future'),
          (('ncos', 'twoway', 'kl'), C['aqua'], 'normalized cosine'),
          (('l2', 'joint', 'kl'), C['blue'], r'$L^2$ joint'),
          (('tk', 'oneway', 'kl'), C['violet'], r'Student-$t$ kernel'),
          (('tanh', 'twoway', 'hyb10'), C['green'], r'$\mathrm{tanh}$ + bank energy')]


def fig_sweep(runs, data='t3'):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9), constrained_layout=True, sharex=True)
    for j, enc in enumerate(('mlp', 'linear')):
        sizes = None
        for key, col, lab in CURVES:
            rs = runs.get((data, enc, key))
            if not rs:
                continue
            sizes = sorted({x['bank'] for x in rs[0]['rows']})
            ex = [ms([row_at(r, B)['excess_pct'] for r in rs])[0] for B in sizes]
            of = [ms([of_factor(r, B, 1) for r in rs])[0] for B in sizes]
            axes[0, j].plot(sizes, ex, color=col, marker='o', label=lab)
            axes[1, j].plot(sizes, [max(x, 1e-3) for x in of], color=col, marker='o', label=lab)
        if sizes is None:
            continue
        rs0 = next(rs for (d, e, _), rs in runs.items() if (d, e) == (data, enc))
        orc = ms([of_oracle(r, 1) for r in rs0])[0]
        if not math.isnan(orc) and orc > 0:
            axes[1, j].axhline(max(orc, 1e-3), color=MUTED, lw=0.8)
            axes[1, j].annotate('oracle', (sizes[0], max(orc, 1e-3)), xytext=(0, 3), textcoords='offset points', fontsize=7, color=MUTED)
        axes[1, j].plot(sizes, sizes, color=MUTED, lw=0.8)
        axes[1, j].annotate(r'ceiling $N$', (sizes[1], sizes[1]), xytext=(5, -2), textcoords='offset points', fontsize=7, color=MUTED)
        axes[0, j].set(xscale='log', yscale='log', title=f"{'MLP' if enc == 'mlp' else 'linear'} encoders")
        axes[1, j].set(xscale='log', yscale='log', xlabel=r'bank size $N$', ylim=(1e-3, 3e3))
    axes[0, 0].set_ylabel('energy score above oracle (%)')
    axes[1, 0].set_ylabel(r'over-forecast factor, $p=10^{-3}$')
    axes[0, 1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.savefig(FIGS / f'sweep_{data}.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_amplification():
    fs = sorted(Path('results/amplification').glob('*.json'))
    if not fs:
        return
    fig, axes = plt.subplots(1, 2 * 1, figsize=(6.6, 2.8), constrained_layout=True)
    r = json.load(open(fs[0]))
    la = r['log10_A']
    of = [max(x[1], 1e-4) for x in r['of_pop']]
    axes[0].plot([-6, 6], [1e-6, 1e6], color=MUTED, lw=0.8)
    axes[0].scatter([10 ** min(max(a, -6), 6) for a in la], of, s=7, color=C['blue'], alpha=0.6, linewidths=0)
    axes[0].set(xscale='log', yscale='log', xlim=(1e-3, 1e4), ylim=(1e-3, 1e4),
                xlabel=r'tail amplification $\bar A(u)$, ray probe', ylabel=r'over-forecast factor, $N=10^6$',
                title='one point per past')
    axes[0].plot([1e-3, 1e4], [1e-3, 1e4], color=MUTED, lw=0.8)
    sizes = r['sizes']
    for g, col in zip(r['groups'], SEQ[-len(r['groups']):]):
        y = [max(v, 1e-4) for v in g['of_by_N']['0.001']]
        axes[1].plot(sizes, y, color=col, marker='o')
        end_label(axes[1], sizes[-1], y[-1], rf"$\bar A\approx{g['A_geomean']:.2g}$")
    axes[1].plot(sizes, sizes, color=MUTED, lw=0.8)
    axes[1].set(xscale='log', yscale='log', xlabel=r'bank size $N$', ylim=(1e-3, 1e4), xlim=(80, 3e7),
                title=r'over-forecast factor, pasts grouped by $\bar A(u)$')
    fig.savefig(FIGS / 'amplification.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# generated table rows
# ---------------------------------------------------------------------------

SIZES = [100, 1000, 10000, 30000, 100000, 200000]
SYN_ROWS = [k for k, _ in MODELS]
LAB = dict(MODELS)
LAB.update({('dot', 'twoway', 'kl'): 'inner product', ('dot', 'twoway', 'hyb10'): 'inner product $+$ bank energy',
            ('tanh', 'twoway', 'hyb10'): '$\\tanh$ future $+$ bank energy',
            ('ncos', 'twoway', 'hyb10'): 'normalized cosine $+$ bank energy',
            ('l2', 'oneway', 'hyb10'): '$L^2$ one-way $+$ bank energy'})


def num(x, signed=True):
    if x != x:
        return '--'
    if abs(x) >= 1000:
        return f'${x:+,.0f}$'.replace(',', '{,}') if signed else f'{x:,.0f}'.replace(',', '{,}')
    if signed:
        return f'${x:+.1f}$'
    return f'{x:.2f}' if x < 10 else f'{x:.1f}' if x < 100 else f'{x:.0f}'


def rows_banksweep(runs, data, enc, keys, sizes=SIZES, bold_turn=True):
    out = []
    for key in keys:
        rs = runs.get((data, enc, key))
        if not rs:
            continue
        m = [ms([row_at(r, B)['excess_pct'] for r in rs if row_at(r, B)])[0] for B in sizes]
        i = min(range(len(m)), key=m.__getitem__)
        cells = [num(x) for x in m]
        if bold_turn and i < len(m) - 1 and m[-1] - m[i] > 0.5:       # a turnaround: bold the best bank size
            cells[i] = '$\\mathbf{' + cells[i].strip('$') + '}$'
        out.append(f'  {LAB[key]} & {len(rs)} & ' + ' & '.join(cells) + ' \\\\')
    return out


def rows_overforecast(runs, data, enc, keys, sizes=SIZES):
    out = []
    rs0 = None
    for key in keys:
        rs = runs.get((data, enc, key))
        if not rs:
            continue
        rs0 = rs0 or rs
        m = [ms([of_factor(r, B, 1) for r in rs if row_at(r, B)])[0] for B in sizes]
        pr = [r['ray_probe']['1000'] for r in rs]
        share = ms([p['frac_pasts_mean_amp_gt1'] for p in pr])[0]
        med = ms([p['log10_mean_amp_median'] for p in pr])[0]
        q90 = ms([p['log10_mean_amp_q90'] for p in pr])[0]
        clip = lambda x: f'${max(x, -99):.1f}$' if x > -99 else '$<-99$'
        out.append(f'  {LAB[key]} & ' + ' & '.join(num(x, signed=False) for x in m) + f' & {share:.2f} & {clip(med)} & {clip(q90)} \\\\')
    if rs0 and rs0[0].get('p_true_mean'):
        o = ms([of_oracle(r, 1) for r in rs0])[0]
        out.append('  \\midrule')
        out.append(f'  oracle forecast & \\multicolumn{{{len(sizes)}}}{{c}}{{{o:.3f}}} & & & \\\\')
    return out


def table_blocks(runs):
    b = {}
    kl = [k for k in SYN_ROWS if k[2] == 'kl']
    en = [k for k in SYN_ROWS if k[2] != 'kl']
    for enc in ('mlp', 'linear'):
        b[f'banksweep_{enc}'] = '\n'.join(rows_banksweep(runs, 't3', enc, kl) + ['  \\midrule'] + rows_banksweep(runs, 't3', enc, en))
        b[f'overforecast_{enc}'] = '\n'.join(rows_overforecast(runs, 't3', enc, kl + en))
    g = []
    for enc in ('mlp', 'linear'):
        for key in (('dot', 'twoway', 'kl'), ('tanh', 'twoway', 'kl')):
            rr = rows_banksweep(runs, 'tinf', enc, [key], bold_turn=False)
            g += [r.replace('  ', f"  {'MLP' if enc == 'mlp' else 'linear'}, ", 1) for r in rr]
    b['gaussian'] = '\n'.join(g)
    real_keys = [k for k in SYN_ROWS if k not in (('tk', 'joint', 'kl'), ('tk2', 'joint', 'kl'), ('dot', 'twoway', 'es'), ('tanh', 'twoway', 'es'),
                                                    ('ncos', 'twoway', 'hyb10'), ('l2', 'oneway', 'hyb10'))]
    for data in ('btc', 'rain', 'flow'):
        rows = []
        for enc in ('mlp', 'linear'):
            any_rs = [rs for (d, e, _), rs in runs.items() if (d, e) == (data, enc)]
            if not any_rs:
                continue
            Bmax = max(x['bank'] for x in any_rs[0][0]['rows'])
            sizes = [1000, 10000, Bmax]
            for key in real_keys:
                rs = runs.get((data, enc, key))
                if not rs:
                    continue
                ex = [ms([row_at(r, B)['excess_pct'] for r in rs])[0] for B in sizes]
                of = [ms([of_factor(r, B, 1) for r in rs])[0] for B in (1000, Bmax)]
                share = ms([r['ray_probe']['1000']['frac_pasts_mean_amp_gt1'] for r in rs])[0]
                rows.append(f"  {'MLP' if enc == 'mlp' else 'linear'} & {LAB[key]} & " + ' & '.join(num(x) for x in ex) + ' & '
                            + ' & '.join(num(x, signed=False) for x in of) + f' & {share:.2f} \\\\')
            if enc == 'mlp':
                rows.append('  \\midrule')
        b[f'real_{data}'] = '\n'.join(rows)
    return b


def text_blocks(runs):
    b = {}
    fs = sorted(Path('results/amplification').glob('*.json'))
    if fs:
        r = json.load(open(fs[0]))
        b['spearman'] = f"${r['spearman_logA_vs_OF']['0.001']:.2f}$ ($p=10^{{-3}}$) and ${r['spearman_logA_vs_OF']['0.0001']:.2f}$ ($p=10^{{-4}}$)"
    parts = []
    rs = runs.get(('t3', 'mlp', ('tk', 'oneway', 'kl')))
    if rs:
        parts.append(f"$b\\approx{ms([r['beta'] for r in rs])[0]:.0f}$ for the one-way Student-$t$ kernel with MLP encoders")
    rs = runs.get(('t3', 'linear', ('tk2', 'oneway', 'kl')))
    if rs:
        parts.append(f"$b-b_v\\approx{ms([r['beta'] - r['beta_uv'][1] for r in rs])[0]:.0f}$ for the one-way log-quadratic family with linear encoders")
    if parts:
        b['exponents'] = '(' + ' and '.join(parts) + ', against $d_u=50$),'
    return b


def fill(tex, name, body):
    pat = re.compile(rf'(%%BEGIN:{name}\n).*?(%%END:{name})', re.S)
    if not pat.search(tex):
        print(f'(no block {name} in main.tex)')
        return tex
    return pat.sub(lambda m: m.group(1) + body + '\n' + m.group(2), tex)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    tex = (NOTE / 'main.tex').read_text()
    tex = fill(tex, 'plateau', fig_plateau())
    runs = load()
    for name, body in table_blocks(runs).items():
        tex = fill(tex, name, body)
    for name, body in text_blocks(runs).items():
        tex = fill(tex, name, body)
    fig_sweep(runs)
    for data in ('btc', 'rain', 'flow'):
        if any(d == data for d, _, _ in runs):
            fig_sweep(runs, data)
    fig_amplification()
    (NOTE / 'main.tex').write_text(tex)


if __name__ == '__main__':
    main()
