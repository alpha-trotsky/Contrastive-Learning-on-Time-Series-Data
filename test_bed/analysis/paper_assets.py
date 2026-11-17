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

from analysis.summarize_tail_sweep import load, ms, row_at, of_factor, of_oracle

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


def fill(tex, name, body):
    pat = re.compile(rf'(%%BEGIN:{name}\n).*?(%%END:{name})', re.S)
    assert pat.search(tex), name
    return pat.sub(lambda m: m.group(1) + body + '\n' + m.group(2), tex)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    tex = (NOTE / 'main.tex').read_text()
    tex = fill(tex, 'plateau', fig_plateau())
    runs = load()
    fig_sweep(runs)
    fig_amplification()
    (NOTE / 'main.tex').write_text(tex)


if __name__ == '__main__':
    main()
