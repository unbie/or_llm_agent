import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Academic paper style ────────────────────────────────────────────────────
matplotlib.rcParams.update({
    # Font — Times New Roman for a journal look
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',

    # Axes
    'axes.linewidth':     1.0,
    'axes.edgecolor':     'black',
    'axes.facecolor':     'white',
    'axes.grid':          True,
    'axes.grid.axis':     'y',
    'grid.linestyle':     '--',
    'grid.linewidth':     0.5,
    'grid.color':         '#CCCCCC',
    'grid.alpha':         0.8,

    # Tick marks
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.labelsize':    9,
    'ytick.labelsize':    10,

    # Legend
    'legend.frameon':     True,
    'legend.framealpha':  1.0,
    'legend.edgecolor':   'black',
    'legend.fontsize':    11,

    # Figure
    'figure.facecolor':   'white',
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
})

# ── Load data ────────────────────────────────────────────────────────────────
file_path = 'd:/pythonProject/or_llm_agent/experiments_compare/figures/cost_by_instance.csv'
df = pd.read_csv(file_path)

instances = df['instance'].tolist()
x = np.arange(len(instances))

# Rename ALNS to LLM-ALNS in data mapping to avoid KeyError and rename the labels
df = df.rename(columns={'ALNS': 'LLM-ALNS'})

# ── Colour palette (print-friendly, colorblind-safe, clean solid colors) ──────
COLORS  = {'ACO': '#4878CF', 'LLM-ALNS': '#6ACC65', 'GA': '#D65F5F'}
MARKERS = {'ACO': 'o',       'LLM-ALNS': 's',        'GA': '^'}
LINES   = {'ACO': '-',       'LLM-ALNS': '--',       'GA': '-.'}

# ════════════════════════════════════════════════════════════════════════════
#  Figure 1 – Grouped bar chart (All 30 instances together in a single row)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 6))
w = 0.26

for k, algo in enumerate(['ACO', 'LLM-ALNS', 'GA']):
    ax.bar(x + (k - 1) * w, df[algo], w,
           label=algo,
           color=COLORS[algo],
           edgecolor='black',
           linewidth=0.7,
           alpha=0.90)

ax.set_xticks(x)
ax.set_xticklabels(instances, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Total Cost', fontsize=12)
ax.set_xlabel('Instance', fontsize=12)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e3:.0f}k'))

# Place legend at upper left to avoid covering RC Instances label on the upper right
ax.legend(loc='upper left', ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add vertical dividers for classes in bar chart as well
c_count  = len([i for i in instances if i.startswith('c')])
r_count  = len([i for i in instances if i.startswith('r') and not i.startswith('rc')])
ax.axvline(c_count - 0.5,  color='gray', linestyle=':', linewidth=1.0)
ax.axvline(c_count + r_count - 0.5, color='gray', linestyle=':', linewidth=1.0)

# Label regions
def add_region_label(ax_obj, xstart, xend, label, y_frac=0.95):
    ax_obj.text((xstart + xend) / 2, ax_obj.get_ylim()[1] * y_frac,
                label, ha='center', va='top', fontsize=11,
                color='black', style='italic', fontweight='bold')

# Increase y-limit overhead significantly to prevent overlap with RC Instances
ylo, yhi = ax.get_ylim()
ax.set_ylim(ylo, yhi * 1.15)
add_region_label(ax, 0,        c_count - 0.5, 'C Instances', 0.98)
add_region_label(ax, c_count,  c_count + r_count - 0.5, 'R Instances', 0.98)
add_region_label(ax, c_count + r_count, len(instances) - 1, 'RC Instances', 0.98)

ax.set_title(
    'Figure 1: Total Cost Comparison Across Solomon Benchmark Instances (ACO vs. LLM-ALNS vs. GA)',
    fontsize=13, fontweight='bold', pad=15)

out_bar = 'd:/pythonProject/or_llm_agent/experiments_compare/figures/cost_comparison_bar.png'
fig.savefig(out_bar, dpi=300, bbox_inches='tight', facecolor='white')
print(f'[OK] Bar chart saved → {out_bar}')
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
#  Figure 2 – Line chart (all 30 instances, separated by instance class)
# ════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(15, 6))

for algo in ['ACO', 'LLM-ALNS', 'GA']:
    ax2.plot(x, df[algo],
             marker=MARKERS[algo],
             linestyle=LINES[algo],
             color=COLORS[algo],
             linewidth=1.5,
             markersize=5,
             label=algo)

# Draw vertical dividers between instance classes
ax2.axvline(c_count - 0.5,  color='gray', linestyle=':', linewidth=1.0)
ax2.axvline(c_count + r_count - 0.5, color='gray', linestyle=':', linewidth=1.0)

# Annotate class regions with more space
ylo2, yhi2 = ax2.get_ylim()
ax2.set_ylim(ylo2, yhi2 * 1.15)
add_region_label(ax2, 0,        c_count - 0.5, 'C Instances', 0.98)
add_region_label(ax2, c_count,  c_count + r_count - 0.5, 'R Instances', 0.98)
add_region_label(ax2, c_count + r_count, len(instances) - 1, 'RC Instances', 0.98)

ax2.set_xticks(x)
ax2.set_xticklabels(instances, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Total Cost', fontsize=12)
ax2.set_xlabel('Instance', fontsize=12)
ax2.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e3:.0f}k'))
ax2.legend(loc='upper left', ncol=3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.set_title(
    'Figure 2: Total Cost Trend Across Solomon Benchmark Instances (ACO vs. LLM-ALNS vs. GA)',
    fontsize=13, fontweight='bold', pad=15)

out_line = 'd:/pythonProject/or_llm_agent/experiments_compare/figures/cost_comparison_line.png'
fig2.savefig(out_line, dpi=300, bbox_inches='tight', facecolor='white')
print(f'[OK] Line chart saved → {out_line}')
plt.close(fig2)
