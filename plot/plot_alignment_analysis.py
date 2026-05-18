import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib

matplotlib.use("TkAgg")

# =========================
# Data
# =========================
methods_top = [
    "SW", "NW", "Dedal",
    "pLM-BLAST_ESM2-3B", "pLM-BLAST_ProtT5", "pLM-BLAST_ProstT5",
    "PEBA_ESM2-3B", "PEBA_ProtT5", "PEBA_ProstT5",
    "EBA_ESM2-3B", "EBA_ProtT5", "EBA_ProstT5",
    "SEBA_ESM2-3B", "SEBA_ProtT5", "SEBA_ProstT5"
]

malisam = [0.0161, 0.0563, 0.0703, 0.0115, 0.1697, 0.2268,
           0.0444, 0.1469, 0.2442, 0.0981, 0.1483, 0.2467,
           0.1374, 0.1917, 0.2893]

malidup = [0.2648, 0.3299, 0.4166, 0.2323, 0.5291, 0.5942,
           0.3300, 0.5583, 0.6529, 0.5610, 0.5500, 0.6254,
           0.6220, 0.6159, 0.7003]

sabmark = [0.3079, 0.3255, 0.2556, 0.2570, 0.4295, 0.4614,
           0.3671, 0.4756, 0.4953, 0.4734, 0.4606, 0.4787,
           0.5037, 0.4946, 0.5135]

balibase_methods = ["vcMSA", "DEDAL", "BLOSUM62", "PEbA_ESM2", "PEbA_ProtT5", "SEBA"]
balibase_subsets = ["RV11", "RV12", "RV911", "RV912", "RV913"]
balibase = np.array([
    [0.559, 0.828, 0.437, 0.685, 0.922],
    [0.413, 0.648, 0.092, 0.377, 0.203],
    [0.220, 0.626, 0.276, 0.594, 0.874],
    [0.336, 0.715, 0.242, 0.633, 0.900],
    [0.590, 0.844, 0.461, 0.755, 0.940],
    [0.650, 0.870, 0.481, 0.769, 0.944],
])

homstrad_methods = ["MMseqs2", "CLE-SW", "EBA-plain", "EBA", "SEBA",
                    "Foldseek", "Foldseek-TM", "TMalign", "DALI"]
homstrad_f1 = [0.421, 0.776, 0.810, 0.860, 0.881, 0.850, 0.878, 0.883, 0.885]

# =========================
# Style settings
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

gray = "#D9D9D9"
gray_edge = "#5F5F5F"
light_blue = "#83AEEB"
mid_blue = "#5B8DDF"
navy = "#003B8E"
grid_color = "#C7C7C7"
gray_deep="#5F5F5F"

# =========================
# Helper functions
# =========================
def style_axes(ax):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color("black")
    ax.tick_params(width=0.8, length=3, color="black")
    ax.set_facecolor("white")


def draw_top_panel(ax, values, title, show_ylabels=False):
    y = np.arange(len(methods_top))
    colors = [gray] * 12 + [gray, gray, gray_deep]

    ax.barh(
        y, values,
        height=0.58,
        color=colors,
        edgecolor=gray_edge,
        linewidth=0.6
    )

    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_xlim(0, 0.8)
    ax.set_xticks([0, 0.16, 0.32, 0.48, 0.64, 0.80])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("F1-score")
    ax.grid(axis="x", linestyle=":", linewidth=0.7, color=grid_color, alpha=0.75)
    ax.set_axisbelow(True)

    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(methods_top)
    else:
        ax.set_yticklabels([""] * len(methods_top))
        ax.tick_params(axis="y", length=0)

    ax.invert_yaxis()

    # Group separators
    for sep in [2.5, 5.5, 8.5, 11.5]:
        ax.axhline(sep, color="#8A8A8A", linestyle=(0, (4, 2)), linewidth=0.85, zorder=0)

    # Value labels
    for i, v in enumerate(values):
        label_weight = "bold" if i == len(values) - 1 else "normal"
        ax.text(
            v + 0.012, i, f"{v:.4f}",
            va="center", ha="left",
            fontsize=8.2,
            fontweight=label_weight,
            clip_on=False
        )

    style_axes(ax)


def draw_balibase_panel(ax):
    x = np.arange(len(balibase_subsets))
    width = 0.12
    colors = [gray, "#C2C2C2", "#AFAFAF", "#A6A6A6", "#8C8C8C", gray_deep]

    for i, method in enumerate(balibase_methods):
        offset = (i - (len(balibase_methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            balibase[i],
            width=width,
            label=method,
            color=colors[i],
            edgecolor=gray_edge,
            linewidth=0.55
        )

        for bar, value in zip(bars, balibase[i]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
                rotation=90
            )

    ax.set_title("(D) BAliBASE", fontweight="bold", pad=6)
    ax.set_ylabel("SP-score")
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.linspace(0, 1.1, 6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xticks(x)
    ax.set_xticklabels(balibase_subsets)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color=grid_color, alpha=0.75)
    ax.set_axisbelow(True)

    ax.legend(
        ncol=6,
        loc="upper left",
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
        borderaxespad=0.4
    )

    style_axes(ax)


def draw_homstrad_panel(ax):
    y = np.arange(len(homstrad_methods))
    colors = [gray] * len(homstrad_methods)
    colors[homstrad_methods.index("SEBA")] = gray_deep
    colors[homstrad_methods.index("DALI")] = gray_deep

    ax.barh(
        y,
        homstrad_f1,
        height=0.56,
        color=colors,
        edgecolor=gray_edge,
        linewidth=0.6
    )

    ax.set_title("(E) HOMSTRAD", fontweight="bold", pad=6)
    ax.set_xlim(0, 1.00)
    ax.set_xticks(np.linspace(0, 1.0, 6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("F1-score")
    ax.set_yticks(y)
    ax.set_yticklabels(homstrad_methods)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, color=grid_color, alpha=0.75)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    # 在 SEBA 和 Foldseek 之间添加虚线
    ax.axhline(4.5, color="#8A8A8A", linestyle=(0, (4, 2)), linewidth=0.85, zorder=0)

    for i, v in enumerate(homstrad_f1):
        label_weight = "bold" if homstrad_methods[i] == "SEBA" else "normal"
        ax.text(
            v + 0.015, i, f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=8.4,
            fontweight=label_weight,
            clip_on=False
        )

    style_axes(ax)


# =========================
# Figure layout
# =========================
fig = plt.figure(figsize=(16, 10), dpi=150)
gs = fig.add_gridspec(
    nrows=2,
    ncols=38,
    height_ratios=[0.95, 1.25],
    wspace=0.33,
    hspace=0.28
)

ax_a = fig.add_subplot(gs[0, 0:12])
ax_b = fig.add_subplot(gs[0, 13:25])
ax_c = fig.add_subplot(gs[0, 26:38])
ax_d = fig.add_subplot(gs[1, 0:24])
ax_e = fig.add_subplot(gs[1, 27:38])

draw_top_panel(ax_a, malisam, "(A) MALISAM", show_ylabels=True)
draw_top_panel(ax_b, malidup, "(B) MALIDUP", show_ylabels=False)
draw_top_panel(ax_c, sabmark, "(C) SABMark", show_ylabels=False)
draw_balibase_panel(ax_d)
draw_homstrad_panel(ax_e)

fig.subplots_adjust(left=0.13, right=0.985, top=0.95, bottom=0.08)

# =========================
# Save and show
# =========================
# plt.savefig("seba_five_panel_figure.png", dpi=600, bbox_inches="tight", facecolor="white")
# plt.savefig("seba_five_panel_figure.svg", bbox_inches="tight", facecolor="white")
plt.show()