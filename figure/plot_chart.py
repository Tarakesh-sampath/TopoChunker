import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def plot_vertical_bar_with_top_left_box():
    labels_short = ["Lumber", "Topo-S", "Topo-C", "Topo-A"]
    values = [16500, 5000, 18800, 10520]

    C_GREEN  = "#8AD46A"
    C_BLUE   = "#43B5E6"
    C_ORANGE = "#F1A77F"
    C_GRAY   = "#6E6E6E"
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_GRAY]

    # 图稍微加宽 + 加高一点点
    FIG_W, FIG_H = 3.2, 2.85
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)

    # =========================
    # ✅ 更细的柱子 + 更小的间距
    # =========================
    gap = 0.55      # 原 0.70 -> 更小
    bar_w = 0.28    # 原 0.40 -> 更细

    x_pos = [i * gap for i in range(len(labels_short))]

    ax.bar(
        x_pos, values,
        width=bar_w,
        color=colors,
        edgecolor="#4A4A4A",
        linewidth=0.5,
        zorder=3
    )

    # 柱顶数字
    y_offset = max(values) * 0.02
    for x, v in zip(x_pos, values):
        ax.text(
            x, v + y_offset, f"{v}",
            ha="center", va="bottom",
            fontsize=8.5, color="black"
        )

    ax.set_ylabel("Total LLM Tokens")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short)

    # ✅ 关键：缩小间距后，手动给左右留一点边距，别贴边
    ax.set_xlim(min(x_pos) - 0.35, max(x_pos) + 0.35)

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, color="#D0D0D0", zorder=0)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#666666")
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(axis="both", length=3, width=0.8)

    # 顶部留更多空白
    ax.set_ylim(0, max(values) * 1.40)

    # 说明框
    legend_items = [
        mpatches.Patch(color="none", label="S: Simple Doc"),
        mpatches.Patch(color="none", label="C: Complex-Max"),
        mpatches.Patch(color="none", label="A: Avg. Mixed"),
    ]

    leg = ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.02),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.35,
        handlelength=0,
        handletextpad=0.0,
        fontsize=8.2
    )
    leg.get_frame().set_edgecolor("#666666")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_facecolor("white")

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, bottom=0.20)

    plt.savefig("token_efficiency_bar_top_left_box.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.savefig("token_efficiency_bar_top_left_box.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_vertical_bar_with_top_left_box()
