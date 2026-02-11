import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ==========================================
# 1. 全局设置：全面加大字号提升清晰度
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral']
# 将基础字号从 14 提升到 16
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.5 # 边框也稍微加粗一点匹配大字体

def draw_evolution_chart_large_font():
    # 数据
    labels = ['Baseline', 'w/ AR', 'TopoChunker']
    scores = [0.52, 0.71, 0.90]
    
    # 画布设置 (适当增加一点画布高度以容纳大字体)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)

    # 配色
    color_stage1 = '#55BC89' 
    color_stage2 = '#F3B25F' 
    color_stage3 = '#E64E53'
    colors = [color_stage1, color_stage2, color_stage3]

    # 绘制柱状图
    width = 0.5
    ax.bar(labels, scores, width=width, color=colors, 
           edgecolor='black', linewidth=1.0, alpha=0.9, zorder=2)

    # 绘制折线趋势
    # 加大 markersize 让标记更清晰
    ax.plot(labels, scores, color='black', linewidth=2.5, linestyle='-', 
            marker='^', markersize=12, markerfacecolor='black', zorder=3)

    # ==========================================
    # 🏷️ 数值标注 (加大字号)
    # ==========================================
    for i, v in enumerate(scores):
        # fontsize 从 14 提升到 17
        ax.text(i, v + 0.03, f"{v:.2f}", ha='center', va='bottom', 
                fontsize=17, fontweight='bold', color='black')

    # ==========================================
    # 📐 坐标轴与排版 (加大字号)
    # ==========================================
    ax.set_ylim(0, 1.15) # 稍微增加顶部空间给大标签
    # Y轴标题加大到 18
    ax.set_ylabel('Fidelity Score (TED Metric)', fontsize=18, fontweight='bold', labelpad=10)
    
    ax.grid(axis='y', linestyle='--', color='gray', alpha=0.4, zorder=0)
    
    # 柱子内部的阶段说明文字
    stage_texts = ["Stage I:\nNode Id.", "Stage II:\n+ Anaphora", "Stage III:\n+ Hierarchy"]
    for i, (text, col) in enumerate(zip(stage_texts, colors)):
        # fontsize 从 11 提升到 14
        ax.text(i, 0.1, text, ha='center', va='center', color='black', 
                fontsize=14, fontweight='bold', zorder=4, linespacing=1.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 刻度标签字号加大
    ax.tick_params(axis='x', labelsize=16, length=0, pad=8) # X轴标签加大
    ax.tick_params(axis='y', labelsize=15, pad=5)           # Y轴刻度加大

    # ==========================================
    # 🧩 图例 (加大字号)
    # ==========================================
    legend_bar = mpatches.Patch(color='gray', label='Fidelity Score (Bar)')
    legend_line = Line2D([0], [0], color='black', marker='^', markersize=10, label='Performance Trend')

    # fontsize 从 12 提升到 15
    ax.legend(handles=[legend_bar, legend_line], 
              loc='upper left', frameon=False, fontsize=15)

    plt.tight_layout()
    plt.savefig('Chart_Evolution_LargeFont.png', dpi=300)
    plt.savefig('Chart_Evolution_LargeFont.pdf', format='pdf')
    print("图片已生成：Chart_Evolution_LargeFont.png (大字体版)")
    plt.show()

if __name__ == "__main__":
    draw_evolution_chart_large_font()