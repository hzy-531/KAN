#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""双联图: (a) 预测 vs 观测散点 (fig:scatter 左) + (b) 武汉 观测 y vs 预测 y_hat 全时序 (右)。

选武汉为典型城: 全季峰值最大 (1104), 有 01-23 封城, 预测几乎重合 (峰值 1104 vs 1077,
总量 40464 vs 40438), 且同时能衔接后面的封城/参数讨论。

(a) 数据 = seed_0 测试集 predictions/targets (26×24 摊平); 标注 20 种子 R²/RMSE/MAE。
(b) 数据 = seed_0 full_pred 的 true_total/pred_total 武汉市 (全 136 天), 测试期(末26天)浅灰阴影。

输出: result/paper_figures_20seed/visualizations/prediction/scatter_wuhan_timeseries_Full_KAN_Graphormer.{pdf,png,svg}
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import __main__
from exp_lib import models, layers, physics, loss, data

_COMPAT_CLASSES = [
    ('EnhancedFull_Graphormer_V3', models.EnhancedFull_Graphormer_V3),
    ('EnhancedMLP_Baseline_V3', models.EnhancedMLP_Baseline_V3),
    ('EnhancedLSTM_Baseline_V3', models.EnhancedLSTM_Baseline_V3),
    ('EnhancedGAT_Baseline_V3', models.EnhancedGAT_Baseline_V3),
    ('EnhancedKAN_Only_V3', models.EnhancedKAN_Only_V3),
    ('M_Graphormer_Baseline', models.M_Graphormer_Baseline),
    ('NoPhysics_KAN_Graphormer', models.NoPhysics_KAN_Graphormer),
    ('RobustKANLinear', layers.RobustKANLinear),
    ('StandardMLP', layers.StandardMLP),
    ('GatedTCN', layers.GatedTCN),
    ('RBFSpatialEncoding', layers.RBFSpatialEncoding),
    ('GraphormerLayer', layers.GraphormerLayer),
    ('ImprovedGATLayer', layers.ImprovedGATLayer),
    ('DataCollector', data.DataCollector),
    ('CityLevelLogMinMaxScaler', data.CityLevelLogMinMaxScaler),
    ('CityLevelLinearMinMaxScaler', data.CityLevelLinearMinMaxScaler),
    ('TemporalFeatureEnhancer', data.TemporalFeatureEnhancer),
    ('ImprovedTemporalDataset', data.ImprovedTemporalDataset),
    ('CurriculumScientificLoss', loss.CurriculumScientificLoss),
]
for _name, _cls in _COMPAT_CLASSES:
    setattr(__main__, _name, _cls)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

BASE = r'E:\Claude code\KAN+\result\ablation_20seeds_6models_20260915_200323'
OUT = r'E:\Claude code\KAN+\result\paper_figures_20seed\visualizations\prediction'
SEED = 0
WUHAN_IDX = 8
LOCKDOWN_DAY = 83
TEST_START = 108   # 测试期起点 (末 26 天)

METRICS_20SEED = (0.744, 0.002, 3.503, 0.015, 1.901, 0.012)

OBS_C = '#333333'
PRED_C = '#1f77b4'
LOCK_C = '#c0392b'
TEST_C = '#e8e8e8'


def main():
    os.makedirs(OUT, exist_ok=True)
    pkl = os.path.join(BASE, f'seed_{SEED}', 'results', 'Full_KAN_Graphormer_result.pkl')
    with open(pkl, 'rb') as f:
        r = pickle.load(f)

    ev = r['eval_data']['evaluation_results']
    pred = np.array(ev['predictions']).flatten()
    targ = np.array(ev['targets']).flatten()

    fp = r['full_pred']
    true = np.array(fp['true_total'])[:, WUHAN_IDX]
    predc = np.array(fp['pred_total'])[:, WUHAN_IDX]
    dates = list(fp['dates'])
    dates = [pd.Timestamp(d) for d in dates] if not hasattr(dates[0], 'strftime') else dates

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    # ---- (a) 散点 ----
    ax = axes[0]
    ax.scatter(targ, pred, s=16, alpha=0.55, color=PRED_C, edgecolors='none')
    lo = min(targ.min(), pred.min()); hi = max(targ.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.6, alpha=0.75)
    r2, r2s, rmse, rmses, mae, maes = METRICS_20SEED
    ax.text(0.04, 0.96,
            f'$R^2$ = {r2:.3f} $\\pm$ {r2s:.3f}\n'
            f'RMSE = {rmse:.3f} $\\pm$ {rmses:.3f}\n'
            f'MAE = {mae:.3f} $\\pm$ {maes:.3f}',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='#cccccc'))
    ax.set_xlabel(r'Observed cases $y$', fontsize=12)
    ax.set_ylabel(r'Predicted cases $\hat{y}$', fontsize=12)
    ax.text(0.03, 0.03, '(a)', transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='bottom', ha='left')

    # ---- (b) 武汉 时序 ----
    ax = axes[1]
    x = np.arange(len(dates))
    ax.axvspan(TEST_START, len(dates) - 1, color=TEST_C, zorder=0)
    ax.plot(x, true, '-', color=OBS_C, lw=1.7, label='Observed $y$', zorder=3)
    ax.plot(x, predc, '-', color=PRED_C, lw=1.7, label='Predicted $\\hat{y}$', zorder=4)
    ax.axvline(LOCKDOWN_DAY, color=LOCK_C, ls='--', lw=1.2, alpha=0.85, zorder=2)
    ax.text(LOCKDOWN_DAY + 1, ax.get_ylim()[1] * 0.90 if False else 0.95,
            'lockdown\n(Jan 23)', transform=ax.get_xaxis_transform(),
            fontsize=8, color=LOCK_C, va='top', ha='left')
    ax.text(0.5 * (TEST_START + len(dates) - 1), 0.03, 'test',
            transform=ax.transData, fontsize=8, color='#666666', va='bottom', ha='center')
    ax.set_xlabel('date $t$', fontsize=12)
    ax.set_ylabel('daily cases', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.text(0.03, 0.97, '(b) Wuhan', transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')

    # 统一白底 + 左/下黑轴
    for ax in axes:
        ax.set_facecolor('white')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color('black')
            ax.spines[sp].set_linewidth(0.9)
        ax.tick_params(labelsize=9.5)

    # 时间轴日期刻度
    ax = axes[1]
    n_tk = 6
    tk = np.linspace(0, len(dates) - 1, n_tk, dtype=int)
    ax.set_xticks(x[tk])
    ax.set_xticklabels([dates[t].strftime('%m-%d') for t in tk], fontsize=9)

    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=1.0)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT, f'scatter_wuhan_timeseries_Full_KAN_Graphormer.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'武汉: obs峰={true.max():.0f} (day {int(true.argmax())}), pred峰={predc.max():.0f}')
    print(f'      总量 obs={true.sum():.0f}, pred={predc.sum():.0f}')
    print(f'      测试段 corr={np.corrcoef(true[TEST_START:], predc[TEST_START:])[0,1]:.3f}')
    print('Done. 输出:', os.path.join(OUT, 'scatter_wuhan_timeseries_Full_KAN_Graphormer.{pdf,png,svg}'))


if __name__ == '__main__':
    import pandas as pd
    main()
