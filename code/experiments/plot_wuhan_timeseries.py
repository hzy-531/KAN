#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""武汉 观测 y vs 预测 y_hat 全时序图 (单图, 用于与 fig4 并排)。

- 无封城竖线/标注, 改为横轴普通刻度 01-23 (其余刻度取月初)。
- 无 test 阴影。
- 观测(灰) / 预测(蓝) 两条曲线。

输出: result/paper_figures_20seed/visualizations/prediction/wuhan_timeseries_Full_KAN_Graphormer.{pdf,png,svg}
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
import pandas as pd
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

OBS_C = '#333333'
PRED_C = '#1f77b4'

# 横轴刻度 (月初 + 01-23)
TICK_DATES = ['2019-11-01', '2019-12-01', '2020-01-01', '2020-01-23',
              '2020-02-01', '2020-03-01', '2020-03-15']


def main():
    os.makedirs(OUT, exist_ok=True)
    pkl = os.path.join(BASE, f'seed_{SEED}', 'results', 'Full_KAN_Graphormer_result.pkl')
    with open(pkl, 'rb') as f:
        r = pickle.load(f)

    fp = r['full_pred']
    true = np.array(fp['true_total'])[:, WUHAN_IDX]
    pred = np.array(fp['pred_total'])[:, WUHAN_IDX]
    dates = list(fp['dates'])
    start = pd.Timestamp('2019-11-01')

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(dates))

    ax.plot(x, true, '-', color=OBS_C, lw=1.8, label='Observed $y$', zorder=3)
    ax.plot(x, pred, '-', color=PRED_C, lw=1.8, label='Predicted $\\hat{y}$', zorder=4)

    # 横轴刻度: 月初 + 01-23
    tick_idx = [(pd.Timestamp(d) - start).days for d in TICK_DATES]
    tick_lab = [pd.Timestamp(d).strftime('%m-%d') for d in TICK_DATES]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_lab, fontsize=10)

    ax.set_ylabel('daily cases', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

    ax.set_facecolor('white')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('black')
        ax.spines[sp].set_linewidth(0.9)
    ax.tick_params(labelsize=9.5)

    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=1.0)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT, f'wuhan_timeseries_Full_KAN_Graphormer.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'武汉: obs峰={true.max():.0f}, pred峰={pred.max():.0f}, '
          f'总量 obs={true.sum():.0f} pred={pred.sum():.0f}')
    print('Done. 输出:', os.path.join(OUT, 'wuhan_timeseries_Full_KAN_Graphormer.{pdf,png,svg}'))


if __name__ == '__main__':
    main()
