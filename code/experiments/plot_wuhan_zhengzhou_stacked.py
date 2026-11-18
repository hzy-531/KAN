#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""武汉(上) + 郑州(下) 上下两张子图的 y vs y_hat 全时序图 (与单城图完全同款样式)。

- 无封城竖线/标注, 横轴普通刻度 (月初 + 01-23)。
- 无 test 阴影。
- 观测(灰) / 预测(蓝)。

输出: result/paper_figures_20seed/visualizations/prediction/wuhan_zhengzhou_timeseries_Full_KAN_Graphormer.{pdf,png,svg}
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

# 城市索引 (0-23): 武汉=8, 郑州=13
PANELS = [(8, 'Wuhan'), (13, 'Zhengzhou')]

# 横轴刻度: 均匀等分 (间隔 20 天), 01-23 (封城日) 作为普通刻度落在其中, 不单独高亮
TICK_STEP = 20

OBS_C = '#333333'
PRED_C = '#1f77b4'


def main():
    os.makedirs(OUT, exist_ok=True)
    pkl = os.path.join(BASE, f'seed_{SEED}', 'results', 'Full_KAN_Graphormer_result.pkl')
    with open(pkl, 'rb') as f:
        r = pickle.load(f)

    fp = r['full_pred']
    dates = list(fp['dates'])
    start = pd.Timestamp('2019-11-01')
    lockdown_day = (pd.Timestamp('2020-01-23') - start).days  # day 83
    first_day = lockdown_day % TICK_STEP  # 3, 使 01-23 落在均匀刻度上
    tick_idx = list(range(first_day, len(dates), TICK_STEP))
    tick_lab = [pd.Timestamp(dates[i]).strftime('%m-%d') for i in tick_idx]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0))
    x = np.arange(len(dates))

    for ax, (idx, cname) in zip(axes, PANELS):
        true = np.array(fp['true_total'])[:, idx]
        pred = np.array(fp['pred_total'])[:, idx]

        ax.plot(x, true, '-', color=OBS_C, lw=1.8, label='Observed $y$', zorder=3)
        ax.plot(x, pred, '-', color=PRED_C, lw=1.8, label='Predicted $\\hat{y}$', zorder=4)

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_lab, fontsize=9)
        ax.set_ylabel('daily cases', fontsize=12)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax.text(0.02, 0.97, cname, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top', ha='left')

        ax.set_facecolor('white')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color('black')
            ax.spines[sp].set_linewidth(0.9)
        ax.tick_params(labelsize=9.5)

    axes[1].set_xlabel(r'date $t$', fontsize=12)

    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=1.0)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(os.path.join(OUT, f'wuhan_zhengzhou_timeseries_Full_KAN_Graphormer.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print('Done. 输出:', os.path.join(OUT, 'wuhan_zhengzhou_timeseries_Full_KAN_Graphormer.{pdf,png,svg}'))


if __name__ == '__main__':
    main()
