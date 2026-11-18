#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""「特殊城市」病例时间序列 + 参数演化 联动图。

把结果里两类特殊城市挑出来:
  (1) R^2 低/负的城市 (Tianjin, Qingdao, Jinan, Zhengzhou, Changchun, Harbin, Beijing, Hangzhou, Xi'an)
  (2) 参数曲线特殊的城市 (Wuhan 封城 c/Rt 骤降; Harbin Rt<1; Shijiazhuang Rt 先降后反弹)

每个城市一行, 上下两个子图共享时间轴:
  上: 每日新增病例 (灰色面积), 标封城竖线 + 阴影标注测试期(末 26 天)
  下: β / c (左轴) + Rt (右轴, Rt=1 虚线), 标封城竖线
  标题: 城市名 + R^2 + RMSE (20 种子均值)

输出: result/paper_figures_20seed/visualizations/interpretation/special_cities_cases_params.{pdf,png,svg}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

from explore_city_interpretation import (
    load_daily_cases, load_region, load_per_city_r2_rmse,
    load_param_mean, CITIES, _CITY_EN)

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

OUT = r'E:\Claude code\KAN+\result\paper_figures_20seed\visualizations\interpretation'
LOCKDOWN_DAY = 83     # 2020-01-23
TEST_START = 108      # 测试期起点 (末 26 天, 2020-02-17)
TEST_END = 133
TRIM_START = 7        # 参数有效段 (去掉 lookback 学习期)
TRIM_END = 133

# 颜色
CASE_C = '#333333'
BETA_C = '#1f77b4'
CONTACT_C = '#9467bd'
RT_C = '#ff7f0e'
RT1_C = '#999999'
TEST_C = '#f2f2f2'

# 特殊城市 (按叙事分组排序)
SPECIAL = [
    '武汉市', '哈尔滨市', '石家庄市',     # 参数曲线特殊
    '天津市', '青岛市', '济南市', '郑州市', '长春市',  # R^2 负 (北方低发)
    '北京市', '杭州市', '西安市',        # R^2 低但高负担
]


def main():
    os.makedirs(OUT, exist_ok=True)
    cases, _ = load_daily_cases()
    r2, r2_sem, rmse, rmse_sem = load_per_city_r2_rmse()
    beta, contact, rt = load_param_mean()   # (136, 24) 20 种子均值
    dates = cases.index                     # DatetimeIndex(136)

    r2_map = {c: r2[CITIES.index(c)] for c in CITIES}
    rmse_map = {c: rmse[CITIES.index(c)] for c in CITIES}

    # 打印全部城市 R^2 排序 (便于核对特殊城市)
    order = sorted(CITIES, key=lambda c: r2_map[c])
    print('=== 全部城市按 R^2 升序 (特殊城市加 * ) ===')
    for c in order:
        star = ' *' if c in SPECIAL else ''
        print(f'  {_CITY_EN[c]:12s}  R2={r2_map[c]:+.3f}  RMSE={rmse_map[c]:.2f}{star}')

    n = len(SPECIAL)
    fig = plt.figure(figsize=(12.5, 2.55 * n))
    outer = fig.add_gridspec(n, 1, hspace=0.55, left=0.08, right=0.97,
                             top=0.985, bottom=0.03)

    x = np.arange(len(dates))

    for i, cn in enumerate(SPECIAL):
        ci = CITIES.index(cn)
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], hspace=0.28)

        # ---- 上: 每日病例 ----
        ax_c = fig.add_subplot(inner[0])
        daily = cases[cn].values
        ax_c.fill_between(x, daily, color=CASE_C, alpha=0.35, lw=0)
        ax_c.plot(x, daily, '-', color=CASE_C, lw=1.0, alpha=0.95)
        ax_c.axvspan(TEST_START, TEST_END, color=TEST_C, zorder=0)
        ax_c.axvline(LOCKDOWN_DAY, color='#c0392b', ls='--', lw=1.0, alpha=0.85)
        ax_c.set_ylabel('daily\ncases', fontsize=9)
        ax_c.set_facecolor('white')
        for sp in ('top', 'right'):
            ax_c.spines[sp].set_visible(False)
        ax_c.tick_params(labelsize=8, labelbottom=False)
        ax_c.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax_c.set_title(
            f"{_CITY_EN[cn]}   R$^2$={r2_map[cn]:+.2f}   RMSE={rmse_map[cn]:.2f}",
            fontsize=11, fontweight='bold', loc='left', pad=4)

        # ---- 下: 参数 ----
        ax_p = fig.add_subplot(inner[1])
        ax_rt = ax_p.twinx()
        seg = slice(TRIM_START, TRIM_END + 1)
        xs = x[seg]
        ax_p.plot(xs, beta[seg, ci], '-', color=BETA_C, lw=1.4, label=r'$\beta$')
        ax_p.plot(xs, contact[seg, ci], '--', color=CONTACT_C, lw=1.4, label='$c$')
        ax_rt.plot(xs, rt[seg, ci], '-', color=RT_C, lw=1.4, label=r'$R_t$')
        ax_rt.axhline(1.0, color=RT1_C, ls='--', lw=0.9, alpha=0.8)
        ax_p.axvline(LOCKDOWN_DAY, color='#c0392b', ls='--', lw=1.0, alpha=0.85)
        ax_p.set_ylabel(r'$\beta$ / $c$', fontsize=9, color=BETA_C, labelpad=1)
        ax_rt.set_ylabel(r'$R_t$', fontsize=9, color=RT_C, labelpad=1)
        ax_p.tick_params(axis='y', labelsize=8, colors=BETA_C)
        ax_rt.tick_params(axis='y', labelsize=8, colors=RT_C)
        ax_p.set_facecolor('white')
        for sp in ('top', 'right'):
            ax_p.spines[sp].set_visible(False)
        ax_rt.spines['top'].set_visible(False)
        # x 轴日期刻度 (mm-dd)
        n_xticks = 6
        tk = np.linspace(0, len(dates) - 1, n_xticks, dtype=int)
        ax_p.set_xticks(x[tk])
        ax_p.set_xticklabels([dates[t].strftime('%m-%d') for t in tk], fontsize=8)

        if i == 0:
            h1, l1 = ax_p.get_legend_handles_labels()
            h2, l2 = ax_rt.get_legend_handles_labels()
            ax_p.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8,
                        ncol=3, framealpha=0.9, handlelength=1.4, columnspacing=0.7)

    fig.patch.set_facecolor('white')
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT, f'special_cities_cases_params.{ext}'),
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Done. 输出:', os.path.join(OUT, 'special_cities_cases_params.{pdf,png,svg}'))


if __name__ == '__main__':
    main()
