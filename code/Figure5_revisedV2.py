import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib.location import Location
from pathlib import Path

# 相对于项目根目录的路径（脚本需位于 Code/ 文件夹下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 全局样式
# =====================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'stix',
    'mathtext.default': 'regular',
    'pdf.fonttype': 42,
    'axes.linewidth': 0.8,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300
})

C_VIB  = '#E64B35'
C_SOL  = '#4DBBD5'
C_WIN  = '#00A087'
C_TRAF = '#7F7F7F'
TITLE_CFG = {'loc': 'left', 'pad': 12}

# =====================================================
# (a) 时空同步示意（明确标注为基于简化假设的概念性曲线）
# =====================================================
loc = Location(34.05, -118.24, 'US/Pacific')
times = pd.date_range('2024-06-21', '2024-06-22', freq='10min', tz=loc.tz)
solar_real = (loc.get_clearsky(times)['ghi'] / 1100 * 100).values[:144]
t_h = np.linspace(0, 24, 144)

# 概念性交通强度曲线（早晚高峰高斯近似，仅用于示意，非实测数据）
traffic_demand = (40 * np.exp(-(t_h - 8)**2 / 4) + 40 * np.exp(-(t_h - 18)**2 / 5) + 15)
# 振动能供给假设与交通强度成正比（比例系数0.9为示意性假设，非独立计算值）
VIB_TRAFFIC_COUPLING = 0.9
vibration_supply = traffic_demand * VIB_TRAFFIC_COUPLING

fig, axs = plt.subplots(2, 2, figsize=(10.5, 8.5))
plt.subplots_adjust(wspace=0.32, hspace=0.42)

ax = axs[0, 0]
ax.plot(t_h, traffic_demand, color=C_TRAF, lw=1.5, ls='--', label='Infrastructure Demand (illustrative)')
ax.plot(t_h, solar_real, color=C_SOL, lw=1.2, alpha=0.8, label='Solar Power (PVLib clear-sky model)')
ax.plot(t_h, vibration_supply, color=C_VIB, lw=2.2, label='Vibration Power (assumed prop. traffic, k=0.9)')
ax.axvspan(0, 5, color='gray', alpha=0.08, label='Off-grid Night Void')
ax.set_title('a | Conceptual Temporal Synchronization', **TITLE_CFG)
ax.set_xlabel('Local Time (h)')
ax.set_ylabel('Normalized Energy Flow (%)')
ax.set_xlim(0, 24)
ax.set_ylim(0, 110)
ax.legend(frameon=False, ncol=1, loc='upper right', fontsize=7.5)

# -----------------------------------------------------
# (b) 雷达图：精简为4个有物理依据的维度，三档定性评分（1/2/3，避免假精度）
# -----------------------------------------------------
axs[0, 1].axis('off')
ax_r = fig.add_subplot(2, 2, 2, projection='polar')

# 4个维度均可由各能源的基本物理工作原理直接说明，不依赖主观打分：
#  - Night/Low-light Availability: 光伏依赖直接辐照，夜间/隧道内出力为0（物理事实）
#  - Weather Resilience: 云层/降水显著降低光伏与风电出力；振动能不依赖光照或风速
#  - Demand Synchronization: 振动能产生与车流（即用电需求）天然同步；光伏/风电出力与交通需求时序无关
#  - Storage Independence: 振动能供给与需求同步，无需缓冲储能；光伏/风电需储能弥合时序错配
labels = ['Night/Low-light\nAvailability', 'Weather\nResilience',
          'Demand\nSynchronization', 'Storage\nIndependence']
# 三档定性等级：1=Low/None, 2=Partial, 3=Full/High
v_d = [3, 3, 3, 3]   # 振动能：与车流耦合，不依赖光照/天气，天然同步，无需储能
s_d = [1, 1, 1, 1]   # 光伏：夜间为0，受天气影响大，与需求不同步，依赖储能
w_d = [2, 1, 1, 1]   # 风电：可有夜间出力但具随机性，受天气影响大，与需求不同步，依赖储能

ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
ang += ang[:1]

for d, c, l in zip([v_d, s_d, w_d], [C_VIB, C_SOL, C_WIN], ['Vibration', 'Solar', 'Wind']):
    d = d + d[:1]
    ax_r.plot(ang, d, color=c, lw=1.8, label=l)
    ax_r.fill(ang, d, color=c, alpha=0.06)
ax_r.set_xticks(ang[:-1])
ax_r.set_xticklabels(labels, size=7.0)
ax_r.set_yticks([1, 2, 3])
ax_r.set_yticklabels(['Low', 'Partial', 'High'], size=6.0)
ax_r.set_ylim(0, 3.3)
ax_r.set_title('b | Qualitative Comparison of Operating\nCharacteristics (physically grounded)', **TITLE_CFG)
ax_r.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=7.5)

# -----------------------------------------------------
# (c) 空间场景：改为定性分级（不再使用暗示精确测量的百分比）
# -----------------------------------------------------
ax = axs[1, 0]
x = np.arange(5)
scenario_labels = ['Open Highway', 'Mountain Road', 'Viaduct/Bridge', 'Portal Shed', 'Deep Tunnel']
# 定性等级：3=Full, 2=Partial, 1=Minimal, 0=None
# 振动能：与光照、风速、地形遮蔽无关，全场景理论上均可正常产生（物理依据）
vib_q = [3, 3, 3, 3, 3]
# 光伏：随遮蔽程度增加而递减，隧道内为0（物理事实）
sol_q = [3, 2, 2, 1, 0]
# 风电：受地形/管道效应影响，趋势具有较大不确定性，此处仅作定性示意
win_q = [3, 2, 2, 1, 1]

ax.step(x, vib_q, where='mid', color=C_VIB, lw=2.5, label='Vibration')
ax.step(x, sol_q, where='mid', color=C_SOL, alpha=0.7, ls=':', label='Solar')
ax.step(x, win_q, where='mid', color=C_WIN, alpha=0.7, ls='-.', label='Wind')
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, rotation=15, ha='right')
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['None', 'Minimal', 'Partial', 'Full'])
ax.set_title('c | Qualitative Spatial Scenario Comparison', **TITLE_CFG)
ax.set_ylabel('Expected Operational Continuity')
ax.set_ylim(-0.3, 3.5)
ax.legend(frameon=False, loc='lower left', fontsize=7.5)

# -----------------------------------------------------
# (d) 经济收益：去除不对称储能惩罚项，只展示裸收益对比
# -----------------------------------------------------
ax = axs[1, 1]
ELECTRICITY_PRICE = 0.16        # US$/kWh — global average business electricity price, GlobalPetrolPrices Q1 2026 update
CARBON_FACTOR = 0.45e-3         # tCO2/kWh (=0.45 kgCO2/kWh) — global average CO2 intensity of electricity, IEA Electricity 2025 (445 gCO2/kWh in 2024)
cp = np.linspace(0, 180, 200)

# 振动能采用与全文统一口径一致的ISO Class D全球平均值
annual_kwh = {'Vibration': 8272, 'Solar': 14500, 'Wind': 9500}

for src, kwh in annual_kwh.items():
    rev = kwh * ELECTRICITY_PRICE + kwh * CARBON_FACTOR * cp
    if src == 'Vibration':
        ax.plot(cp, rev, color=C_VIB, lw=2.2, label=f'{src} (zero storage required)')
        ax.fill_between(cp, rev * 0.92, rev * 1.08, color=C_VIB, alpha=0.1)
    elif src == 'Solar':
        ax.plot(cp, rev, color=C_SOL, lw=1.5, ls='--', label=f'{src} (requires storage, not included)')
    else:
        ax.plot(cp, rev, color=C_WIN, lw=1.5, ls='-.', label=f'{src} (requires storage, not included)')

ax.axvline(x=39.5, color='gray', lw=0.8, ls='--', alpha=0.6)
ax.text(41, 2600, 'Global Mean\nCarbon Price', ha='left', va='top', fontsize=6.5, color='gray')

ax.set_title('d | Baseline Net Revenue per km\n(electricity + carbon credit only)', **TITLE_CFG)
ax.set_xlabel('Carbon Price (US$ / tCO2e)')
ax.set_ylabel(r'Net Annual Value (US$/km)')
ax.legend(frameon=False, loc='upper left', fontsize=7.5)

# -----------------------------------------------------
# (d) 图内假设边界注释
# 紧贴图表放置方法论限定说明，防止读者将收益侧对比
# 误读为完整成本效益分析（capex/O&M/储能均未纳入）
# -----------------------------------------------------
ax.text(
    0.01, -0.18,
    "Note: Capex, O&M, storage and civil works excluded. "
    "Revenue-side illustration only; not a full cost-benefit analysis.",
    transform=ax.transAxes,
    fontsize=6.5, color="#555555",
    va="top", ha="left", style="italic"
)
for a in [axs[0, 0], axs[1, 0], axs[1, 1]]:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Nature_Energy_Figure5_Revised.png', dpi=300, bbox_inches='tight')
plt.show()
