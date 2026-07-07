# -*- coding: utf-8 -*-
"""
Global Road Vibration Energy Density Mapping
v2: Single shared colorbar panel figure (Nature portfolio submission spec)

Differences from the ESI panel figure (Figure 4):
  - This figure shows recoverable energy DENSITY only (supply side),
    not a supply/demand ratio.
  - Uses a SEQUENTIAL colormap (inferno) with a fixed range
    [VMIN=7, VMAX=16] in log10(J/km), not a diverging colormap.
  - One shared horizontal colorbar at the bottom of the panel.

Nature portfolio figure requirements (raster):
  - Minimum 300 dpi for colour figures (500 dpi recommended for combination figures)
  - 2-column width = 183 mm
  - Output format: TIFF (LZW lossless compression)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =====================================================
# Nature 排版参数（与 Figure 4 一致）
# =====================================================
PANEL_COLS  = 2
PANEL_ROWS  = 4
SUB_W_INCH  = 7.205 / PANEL_COLS     # 183 mm / 2
SUB_H_INCH  = SUB_W_INCH * 0.50
CBAR_H_INCH = 0.65
TOP_H_INCH  = 0.05
FIG_W_INCH  = SUB_W_INCH * PANEL_COLS
FIG_H_INCH  = SUB_H_INCH * PANEL_ROWS + CBAR_H_INCH + TOP_H_INCH
OUTPUT_DPI  = 500

plt.rcParams.update({
    "font.family":      "Arial",
    "font.size":        7,
    "axes.titlesize":   8,
    "axes.titleweight": "bold",
    "axes.titlepad":    4,
    "legend.fontsize":  7,
})

# =====================================================
# 路径设置（相对于项目根目录，脚本需位于 Code/ 文件夹下）
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GIS_DIR      = PROJECT_ROOT / "GIS"
DATA_DIR     = PROJECT_ROOT / "Data"
BASE_DIR     = PROJECT_ROOT / "Output"
BASE_DIR.mkdir(parents=True, exist_ok=True)

SHP_PATH     = GIS_DIR / "ne_10m_roads" / "ne_10m_roads.shp"
WORLD_SHP    = GIS_DIR / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
VIB_DATA     = DATA_DIR / "Global_Vibration_Results_Final.xlsx"
TRAFFIC_DATA = DATA_DIR / "traffic_prediction_results_40pct.xlsx"

OUTPUT_FILE = BASE_DIR / "Figure3_EnergyDensity_panel_Nature.tif"

# 固定色标范围（与原代码一致）
VMIN = 7
VMAX = 16

# =====================================================
# 1. 数据加载
# =====================================================
print("正在加载全球路网数据...")
all_roads = gpd.read_file(SHP_PATH).to_crs(epsg=4326)

print("正在加载国家边界...")
world_ref = gpd.read_file(WORLD_SHP).to_crs(epsg=4326)
iso_col = 'ADM0_A3'
world_ref = world_ref[world_ref[iso_col] != '-99']

# =====================================================
# 2. 空间连接
# =====================================================
print("正在执行空间连接...")
roads_iso = gpd.sjoin(
    all_roads, world_ref[[iso_col, 'geometry']],
    how="inner", predicate='intersects'
)

# =====================================================
# 3. 计算道路长度
# =====================================================
print("正在计算国家道路总长度...")
roads_proj = roads_iso.to_crs("ESRI:54009")
roads_iso['len_km'] = roads_proj.geometry.length / 1000
road_sum = roads_iso.groupby(iso_col)['len_km'].sum().reset_index()

# =====================================================
# 4. 读取模型输出 & 国家代码映射
# =====================================================
print("正在读取振动能结果...")
df_vib     = pd.read_excel(VIB_DATA)
df_traffic = pd.read_excel(TRAFFIC_DATA)

def clean(s):
    return str(s).strip().upper()

df_vib['C_Clean']     = df_vib['Country'].apply(clean)
df_traffic['C_Clean'] = df_traffic['Country'].apply(clean)

name_map = dict(zip(df_traffic['C_Clean'], df_traffic['Country Code'].str.strip()))
name_map.update({
    'DEM. REP. CONGO': 'COD', 'CONGO': 'COG', 'RUSSIA': 'RUS',
    'EGYPT': 'EGY', 'UNITED STATES': 'USA', 'VIETNAM': 'VNM',
    'IRAN': 'IRN', 'SYRIA': 'SYR'
})
df_vib['MAP_ISO'] = df_vib['C_Clean'].map(name_map)

# =====================================================
# 5. ISO 等级
# =====================================================
iso_levels   = ['ISO_A','ISO_B','ISO_C','ISO_D','ISO_E','ISO_F','ISO_G','ISO_H']
panel_labels = ['a','b','c','d','e','f','g','h']

all_stats_summary = []

# =====================================================
# 6. Pass 1 — 计算每个ISO等级的 render_gdf（Robinson投影）
# =====================================================
print("\nPass 1: computing energy density for all ISO levels...")

roads_bg = all_roads.to_crs("+proj=robin")   # 全球背景灰色路网（只需一次）

render_gdfs = []

for level in iso_levels:
    print(f"  Processing {level}...")

    df_level = df_vib[df_vib['ISO_Level'] == level].copy()
    df_res   = df_level.merge(road_sum, left_on='MAP_ISO', right_on=iso_col)

    # Recoverable Energy Density
    df_res['Energy_Density'] = df_res['E_Total'] / df_res['len_km']

    # 去除超小国家
    df_res = df_res[df_res['len_km'] >= 50].copy()

    # 对数尺度
    df_res['log10_ED'] = np.log10(df_res['Energy_Density'].replace(0, np.nan))

    # 统计分析
    stats   = df_res['log10_ED'].describe(percentiles=[.25, .5, .75])
    top_5   = df_res.sort_values('log10_ED', ascending=False).head(5)

    print(f"    median={stats['50%']:.2f}, mean={stats['mean']:.2f}, "
          f"top1={top_5['Country'].iloc[0]}")

    all_stats_summary.append({
        'Level': level,
        'Median': stats['50%'],
        'Mean': stats['mean'],
        'Max': stats['max'],
        'Top1_Country': top_5['Country'].iloc[0]
    })

    # 构建绘图 GeoDataFrame
    render_gdf = roads_iso.merge(
        df_res[['MAP_ISO', 'log10_ED']],
        left_on=iso_col, right_on='MAP_ISO'
    )
    render_gdf = render_gdf[
        render_gdf.geometry.type.isin(['LineString', 'MultiLineString'])
    ]
    render_gdf = render_gdf.to_crs("+proj=robin")

    render_gdfs.append(render_gdf)

# =====================================================
# 7. Pass 2 — 绘制组图（共享色柱）
# =====================================================
print("\nPass 2: rendering panel figure...")

cmap = plt.get_cmap('inferno')
norm = Normalize(vmin=VMIN, vmax=VMAX)

fig = plt.figure(figsize=(FIG_W_INCH, FIG_H_INCH), dpi=OUTPUT_DPI, facecolor='white')

cbar_frac = CBAR_H_INCH / FIG_H_INCH
top_frac  = TOP_H_INCH  / FIG_H_INCH
map_frac  = 1.0 - cbar_frac - top_frac

gs = fig.add_gridspec(
    nrows=PANEL_ROWS + 1, ncols=PANEL_COLS,
    height_ratios=[map_frac / PANEL_ROWS] * PANEL_ROWS + [cbar_frac],
    hspace=0.12, wspace=0.02,
    left=0.01, right=0.99,
    top=1.0 - top_frac, bottom=0.0
)

map_bounds = roads_bg.total_bounds

for i, (level, render_gdf) in enumerate(zip(iso_levels, render_gdfs)):
    row, col = i // PANEL_COLS, i % PANEL_COLS
    ax = fig.add_subplot(gs[row, col])

    # 背景灰色全球路网
    roads_bg.plot(ax=ax, color='#d0d0d0', linewidth=0.10, alpha=0.70, zorder=1)

    # 着色路网（统一色标，不单独画色柱）
    if not render_gdf.empty:
        render_gdf.plot(
            ax=ax, column='log10_ED',
            cmap=cmap, norm=norm,
            linewidth=0.45,
            legend=False, zorder=2
        )

    ax.set_xlim(map_bounds[0], map_bounds[2])
    ax.set_ylim(map_bounds[1], map_bounds[3])
    ax.axis('off')

    # 标签放在子图外部上方，避免与地图重叠
    label_str = f'({panel_labels[i]}) {level.replace("_", " ")}'
    ax.set_title(label_str, loc='left', fontsize=8, fontweight='bold', pad=3)

    print(f"  [{level}] rendered")

# =====================================================
# 8. 底部共享色柱
# =====================================================
cbar_left   = 0.20
cbar_width  = 0.60
cbar_bottom = 0.012
cbar_height = 0.018

cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cb.set_label(
    r'Log$_{10}$ Recoverable Energy Density (J km$^{-1}$)',
    fontsize=7, labelpad=3
)
cb.ax.tick_params(labelsize=6)

# 色柱两端文字说明（顺序色标：低 → 高）
cbar_ax.text(-0.01, 0.5, 'Lower',
              transform=cbar_ax.transAxes,
              ha='right', va='center', fontsize=6, color='#444444')
cbar_ax.text(1.01, 0.5, 'Higher',
              transform=cbar_ax.transAxes,
              ha='left', va='center', fontsize=6, color='#444444')

# =====================================================
# 9. 保存
# =====================================================
fig.savefig(
    OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches='tight',
    pad_inches=0.03, format='tiff',
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()

print(f"\n✓ Panel figure saved: {OUTPUT_FILE}")
print(f"  Target size: {FIG_W_INCH:.2f} x {FIG_H_INCH:.2f} inch @ {OUTPUT_DPI} dpi")
print(f"  -> approx. {int(FIG_W_INCH*OUTPUT_DPI)} x {int(FIG_H_INCH*OUTPUT_DPI)} px")

# =====================================================
# 10. 输出统计文件
# =====================================================
summary_df   = pd.DataFrame(all_stats_summary)
summary_path = BASE_DIR / "Global_EnergyDensity_Statistics.xlsx"
summary_df.to_excel(summary_path, index=False)

print("\n" + "="*60)
print("全部任务完成！")
print("="*60)
print("\n输出文件：")
print(f"1. {OUTPUT_FILE}")
print(f"2. {summary_path}")
print("\n说明：")
print("- 8个ISO等级组图，单一共享色柱（inferno, log10 ED, 7-16）")
print("- 标签置于子图外部（set_title），避免与地图重叠")
print("- Robinson投影 + 灰色背景全球路网")
print("- 183 mm 宽，500 dpi，TIFF (LZW无损压缩)，符合Nature子刊要求")
print("="*60)
