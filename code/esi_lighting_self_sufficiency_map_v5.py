# -*- coding: utf-8 -*-
"""
Global Road Vibration ESI Mapping
v3: Multi-reference calibration + single shared colorbar panel figure
    Output: one combined 4×2 panel image meeting Nature portfolio submission specs

Nature portfolio figure requirements (raster):
  - Minimum 300 dpi for colour figures (combination figures: 500 dpi recommended)
  - Width: 89 mm (1 column) / 183 mm (2 columns) / 247 mm (full page)
  - For a 2-column full-width panel (183 mm), 500 dpi → 3602 px wide
  - We target 500 dpi at 183 mm width → set figsize accordingly in inches
  - File format: TIFF (preferred) or high-quality PNG

Calibration references:
  [1] Hong Kong  — 102.2 GWh/yr  [Schedler et al. 2024; Hichou et al. 2024]
  [2] Singapore  —  81.0 GWh/yr  [LTA press release, 2023; sg101.gov.sg, 2024]
  [3] United Kingdom — 1887.0 GWh/yr  [UKRLG/CIHT State of the Nation, 2020]
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colorbar as mcolorbar
import numpy as np
import rasterio
import os
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ====================== Nature子刊排版参数 ======================
# 2列宽 = 183 mm = 7.205 inch；4行×2列组图
# 每个子图高宽比约 0.5（世界地图Robinson投影）
# 标签放在子图上方（ax.set_title），需要留出行间距
PANEL_COLS   = 2
PANEL_ROWS   = 4
SUB_W_INCH   = 7.205 / PANEL_COLS   # 每个子图宽度（inch）
SUB_H_INCH   = SUB_W_INCH * 0.50    # 每个子图高度（inch）
CBAR_H_INCH  = 0.65                 # 底部色柱区高度（含两端文字留白）
TOP_H_INCH   = 0.05                 # 顶部留白
FIG_W_INCH   = SUB_W_INCH * PANEL_COLS
FIG_H_INCH   = SUB_H_INCH * PANEL_ROWS + CBAR_H_INCH + TOP_H_INCH
OUTPUT_DPI   = 500                  # Nature推荐组合图500 dpi

# ====================== 字体设置（Nature风格：Arial，偏小） ======================
plt.rcParams.update({
    "font.family":       "Arial",
    "font.size":         7,          # Nature正文图注字号
    "axes.titlesize":    8,
    "axes.titleweight":  "bold",
    "axes.titlepad":     4,
    "legend.fontsize":   7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
})

# ====================== 路径配置（相对于项目根目录，脚本需位于 Code/ 文件夹下） ======================
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR         = PROJECT_ROOT / "Data"
GIS_DIR          = PROJECT_ROOT / "GIS"
RASTER_DIR       = PROJECT_ROOT / "Raster"
OUTPUT_DIR       = PROJECT_ROOT / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_PATH       = DATA_DIR / "Global_Vibration_Results_Final.xlsx"
TRAFFIC_PATH     = DATA_DIR / "traffic_prediction_results_40pct.xlsx"
ROAD_SHP_PATH    = GIS_DIR / "ne_10m_roads" / "ne_10m_roads.shp"
COUNTRY_SHP_PATH = GIS_DIR / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
OUTPUT_FILE      = OUTPUT_DIR / "Figure4_ESI_panel_Nature_v4.tif"  # v4: RdBu_r配色 + 标签移出图外

# ====================== 三参考区域 ======================
REFERENCE_REGIONS = [
    (113.8, 114.5, 22.1, 22.6,   102.2, "Hong Kong"),
    (103.6, 104.1,  1.1,  1.5,    81.0, "Singapore"),
    ( -8.2,   2.0, 49.8, 60.9,  1887.0, "United Kingdom"),
]

# ====================== 辅助函数 ======================
def find_tif(root):
    for r, d, f in os.walk(root):
        for file in f:
            if file.endswith(".tif") and not file.endswith(".ovr") \
                    and ("VNL" in file or "average_masked" in file):
                path = os.path.join(r, file)
                print("✓ Night-light TIF:", path)
                return path
    raise FileNotFoundError("VNL TIF not found")


def sample_vnl(line, src, step=0.01):
    if line is None or line.is_empty:
        return 0
    try:
        n = max(int(line.length / step), 3)
        pts = [line.interpolate(i / n, normalized=True) for i in range(n + 1)]
        vals = [float(v[0]) for v in src.sample([(p.x, p.y) for p in pts])]
        vals = [v for v in vals if np.isfinite(v) and v >= 0]
        return np.mean(vals) if vals else 0
    except Exception:
        return 0


def roads_in_bbox(roads_gdf, lon_min, lon_max, lat_min, lat_max):
    return roads_gdf.cx[lon_min:lon_max, lat_min:lat_max].copy()


def clean(s):
    return str(s).strip().upper()


def calibrate_multi_reference(roads, src, ref_regions):
    log_A_values, weights = [], []
    for (lon_min, lon_max, lat_min, lat_max, demand_GWh, name) in ref_regions:
        ref_roads = roads_in_bbox(roads, lon_min, lon_max, lat_min, lat_max).copy()
        if len(ref_roads) == 0:
            print(f"  [WARN] No roads in {name} — skipping")
            continue
        ref_roads['v'] = ref_roads.geometry.apply(lambda g: sample_vnl(g, src))
        vnl_total = (ref_roads['v'] * ref_roads['length_km']).sum()
        if vnl_total <= 0:
            print(f"  [WARN] VNL total=0 for {name} — skipping")
            continue
        demand_J   = demand_GWh * 3.6e12
        A_cand     = demand_J / vnl_total
        w          = np.log10(demand_GWh)
        log_A_values.append(np.log(A_cand) * w)
        weights.append(w)
        print(f"  [{name}]  demand={demand_GWh} GWh | "
              f"VNL·km={vnl_total:.4e} | A={A_cand:.4e}")
    if not log_A_values:
        raise RuntimeError("All calibration regions failed")
    A = np.exp(np.sum(log_A_values) / np.sum(weights))
    print(f"\n✓ Calibrated A = {A:.4e}  (weighted geometric mean, 3 references)\n")
    return A


# ====================== 主程序 ======================
def run():
    # --- 读取数据 ---
    roads     = gpd.read_file(ROAD_SHP_PATH).to_crs(4326)
    countries = gpd.read_file(COUNTRY_SHP_PATH).to_crs(4326)
    countries = countries[countries['ADM0_A3'] != '-99'].copy()
    roads     = roads[roads.is_valid].copy()

    print("Calculating road lengths...")
    roads['length_km'] = roads.to_crs("ESRI:54009").geometry.length / 1000

    countries_buf             = countries.copy()
    countries_buf['geometry'] = countries_buf.geometry.buffer(0.05)
    roads_iso = gpd.sjoin(
        roads.reset_index(drop=True),
        countries_buf[['ADM0_A3', 'geometry']],
        how='left', predicate='intersects'
    )
    roads_iso = (roads_iso
                 .sort_values('length_km', ascending=False)
                 .drop_duplicates(subset=['geometry'], keep='first'))
    print(f"Roads matched: {roads_iso['ADM0_A3'].notna().sum()} / {len(roads)}")
    road_len_map = roads_iso.groupby('ADM0_A3')['length_km'].sum().to_dict()

    # --- 国家代码映射 ---
    df_traffic = pd.read_excel(TRAFFIC_PATH)
    df_traffic['CLEAN'] = df_traffic['Country'].apply(clean)
    name_map = dict(zip(df_traffic['CLEAN'], df_traffic['Country Code'].str.strip()))
    name_map.update({
        'DEM. REP. CONGO': 'COD', 'DEMOCRATIC REPUBLIC OF THE CONGO': 'COD',
        'CONGO': 'COG', 'UNITED STATES': 'USA', 'RUSSIA': 'RUS',
        'VIETNAM': 'VNM', 'IRAN': 'IRN', 'EGYPT': 'EGY', 'SYRIA': 'SYR',
        'UNITED KINGDOM': 'GBR', 'UK': 'GBR',
    })
    df_full = pd.read_excel(EXCEL_PATH)
    df_full['CLEAN'] = df_full['Country'].apply(clean)
    df_full['ISO3']  = df_full['CLEAN'].map(name_map)
    df_full.loc[df_full['ISO3'].isna(), 'ISO3'] = \
        df_full.loc[df_full['ISO3'].isna(), 'CLEAN'].str[:3]

    # --- 标定 + 全球VNL采样 ---
    TIF_PATH = find_tif(RASTER_DIR)
    with rasterio.open(TIF_PATH) as src:
        print("=== Multi-reference VNL calibration ===")
        A = calibrate_multi_reference(roads, src, REFERENCE_REGIONS)
        print("Global night-light sampling...")
        roads_iso['v']      = roads_iso.geometry.apply(
            lambda g: sample_vnl(g, src))
        roads_iso['Demand'] = roads_iso['v'] * A

    # ── 第一遍：计算所有8个ISO等级的ESI，确定全局vmin/vmax ──────────
    iso_levels = sorted(df_full['ISO_Level'].unique())
    labels     = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    print("Pass 1: computing ESI for all ISO levels to find global color range...")
    all_esi_data = []   # 存储每个等级的 (v_roads_robin, esi_series)

    roads_bg = roads.to_crs("+proj=robin")   # 背景灰色道路（只算一次）

    # ====== 新增代码：用于记录各ISO等级的达标道路占比 ======
    coverage_results = []
    # =======================================================

    for lvl in iso_levels:
        df    = df_full[df_full['ISO_Level'] == lvl].copy()
        e_map = dict(zip(df['ISO3'], df['E_Total']))
        roads_iso['Country_E'] = roads_iso['ADM0_A3'].map(e_map)
        v_roads = roads_iso.dropna(subset=['Country_E']).copy()
        v_roads['Road_Total']     = v_roads['ADM0_A3'].map(road_len_map)
        v_roads['Supply_Density'] = v_roads['Country_E'] / v_roads['Road_Total']
        
        # 计算 ESI
        v_roads['ESI'] = np.log10(
            (v_roads['Supply_Density'] / (v_roads['Demand'] + 1e-9)) + 1e-20
        )
        roads_plot = v_roads.to_crs("+proj=robin")
        all_esi_data.append(roads_plot)

        # ====== 新增代码：计算可满足本地照明需求的道路占比 ======
        # 核心逻辑：ESI >= 0 即代表 Supply >= Demand
        mask_meet = v_roads['ESI'] >= 0
        meet_len = v_roads.loc[mask_meet, 'length_km'].sum()
        total_len = v_roads['length_km'].sum()
        pct = (meet_len / total_len) * 100 if total_len > 0 else 0
        
        coverage_results.append((lvl, pct))
        print(f"  [{lvl}] Roads meeting demand (ESI >= 0): {pct:.2f}% ({meet_len:.1f} km / {total_len:.1f} km)")
        # ========================================================

    # ====== 新增代码：循环结束后集中打印，供论文摘要提取数据 ======
    print("\n" + "="*55)
    print(" SUMMARY: Percentage of Roads Meeting Lighting Demand")
    print("="*55)
    for lvl, pct in coverage_results:
        print(f" {lvl}: {pct:.2f}%")
    print("="*55 + "\n")
    # ========================================================

    # 全局5th–95th百分位，确保所有子图用同一色阶
    all_esi_vals = pd.concat([d['ESI'] for d in all_esi_data], ignore_index=True)
    global_vmin  = np.percentile(all_esi_vals.dropna(), 5)
    global_vmax  = np.percentile(all_esi_vals.dropna(), 95)
    if global_vmin >= 0:  global_vmin = -0.1
    if global_vmax <= 0:  global_vmax =  0.1
    print(f"Global ESI range (5th–95th pct): [{global_vmin:.3f}, {global_vmax:.3f}]")

    norm = TwoSlopeNorm(vmin=global_vmin, vcenter=0, vmax=global_vmax)
    # RdBu_r: 负值深红(deficit) → 白色(zero) → 深蓝(surplus)
    # 在白色背景上对比度远优于RdYlGn（中间黄色与白色难以区分）
    cmap = plt.get_cmap('RdBu_r')

    # ── 第二遍：绘制组图 ─────────────────────────────────────────────
    print("\nPass 2: rendering panel figure...")

    # 计算各区域高度比例（子图区 vs 色柱区）
    map_h_total  = SUB_H_INCH * PANEL_ROWS
    total_h      = FIG_H_INCH
    # gridspec: PANEL_ROWS行子图 + 1行色柱；用height_ratios控制
    fig = plt.figure(figsize=(FIG_W_INCH, FIG_H_INCH), dpi=OUTPUT_DPI,
                     facecolor='white')

    # 手动计算各部分占比
    cbar_frac  = CBAR_H_INCH  / FIG_H_INCH
    top_frac   = TOP_H_INCH   / FIG_H_INCH
    map_frac   = 1.0 - cbar_frac - top_frac

    gs = fig.add_gridspec(
        nrows=PANEL_ROWS + 1,
        ncols=PANEL_COLS,
        height_ratios=[map_frac / PANEL_ROWS] * PANEL_ROWS + [cbar_frac],
        hspace=0.12,   # 增大行间距，为子图标题留出空间（原0.04会导致标题压到地图）
        wspace=0.02,   # 列间距（极小）
        left=0.01, right=0.99,
        top=1.0 - top_frac, bottom=0.0
    )

    map_bounds = roads_bg.total_bounds  # [xmin, ymin, xmax, ymax]

    for i, (lvl, roads_plot) in enumerate(zip(iso_levels, all_esi_data)):
        row = i // PANEL_COLS
        col = i  % PANEL_COLS
        ax  = fig.add_subplot(gs[row, col])

        # 背景灰色道路
        roads_bg.plot(ax=ax, color='#d0d0d0', linewidth=0.10,
                      alpha=0.70, zorder=1)

        # ESI着色道路（legend=False，统一用下方色柱）
        roads_plot.plot(
            column='ESI', cmap=cmap, norm=norm,
            linewidth=0.55, ax=ax,
            legend=False,
            zorder=2
        )

        ax.set_xlim(map_bounds[0], map_bounds[2])
        ax.set_ylim(map_bounds[1], map_bounds[3])
        ax.axis('off')

        # 子图标签：放在子图上方（set_title），完全不与地图内容重叠
        # loc='left' 使标签左对齐，符合Nature子刊惯例
        label_str = f'({labels[i]}) {lvl.replace("_", " ")}'
        ax.set_title(label_str, loc='left', fontsize=8,
                     fontweight='bold', pad=3)

        print(f"  [{lvl}] rendered")

    # ── 底部共享色柱 ──────────────────────────────────────────────────
    # 跨越底行全部两列
    cbar_ax = fig.add_subplot(gs[PANEL_ROWS, :])
    cbar_ax.set_visible(False)   # 隐藏坐标轴本身

    # 在色柱行手动添加一个细长axes
    # 位置用fig.add_axes([left, bottom, width, height]) in figure fraction
    cbar_left   = 0.20
    cbar_width  = 0.60
    cbar_bottom = 0.012
    cbar_height = 0.018

    cbar_ax2 = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax2, orientation='horizontal')
    cb.set_label(
        r'Energy Self-sufficiency Index (ESI) = Log$_{10}$(Supply / Demand)',
        fontsize=7, labelpad=3
    )
    cb.ax.tick_params(labelsize=6)

    # 在色柱两端加文字说明（颜色与RdBu_r配色一致）
    cbar_ax2.text(-0.01, 0.5, 'Deficit\n(Supply < Demand)',
                  transform=cbar_ax2.transAxes,
                  ha='right', va='center', fontsize=6, color='#b2182b')
    cbar_ax2.text(1.01, 0.5, 'Surplus\n(Supply > Demand)',
                  transform=cbar_ax2.transAxes,
                  ha='left', va='center', fontsize=6, color='#2166ac')

    # ── 保存 ─────────────────────────────────────────────────────────
    out_path = OUTPUT_FILE
    fig.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches='tight',
                pad_inches=0.03, format='tiff',
                pil_kwargs={"compression": "tiff_lzw"})  # LZW无损压缩
    plt.close()
    print(f"\n✓ Panel figure saved: {out_path}")
    print(f"  Target size: {FIG_W_INCH:.2f} × {FIG_H_INCH:.2f} inch @ {OUTPUT_DPI} dpi")
    print(f"  → approx. {int(FIG_W_INCH*OUTPUT_DPI)} × {int(FIG_H_INCH*OUTPUT_DPI)} px")
    print("\n✓ ALL DONE")


if __name__ == "__main__":
    run()
