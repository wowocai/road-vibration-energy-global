import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ===============================
# 1. 路径设置（相对于项目根目录，脚本需位于 Code/ 文件夹下）
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "Data"
OUTPUT_DIR   = PROJECT_ROOT / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE   = DATA_DIR / "merged_global_dataset_with_gdp.xlsx"
URBAN_FILE  = DATA_DIR / "API_SP.URB.TOTL.IN.ZS_DS2_en_excel_v2_182.xls"
OUT_FILE    = DATA_DIR / "traffic_prediction_results_40pct.xlsx"
VERIFY_RANK = OUTPUT_DIR / "national_rank_consistency_spearman.xlsx"
VERIFY_ACC  = OUTPUT_DIR / "national_prediction_accuracy_RMSLE.xlsx"
FIG_OUT     = OUTPUT_DIR / "Supplementary_Figure_1_Importance_Fixed.png"

# 变量名定义
TOTAL_COL = "Total four wheeled Traffic (Mio Vehicle-Km Annual)"
PASS_COL  = "Passenger Car Traffic (Mio Vehicle-Km Annual)"
BUS_COL   = "Bus and Motor Coach Traffic (Mio Vehicle-Km Annual)"
FRT_COL   = "Total Van, Pickup, Lorry and Road Tractor Traffic (Mio Vehicle-Km Annual)"

# ===============================
# 2. 数据读取与城市化率合并
# ===============================
print("Step 1: 正在读取数据并合并城市化率...")
df_main = pd.read_excel(DATA_FILE)
df_main.columns = df_main.columns.str.strip()

df_urban_raw = pd.read_excel(URBAN_FILE, sheet_name='Data', header=3)
year_cols = [col for col in df_urban_raw.columns if str(col).isdigit()]
latest_year = year_cols[-1]
df_urban = df_urban_raw[['Country Code', latest_year]].copy()
df_urban.columns = ['Country Code', 'Urbanization_Rate']

df = pd.merge(df_main, df_urban, on='Country Code', how='left')

if '2023 [YR2023]' in df.columns:
    df = df.rename(columns={'2023 [YR2023]': 'Population'})

BASE_FEATURES = [
    "Population", "Passenger Cars In Use (n)", "Buses and Motor Coaches In Use (n)",
    "Total Vans, Pickups, Lorries and Road Tractors In Use (n)",
    "Motorway or Highway Network (km)", "Main or National Road Network - Total (km)",
    "Secondary or Regional Road Network - Total (km)", "Other Roads - Combined (Urban and Rural) - Total (km)",
    "GDP_per_capita", "Urbanization_Rate"
]

for col in BASE_FEATURES + [TOTAL_COL, PASS_COL, BUS_COL, FRT_COL]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors='coerce')

# ===============================
# 3. 特征增强与分类逻辑
# ===============================
print("Step 2: 正在进行特征增强与国家分类...")
df['Vehicles_per_1000'] = (df['Passenger Cars In Use (n)'] / df['Population']) * 1000
df['Road_Density_per_capita'] = (df['Main or National Road Network - Total (km)'] / df['Population']) * 100000
df['GDP_Log'] = np.log1p(df['GDP_per_capita'])

ENHANCED_FEATURES = BASE_FEATURES + ['Vehicles_per_1000', 'Road_Density_per_capita', 'GDP_Log']

temp_gdp = np.log1p(df["GDP_per_capita"].fillna(df["GDP_per_capita"].median()))
temp_pop = np.log1p(df["Population"].fillna(df["Population"].median()))
df["Size_Score"] = temp_gdp * temp_pop
score_threshold = df["Size_Score"].quantile(0.6)
df["Country_Group"] = np.where(df["Size_Score"] >= score_threshold, "Large", "Small")

df["log_total"] = np.log(df[TOTAL_COL].replace(0, np.nan))
df["share_pass"] = df[PASS_COL] / df[TOTAL_COL]
df["share_bus"]  = df[BUS_COL]  / df[TOTAL_COL]
df["share_frt"]  = df[FRT_COL]  / df[TOTAL_COL]

# ===============================
# 4. 模型训练与 VKT 重构 (含特征重要性绘图)
# ===============================
print("Step 3: 正在训练随机森林模型并重构 VKT...")
def train_rf_optimized(X, y):
    model = RandomForestRegressor(n_estimators=800, max_depth=10, min_samples_leaf=2,
                                  max_features='sqrt', random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

outputs = []
for group in ["Large", "Small"]:
    sub = df[df["Country_Group"] == group].copy()
    group_median = sub[ENHANCED_FEATURES].median()
    X_all = sub[ENHANCED_FEATURES].fillna(group_median).fillna(0)

    train_t = sub.dropna(subset=["log_total"])
    if not train_t.empty:
        X_train = train_t[ENHANCED_FEATURES].fillna(group_median).fillna(0)
        y_train = train_t["log_total"]
        m_total = train_rf_optimized(X_train, y_train)
        total_pred = np.exp(m_total.predict(X_all))

        # --- 针对 Large 组生成特征重要性图表 ---
        if group == "Large":
            print("正在生成修正后的特征重要性图 (Supplementary Figure 1)...")
            importances = m_total.feature_importances_
            feat_imp_series = pd.Series(importances, index=ENHANCED_FEATURES).sort_values(ascending=True)

            plt.figure(figsize=(12, 8), dpi=300)
            ax = feat_imp_series.plot(kind='barh', color='#2c3e50', width=0.7)

            plt.title("Feature Importance for VKT Reconstruction (Large Countries)",
                      fontsize=16, fontweight='bold', pad=35)
            plt.xlabel("Gini Importance", fontsize=12, labelpad=10)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.grid(axis='x', linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.subplots_adjust(top=0.88)

            plt.savefig(FIG_OUT, bbox_inches='tight')
            plt.close()
        # --------------------------------------------

        share_t = sub.dropna(subset=["share_pass", "share_bus", "share_frt"])
        if not share_t.empty:
            X_s = share_t[ENHANCED_FEATURES].fillna(group_median).fillna(0)
            m_p = train_rf_optimized(X_s, np.clip(share_t["share_pass"], 1e-6, 1))
            m_b = train_rf_optimized(X_s, np.clip(share_t["share_bus"], 1e-6, 1))
            m_f = train_rf_optimized(X_s, np.clip(share_t["share_frt"], 1e-6, 1))

            sp, sb, sf = m_p.predict(X_all), m_b.predict(X_all), m_f.predict(X_all)
            s_sum = sp + sb + sf
            sub["Passenger_pred"] = total_pred * (sp / s_sum)
            sub["Bus_pred"]       = total_pred * (sb / s_sum)
            sub["Freight_pred"]   = total_pred * (sf / s_sum)
    outputs.append(sub)

df_out = pd.concat(outputs, axis=0)
drop_cols = ['Size_Score', 'Vehicles_per_1000', 'Road_Density_per_capita', 'GDP_Log', 'log_total']
final_save_cols = [c for c in df_out.columns if c not in drop_cols]
df_out[final_save_cols].to_excel(OUT_FILE, index=False)
print(f"主预测文件已保存: {OUT_FILE}")

# ===============================
# 5. 自动化验证 (Spearman + RMSLE)
# ===============================
print("Step 4: 正在进行自动化验证...")
pairs = {
    "Passenger": (PASS_COL, "Passenger_pred"),
    "Bus": (BUS_COL, "Bus_pred"),
    "Freight": (FRT_COL, "Freight_pred")
}

rank_res = []
for name, (true_c, pred_c) in pairs.items():
    sub_v = df_out[[true_c, pred_c]].dropna()
    if len(sub_v) >= 10:
        rho, pval = spearmanr(sub_v[true_c], sub_v[pred_c])
        rank_res.append({"Type": name, "N": len(sub_v), "Spearman_rho": rho, "p_value": pval})
pd.DataFrame(rank_res).to_excel(VERIFY_RANK, index=False)

acc_res = []
def get_rmsle(y_t, y_p):
    return np.sqrt(mean_squared_error(np.log1p(np.maximum(y_t, 0)), np.log1p(np.maximum(y_p, 0))))

for name, (true_c, pred_c) in pairs.items():
    sub_v = df_out[[true_c, pred_c]].dropna()
    if len(sub_v) >= 5:
        rmse = mean_squared_error(sub_v[true_c], sub_v[pred_c], squared=False)
        rmsle_val = get_rmsle(sub_v[true_c].values, sub_v[pred_c].values)
        acc_res.append([name, rmse, rmsle_val, len(sub_v)])
pd.DataFrame(acc_res, columns=["Indicator", "RMSE", "RMSLE", "N"]).to_excel(VERIFY_ACC, index=False)

print("\n========================================")
print("所有处理已完成！")
print(f"1. 预测结果: {OUT_FILE}")
print(f"2. 排序验证: {VERIFY_RANK}")
print(f"3. 精度验证: {VERIFY_ACC}")
print(f"4. 特征重要性图: {FIG_OUT}")
print("========================================")
