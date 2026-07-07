import pandas as pd
import numpy as np
from pathlib import Path

# 相对于项目根目录的路径（脚本需位于 Code/ 文件夹下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 1. 七自由度物理引擎 (7-DOF Dynamics Engine)
#
# v3 修正版说明：
# - 保留完整 7-DOF 耦合结构
# - 保留 pitch / roll 动力学耦合
# - 修正 tf/tr 为半轮距
# - 删除无物理依据的 Q 与 Imp 放大系数
# - Van 参数依据文献重新校准
# - ISO 8608 采用几何均值
# - 输出文件名保持与旧版兼容
# =====================================================

def calculate_unit_energy(Gq0, v_base=25.0, vehicle_type='PC'):
    """
    计算单位行驶距离悬架耗散能量 (J/km)

    Parameters
    ----------
    Gq0 : float
        ISO 8608 路面 PSD 系数 (m^3/cycle)

    v_base : float
        车辆巡航速度 (m/s)

    vehicle_type : str
        'PC' | 'Van' | 'Bus'

    Returns
    -------
    float
        单位距离耗散能量 (J/km)
    """

    # =================================================
    # 空间频率积分范围
    # =================================================
    n_space = np.linspace(0.01, 150.0, 10000)
    dn = n_space[1] - n_space[0]

    # =================================================
    # 车辆参数
    # =================================================
    configs = {

        # Passenger Car
        'PC': {
            'M': 1600.0,
            'Ip': 2400.0,
            'Ir': 1800.0,
            'mu': 60.0,

            'ks': 22000.0,
            'cs': 1600.0,
            'kt': 2e5,

            'a': 1.2,
            'b': 1.6,

            # 半轮距
            'tf': 0.8,
            'tr': 0.8
        },

        # Freight Vehicle / Heavy Truck
        'Van': {

            # 基于文献重新校准
            # Ali et al. (2019)
            # Analysis of the prospective vibrational
            # energy harvesting of heavy-duty truck suspensions

            'M': 8000.0,
            'Ip': 2.5e4,
            'Ir': 1.5e4,
            'mu': 200.0,

            'ks': 4.0e5,

            # 核心修正项
            'cs': 6.0e4,

            'kt': 8e5,

            'a': 1.5,
            'b': 2.5,

            'tf': 0.9,
            'tr': 0.9
        },

        # Bus
        'Bus': {
            'M': 16000.0,
            'Ip': 1.5e5,
            'Ir': 1.0e5,
            'mu': 450.0,

            'ks': 2.5e5,
            'cs': 3.5e4,
            'kt': 8e5,

            'a': 3.5,
            'b': 5.5,

            'tf': 1.1,
            'tr': 1.1
        }
    }

    p = configs[vehicle_type]

    # =================================================
    # 初始化矩阵
    #
    # 自由度：
    # [z_s, theta, phi, z_u1, z_u2, z_u3, z_u4]
    # =================================================
    M_mat = np.zeros((7, 7))
    C_mat = np.zeros((7, 7))
    K_mat = np.zeros((7, 7))

    # =================================================
    # 质量矩阵
    # =================================================
    M_mat[0, 0] = p['M']
    M_mat[1, 1] = p['Ip']
    M_mat[2, 2] = p['Ir']

    for i in range(4):
        M_mat[3 + i, 3 + i] = p['mu']

    # =================================================
    # 几何位置
    # rt : longitudinal lever arm
    # rp : lateral lever arm
    # =================================================
    rt = np.array([
        p['a'],
        p['a'],
        -p['b'],
        -p['b']
    ])

    rp = np.array([
        -p['tf'],
         p['tf'],
        -p['tr'],
         p['tr']
    ])

    ks = p['ks']
    cs = p['cs']
    kt = p['kt']

    # =================================================
    # 构建阻尼矩阵与刚度矩阵
    # =================================================
    for i in range(4):

        idx = 3 + i

        # ---------------------------------------------
        # 主对角项
        # ---------------------------------------------
        K_mat[0, 0] += ks
        C_mat[0, 0] += cs

        K_mat[1, 1] += ks * rt[i]**2
        C_mat[1, 1] += cs * rt[i]**2

        K_mat[2, 2] += ks * rp[i]**2
        C_mat[2, 2] += cs * rp[i]**2

        K_mat[idx, idx] += ks + kt
        C_mat[idx, idx] += cs

        # ---------------------------------------------
        # heave-wheel coupling
        # ---------------------------------------------
        K_mat[0, idx] -= ks
        K_mat[idx, 0] -= ks

        C_mat[0, idx] -= cs
        C_mat[idx, 0] -= cs

        # ---------------------------------------------
        # pitch-wheel coupling
        # ---------------------------------------------
        K_mat[1, idx] -= ks * rt[i]
        K_mat[idx, 1] -= ks * rt[i]

        C_mat[1, idx] -= cs * rt[i]
        C_mat[idx, 1] -= cs * rt[i]

        # ---------------------------------------------
        # roll-wheel coupling
        # ---------------------------------------------
        K_mat[2, idx] -= ks * rp[i]
        K_mat[idx, 2] -= ks * rp[i]

        C_mat[2, idx] -= cs * rp[i]
        C_mat[idx, 2] -= cs * rp[i]

    # =================================================
    # 频域积分
    # =================================================
    total_power = 0.0

    for n in n_space:

        # ISO 8608 PSD
        Sq = Gq0 * (n / 0.1) ** (-2.0)

        # 角频率
        omega = 2.0 * np.pi * n * v_base

        # 动刚度矩阵
        A = (
            -omega**2 * M_mat
            + 1j * omega * C_mat
            + K_mat
        )

        # 激励向量
        F = np.zeros(7, dtype=complex)
        F[3:] = kt

        try:
            X = np.linalg.solve(A, F)

        except np.linalg.LinAlgError:
            continue

        # ---------------------------------------------
        # 四个悬架阻尼器耗散
        # ---------------------------------------------
        for j in range(4):

            z_body = (
                X[0]
                + rt[j] * X[1]
                + rp[j] * X[2]
            )

            z_rel = z_body - X[3 + j]

            total_power += (
                cs
                * (np.abs(omega * z_rel) ** 2)
                * Sq
                * dn
            )

    # J/m → J/km
    return (total_power / v_base) * 1000.0


# =====================================================
# 2. ISO 8608 路面等级（几何均值）
# =====================================================
ISO_GQ_LEVELS = {

    # 几何均值
    "ISO_A": 4e-6,
    "ISO_B": 16e-6,
    "ISO_C": 64e-6,
    "ISO_D": 256e-6,
    "ISO_E": 1024e-6,
    "ISO_F": 4096e-6,
    "ISO_G": 16384e-6,
    "ISO_H": 65536e-6,
}

# =====================================================
# 3. 全球照明需求
# =====================================================

# 119.11 TWh
LIGHT_DEMAND = 4.2880e17

# 多车道增益
MULTI_VEHICLE_GAIN = 1.85

# =====================================================
# 4. 全球能量计算
# =====================================================

df_raw = pd.read_excel(
    DATA_DIR / "traffic_prediction_results_40pct.xlsx"
)

all_detailed = []
summary_data = []

print("正在执行全球振动能量评估...\n")

for iso_name, Gq0 in ISO_GQ_LEVELS.items():

    # 单位耗散能
    u_pc = calculate_unit_energy(
        Gq0,
        vehicle_type='PC'
    )

    u_van = calculate_unit_energy(
        Gq0,
        vehicle_type='Van'
    )

    u_bus = calculate_unit_energy(
        Gq0,
        vehicle_type='Bus'
    )

    # =================================================
    # 国家级结果
    # =================================================
    temp_df = pd.DataFrame({
        "Country": df_raw["Country"],
        "ISO_Level": iso_name
    })

    temp_df["E_PC"] = (
        df_raw["Passenger_pred"]
        * 1e6
        * u_pc
    )

    temp_df["E_VAN"] = (
        df_raw["Freight_pred"]
        * 1e6
        * u_van
    )

    temp_df["E_BUS"] = (
        df_raw["Bus_pred"]
        * 1e6
        * u_bus
    )

    temp_df["E_Total"] = (
        temp_df["E_PC"]
        + temp_df["E_VAN"]
        + temp_df["E_BUS"]
    ) * MULTI_VEHICLE_GAIN

    # =================================================
    # 全球汇总
    # =================================================
    global_total = temp_df["E_Total"].sum()

    all_detailed.append(temp_df)

    summary_data.append({

        "ISO_Level": iso_name,

        "Total_Recoverable_J": global_total,

        "Demand_Met":
            global_total >= LIGHT_DEMAND,

        # 保持旧字段名兼容
        "PC_Unit_E": u_pc,
        "Van_Unit_E": u_van,
        "Bus_Unit_E": u_bus,

        "Van/PC ratio":
            u_van / u_pc if u_pc > 0 else None,
    })

    print(
        f"{iso_name}: "
        f"全球总能量 = {global_total:.3e} J | "
        f"Van/PC = {u_van/u_pc:.2f}x | "
        f"满足照明需求: "
        f"{global_total >= LIGHT_DEMAND}"
    )

# =====================================================
# 5. 输出文件（保持旧版兼容）
# =====================================================

pd.concat(
    all_detailed,
    ignore_index=True
).to_excel(
    DATA_DIR / "Global_Vibration_Results_Final.xlsx",
    index=False
)

pd.DataFrame(
    summary_data
).to_excel(
    DATA_DIR / "Global_Vibration_Summary_by_ISO.xlsx",
    index=False
)

print("\n结果已保存：")
print(f"  {DATA_DIR / 'Global_Vibration_Results_Final.xlsx'}")
print(f"  {DATA_DIR / 'Global_Vibration_Summary_by_ISO.xlsx'}")

# =====================================================
# 6. 核心数值校验
# =====================================================

print("\n" + "=" * 55)
print("核心数值校验")
print("=" * 55)

# =====================================================
# 单阻尼器功率校验
# =====================================================

print("\n【单个阻尼器功率校验】")
print("条件：v = 30 m/s, ISO_C")

print(
    "文献基准："
    "PC~108W, Bus~559W, Truck~892W"
)

for vtype in ['PC', 'Van', 'Bus']:

    u = calculate_unit_energy(
        64e-6,
        v_base=30.0,
        vehicle_type=vtype
    )

    power_per_damper = (
        u
        * 30.0
        / 1000.0
        / 4.0
    )

    print(
        f"{vtype:4s}: "
        f"{power_per_damper:.1f} W"
    )

# =====================================================
# ISO_D 汇总分析
# =====================================================

print("\n【ISO_D 全球能量汇总】")

iso_d_idx = list(
    ISO_GQ_LEVELS.keys()
).index("ISO_D")

analysis_df = all_detailed[iso_d_idx]
sd = summary_data[iso_d_idx]

sum_pc = analysis_df["E_PC"].sum()
sum_van = analysis_df["E_VAN"].sum()
sum_bus = analysis_df["E_BUS"].sum()

grand = sum_pc + sum_van + sum_bus

print("\n单位能量耗散因子 (J/km)：")

print(f"PC   = {sd['PC_Unit_E']:.1f}")
print(f"Van  = {sd['Van_Unit_E']:.1f}")
print(f"Bus  = {sd['Bus_Unit_E']:.1f}")

print(
    f"\n货车/乘用车强度比："
    f"{sd['Van/PC ratio']:.2f} 倍"
)

print("\n全球贡献构成：")

print(f"PC   = {sum_pc/grand:.2%}")
print(f"Van  = {sum_van/grand:.2%}")
print(f"Bus  = {sum_bus/grand:.2%}")

print(
    f"\n全球总能量（ISO_D）="
    f"{sd['Total_Recoverable_J']:.3e} J"
)

print(
    f"全球照明需求="
    f"{LIGHT_DEMAND:.3e} J"
)

print(
    f"满足照明需求="
    f"{sd['Demand_Met']}"
)

# =====================================================
# 各等级供需匹配
# =====================================================

print("\n【各等级是否满足照明需求】")

for s in summary_data:

    mark = (
        "[满足]"
        if s["Demand_Met"]
        else "[不足]"
    )

    print(
        f"{s['ISO_Level']}: "
        f"{s['Total_Recoverable_J']:.3e} J  "
        f"{mark}"
    )

print("=" * 55)
