# -*- coding: utf-8 -*-
"""
Generate Figure 4: Annual Electrical Generation Potential under
System-Level Energy Recovery Efficiency Scenarios across ISO Road Classes.

This script reproduces Figure 4 of the manuscript:

    Global assessment of recoverable road vibration energy potential
    for sustainable roadside electricity supply.

Input
-----
None.

The annual recoverable mechanical vibration energy for each ISO 8608
road class has been precomputed using the global vibration energy
assessment model (全球振动能潜能计算.py) and is embedded below.

Output
------
Output/Fig_Energy_Self_Sufficiency_v7.png

Author
------
Li Ruimin

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# Plot configuration
# ==========================================================

plt.rcParams["font.sans-serif"] = ["Arial"]

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

FIG_SIZE = (13, 7)
OUTPUT_DPI = 600

COLORS = [
    "#F28E2B",
    "#E15759",
    "#76B7B2",
]

LEGEND_LABELS = [
    "Moderate (5%)",
    "Good (15%)",
    "High (25%)",
]

# ==========================================================
# Input data
# ==========================================================

ISO_LEVELS = [
    "ISO A",
    "ISO B",
    "ISO C",
    "ISO D",
    "ISO E",
    "ISO F",
    "ISO G",
    "ISO H",
]

# ----------------------------------------------------------
# Annual recoverable mechanical vibration energy.
#
# These values were generated using the global vibration
# energy assessment model
#
#     全球振动能潜能计算.py
#
# and represent the total annual recoverable mechanical
# vibration energy before applying the system-level
# energy recovery efficiency.
#
# Unit:
#     J yr^-1
# ----------------------------------------------------------

E_TOTAL = np.array([
    2.979e16,
    1.192e17,
    4.766e17,
    1.906e18,
    7.626e18,
    3.050e19,
    1.220e20,
    4.880e20,
])

# ----------------------------------------------------------
# Estimated annual global roadside lighting demand.
#
# 199.37 TWh yr^-1
# converted to Joules.
# ----------------------------------------------------------

GLOBAL_DEMAND_TWH = 199.37

GLOBAL_DEMAND_J = GLOBAL_DEMAND_TWH * 1e12 * 3600

# ----------------------------------------------------------
# System-level energy recovery efficiency scenarios.
# ----------------------------------------------------------

EFFICIENCY_SCENARIOS = {
    "Moderate": 0.05,
    "Good": 0.15,
    "High": 0.25,
}

# ==========================================================
# Output directory
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "Fig_Energy_Self_Sufficiency_v7.png"

# ==========================================================
# Figure generation
# ==========================================================

def main():

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=300)

    x = np.arange(len(ISO_LEVELS))
    width = 0.25

    # ------------------------------------------------------
    # Global roadside lighting demand
    # ------------------------------------------------------

    ax.axhline(
        GLOBAL_DEMAND_J,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Global Roadside Lighting Demand",
        zorder=1,
    )

    ax.text(
        7.4,
        GLOBAL_DEMAND_J * 1.8,
        "Global Roadside Lighting Demand\n199.37 TWh yr$^{-1}$",
        ha="right",
        fontsize=14,
        fontweight="bold",
        color="black",
    )

    # ------------------------------------------------------
    # Energy recovery scenarios
    # ------------------------------------------------------

    for i, (_, eta) in enumerate(EFFICIENCY_SCENARIOS.items()):

        annual_generation = E_TOTAL * eta

        ax.bar(
            x + (i - 1) * width,
            annual_generation,
            width,
            color=COLORS[i],
            edgecolor="white",
            linewidth=0.6,
            label=LEGEND_LABELS[i],
            zorder=2,
        )

    # ------------------------------------------------------
    # Axis configuration
    # ------------------------------------------------------

    ax.set_yscale("log")
    ax.set_ylim(1e13, 2e21)

    ax.set_xticks(x)
    ax.set_xticklabels(ISO_LEVELS)

    ax.set_xlabel("Road Roughness Class (ISO 8608)")
    ax.set_ylabel("Annual Electrical Energy Potential (J yr$^{-1}$)")

    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=1,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.grid(
        True,
        which="both",
        linestyle="-",
        alpha=0.25,
        zorder=0,
    )

    ax.set_axisbelow(True)

    plt.title(
        "Annual Electrical Generation Potential under Different\n"
        "System-Level Energy Recovery Efficiency Scenarios\n"
        "across ISO Road Classes",
        pad=20,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    print(f"Figure saved to: {OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    main()
