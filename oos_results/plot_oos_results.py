import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path

fig_dir = Path(__file__).resolve().parents[1]/"figures"
this_dir = Path(__file__).resolve().parent

nom = pd.read_csv(this_dir/"nominal_oos_details.csv")
ouu = pd.read_csv(this_dir/"ouu_oos_details.csv")

# distribution of number of violating hours
plt.figure(figsize=(8,5))
plt.hist(nom["violating_hours"], bins=range(0, 26), align="left", color="lightskyblue")
plt.xlabel("Violating Hours per Scenario")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.savefig(fig_dir / "nominal_violating_hours.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,5))
plt.hist(ouu["violating_hours"], bins=range(0, 26), align="left", color="lightcoral")
plt.xlabel("Violating Hours per Scenario")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.savefig(fig_dir / "ouu_violating_hours.png", dpi=200, bbox_inches="tight")
plt.close()

# distribution of maximum gap
plt.figure(figsize=(8,5))
plt.hist(nom["max_gap"], bins=40, color="lightskyblue")
plt.xlabel("Maximum gap (MW)")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.savefig(fig_dir / "nominal_max_gap.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,5))
plt.hist(ouu["max_gap"], bins=40, color="lightcoral")
plt.xlabel("Maximum gap (MW)")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.savefig(fig_dir / "ouu_max_gap.png", dpi=200, bbox_inches="tight")
plt.close()