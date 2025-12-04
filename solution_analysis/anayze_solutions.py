import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

results = Path(__file__).resolve().parents[1]/"results"
data_dir = Path(__file__).resolve().parents[1]/"data"

def load_generator_parameters():
    """
    Columns in csv: 'generator_index', 'type', 'max_capacity', 
    'ramp_up', 'ramp_down', 'marginal_cost'
    """

    df = pd.read_csv(data_dir/"Parameters_generators_NL.csv", sep=";")
    df = df.sort_values("generator_index")

    generator_ids = df["generator_index"].astype(int).tolist()
    k = df["max_capacity"].to_numpy(dtype=float)
    gamma_u = df["ramp_up"].to_numpy(dtype=float)
    gamma_d = df["ramp_down"].to_numpy(dtype=float)
    c = df["marginal_cost"].to_numpy(dtype=float)

    return generator_ids, c, k, gamma_u, gamma_d


def load_solution(prefix):
    q = pd.read_csv(results/f"{prefix}_q.csv", index_col="hour")
    prices = pd.read_csv(results/f"{prefix}_prices.csv")["price"].to_numpy()
    obj = float(pd.read_csv(results/f"{prefix}_objective.csv")["objective"][0])
    return q, prices, obj

if __name__ == "__main__":
    q_nom, p_nom, obj_nom = load_solution("nominal")
    q_ouu, p_ouu, obj_ouu = load_solution("ouu")

    print("=== OBJECTIVES ===")
    print("Nominal:", obj_nom)
    print("OUU:", obj_ouu)
    print("Difference:", obj_ouu - obj_nom)

    price_diff = p_ouu - p_nom

    print("\n=== PRICE ANALYSIS ===")
    print("Average nominal price:", np.mean(p_nom))
    print("Average OUU price:", np.mean(p_ouu))
    print("Average price increase:", np.mean(price_diff))

    plt.figure(figsize=(10,5))
    plt.plot(p_nom, label="Nominal price", color='lightskyblue')
    plt.plot(p_ouu, label="OUU price", color='lightcoral')
    # plt.title("Day-Ahead Price Comparison")
    plt.xlabel("Hour")
    plt.xticks(np.arange(0, 24, 2))
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/price_comparison.png")
    plt.close()

    total_nom = q_nom.sum(axis=1)
    total_ouu = q_ouu.sum(axis=1)

    plt.figure(figsize=(10,5))
    plt.plot(total_nom, label="Nominal total production", color='lightskyblue')
    plt.plot(total_ouu, label="OUU total production", color='lightcoral')
    plt.plot(total_ouu - total_nom, label="Production difference", color='lightgreen')
    # plt.title("Total Production Comparison")
    plt.xlabel("Hour")
    plt.xticks(np.arange(0, 24, 2))
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/production_comparison.png")
    plt.close()

    gen_ids, c, k, gamma_u, gamma_d = load_generator_parameters()
    k = np.array(k)
    c = np.array(c)

    util_nom = q_nom / k
    util_ouu = q_ouu / k

    avg_util_nom = util_nom.mean(axis=0)
    avg_util_ouu = util_ouu.mean(axis=0)

    max_util_nom = util_nom.max(axis=0)
    max_util_ouu = util_ouu.max(axis=0)

    order = np.argsort(c)
    gen_sorted = np.array(gen_ids)[order]

    avg_nom_sorted = avg_util_nom[order]
    avg_ouu_sorted = avg_util_ouu[order]
    max_nom_sorted = max_util_nom[order]
    max_ouu_sorted = max_util_ouu[order]

    plt.figure(figsize=(10,5))
    x = np.arange(len(gen_sorted))

    plt.bar(x - 0.2, avg_nom_sorted, width=0.4, label="Nominal", color='lightskyblue')
    plt.bar(x + 0.2, avg_ouu_sorted, width=0.4, label="OUU", color='lightcoral')

    plt.xticks(x, gen_sorted, rotation=90)
    plt.ylabel("Average utilization")
    plt.xlabel("Generator")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/average_utilization.png")
    plt.close()


    