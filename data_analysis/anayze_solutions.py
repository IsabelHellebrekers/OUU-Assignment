import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

results = Path(__file__).resolve().parents[1]/"results"

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
    plt.plot(p_nom, label="Nominal price")
    plt.plot(p_ouu, label="OUU price")
    plt.title("Day-Ahead Price Comparison")
    plt.xlabel("Hour")
    plt.ylabel("Price (€/MWh)")
    plt.legend()
    plt.grid(True)
    plt.show()

    total_nom = q_nom.sum(axis=1)
    total_ouu = q_ouu.sum(axis=1)

    plt.figure(figsize=(10,5))
    plt.plot(total_nom, label="Nominal total production")
    plt.plot(total_ouu, label="OUU total production")
    plt.title("Total Production Comparison")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.show()

    