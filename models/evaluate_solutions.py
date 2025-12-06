import numpy as np
import pandas as pd 
from pathlib import Path

results_dir = Path(__file__).resolve().parents[1]/"results"
test_dir = Path(__file__).resolve().parents[1]/"test_sample"
data_dir = Path(__file__).resolve().parents[1]/"data"
output_dir = Path(__file__).resolve().parents[1]/"oos_results"

def evaluate_solution(q, test_demand, test_emissions, costs):
    N, T = test_demand.shape
    J = q.shape[1]

    total_costs = np.zeros(N)
    violations = np.zeros(N, dtype=bool)
    violating_hours = np.zeros(N, dtype=int)
    max_gaps = np.zeros(N)

    production_per_hour = q.sum(axis=1)

    for s in range(N):
        d = test_demand[s]
        e = test_emissions[s]

        gen_cost = 0.65 * np.sum(costs * q.sum(axis=0))
        emis_cost = 0.35 * np.sum(e * q.sum(axis=0))
        total_costs[s] = gen_cost + emis_cost

        gaps = d - production_per_hour
        violating_hours[s] = np.sum(gaps > 0)
        max_gaps[s] = np.maximum(gaps.max(), 0.0)
        violations[s] = violating_hours[s] > 0
    
    return {
        "avg_cost": np.mean(total_costs),
        "std_cost": np.std(total_costs),
        "violation_rate": np.mean(violations),
        "total_costs": total_costs,
        "violations": violations,
        "violating_hours": violating_hours,
        "max_gaps": max_gaps,
    }

if __name__ == "__main__":
    test_demand = np.load(test_dir/"test_demand.npy")
    test_em = np.load(test_dir/"test_emissions.npy")

    params = pd.read_csv(data_dir/"Parameters_generators_NL.csv", sep=";")
    costs = params["marginal_cost"].to_numpy()

    nom_q = pd.read_csv(results_dir/"nominal_q.csv", index_col=0).to_numpy()
    ouu_q = pd.read_csv(results_dir/"ouu_q.csv", index_col=0).to_numpy()

    out_nom = evaluate_solution(nom_q, test_demand, test_em, costs)
    out_ouu = evaluate_solution(ouu_q, test_demand, test_em, costs)

    summary = pd.DataFrame([
        {
            "model": "nominal",
            "avg_cost": round(out_nom["avg_cost"], 4),
            "std_cost": round(out_nom["std_cost"], 4),
            "violation_rate": round(out_nom["violation_rate"],4),
        },
        {
            "model": "ouu",
            "avg_cost": round(out_ouu["avg_cost"], 4),
            "std_cost": round(out_ouu["std_cost"], 4),
            "violation_rate": round(out_ouu["violation_rate"], 4),
        }
    ])

    summary.to_csv(output_dir/"summary_results.csv", index=False, float_format="%.4f")

    pd.DataFrame({
        "total_cost": out_nom["total_costs"],
        "violation": out_nom["violations"],
        "violating_hours": out_nom["violating_hours"],
        "max_gap": out_nom["max_gaps"],
    }).to_csv(output_dir / "nominal_oos_details.csv", index=False)

    pd.DataFrame({
        "total_cost": out_ouu["total_costs"],
        "violation": out_ouu["violations"],
        "violating_hours": out_ouu["violating_hours"],
        "max_gap": out_ouu["max_gaps"],
    }).to_csv(output_dir / "ouu_oos_details.csv", index=False)

    print("Saved out-of-sample evaluation results to:", output_dir)
    print(summary)

