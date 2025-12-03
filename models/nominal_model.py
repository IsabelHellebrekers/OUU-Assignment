import gurobipy as gp
from gurobipy import GRB
import numpy as np

from data_utils import (
    load_demand_forecast,
    load_generator_parameters,
    compute_emission_means,
)

def build_nominal_model(model_name="nominal"):
    generator_ids, c, k, gamma_u, gamma_d = load_generator_parameters()
    mean_emissions = compute_emission_means()
    forecast = load_demand_forecast()

    T = len(forecast)
    J = len(generator_ids)
    
    m = gp.Model(model_name)
    hours = range(T)
    generators = range(J)

    q = m.addVars(hours, generators, lb=0.0, name="q")

    for j in generators:
        for t in hours:
            q[t, j].ub = float(k[j])

    for j in generators:
        for t in range(T - 1):
            m.addConstr(
                q[t + 1, j] - q[t, j] <= gamma_u[j] * k[j],
                name=f"ramp_up_{t}_{j}",
            )
            m.addConstr(
                q[t + 1, j] - q[t, j] >= gamma_d[j] * k[j],
                name=f"ramp_down_{t}_{j}",
            )

    for t in hours:
        m.addConstr(
            gp.quicksum(q[t, j] for j in generators) >= forecast[t],
            name=f"demand_{t}",
        )
    
    cost_term = gp.quicksum(
        c[j] * q[t, j] for t in hours for j in generators
    )
    emission_term = gp.quicksum(
        mean_emissions[j] * q[t, j] for t in hours for j in generators
    )

    m.setObjective(
        0.65 * cost_term + 0.35 * emission_term, GRB.MINIMIZE
    )

    info = {
        "generator_ids": generator_ids,
        "c": c,
        "k": k,
        "gamma_u": gamma_u,
        "gamma_d": gamma_d,
        "forecast": forecast,
        "mean_emissions": mean_emissions,
    }

    return m. q, info

def solve_nominal():
    m, q, info = build_nominal_model()

    m.Params.OutputFlag = 1
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimization failed (status {m.Status})")
    
    T = len(info["forecast"])
    J = len(info["generator_ids"])

    q_opt = np.zeros((T, J))
    for t in range(T):
        for j in range(J):
            q_opt[t, j] = q[t, j].X

    return q_opt, m.ObjVal, info

if __name__ == "__main__":
    print("Running nominal model...")

    q_opt, obj_val, info = solve_nominal()

    print("\nNominal objective value: " + obj_val)

    print("\nOptimal production per hour (MW):")
    T, J = q_opt.shape
    gen_ids = info["generator_ids"]
    for t in range(T):
        print(f"\nHour {t}:")
        for j in range(J):
            print(f"  Generator {gen_ids[j]}: {q_opt[t, j]:.2f}")
        print(f"  Total: {q_opt[t, :].sum():.2f}")
