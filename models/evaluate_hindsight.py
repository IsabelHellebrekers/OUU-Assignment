import numpy as np
import pandas as pd
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

test_dir = Path(__file__).resolve().parents[1]/"test_sample"
data_dir = Path(__file__).resolve().parents[1]/"data"
output_dir = Path(__file__).resolve().parents[1]/"oos_results"

def solve_hindsight_model(demand_s, emissions_s, costs, k, gamma_u, gamma_d):
    T = len(demand_s)
    J = len(costs)

    m = gp.Model("hindsight")
    m.Params.OutputFlag = 0

    hours = range(T)
    generators = range(J)

    q = m.addVars(T, J, lb=0.0, name="q")

    for t in hours:
        for j in generators:
            q[t,j].ub = float(k[j])
    
    for j in generators:
        for t in range(T-1):
            m.addConstr(q[t+1,j] - q[t,j] <= gamma_u[j] * k[j])
            m.addConstr(q[t+1,j] - q[t,j] >= gamma_d[j] * k[j])
    
    for t in hours:
        m.addConstr(gp.quicksum(q[t,j] for j in generators) >= demand_s[t])
    
    cost_term = gp.quicksum(costs[j] * q[t,j] for t in hours for j in generators)
    emis_term = gp.quicksum(emissions_s[j] * q[t,j] for t in hours for j in generators)
    m.setObjective(0.65 * cost_term + 0.35 * emis_term, GRB.MINIMIZE)

    m.optimize()

    return m.ObjVal 

if __name__ == "__main__":
    test_demand = np.load(test_dir/"test_demand.npy")
    test_em = np.load(test_dir/"test_emissions.npy")

    params = pd.read_csv(data_dir/"Parameters_generators_NL.csv", sep=";")
    costs = params["marginal_cost"].to_numpy()
    k = params["max_capacity"].to_numpy()
    gamma_u = params["ramp_up"].to_numpy()
    gamma_d = params["ramp_down"].to_numpy()

    N = test_demand.shape[0]

    hindsight_objs = np.zeros(N)

    print("Running hindsight optimization for all test scenarios...")

    for s in range(N):
        hindsight_objs[s] = solve_hindsight_model(
            demand_s=test_demand[s],
            emissions_s=test_em[s],
            costs=costs,
            k=k,
            gamma_u=gamma_u,
            gamma_d=gamma_d,
        )

        if (s+1) % 200 == 0:
            print(f"  solved {s+1}/{N} models")

    pd.DataFrame({"hindsight_obj": hindsight_objs}).to_csv(
        output_dir / "hindsight_objectives.csv", index=False
    )

    print("Results saved to oos_results/hindsight_objectives.csv")