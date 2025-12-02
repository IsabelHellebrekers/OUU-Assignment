import gurobipy as gp
from gurobipy import GRB
import numpy as np

def build_base_model(c, k, gamma_u, gamma_d, demand, emissions, model_name="base"):
    demand = np.asarray(demand)
    c = np.asarray(c)
    k = np.asarray(k)
    gamma_u = np.asarray(gamma_u)
    gamma_d = np.asarray(gamma_d)
    emissions = np.asarray(emissions)

    T = demand.shape[0]
    J = c.shape[0]

    m = gp.Model(model_name)

    hours = range(T)
    generators = range(J)

    # decision variables (production of generator j in hour t)
    q = m.addVars(hours, generators, lb=0.0, name="q")

    # 0 <= q_{t,j} <= k_j
    for j in generators:
        for t in hours:
            q[t, j].ub = float(k[j])
    
    # ramp-up and ramp-down constraints
    for j in generators:
        for t in range(T - 1):
            m.addConstr(
                q[t + 1, j] - q[t, j] <= gamma_u[j] * k[j],
                name=f"ramp_up_{t}_{j}"
            )
            m.addConstr(
                q[t + 1, j] - q[t, j] >= gamma_d[j] * k[j],
                name=f"ramp_down_{t}_{j}"
            )
    
    # demand constraints
    for t in hours:
        m.addConstr(
            gp.quicksum(q[t, j] for j in generators) >= demand[t],
            name=f"demand_{t}"
        )

    # objective (long-term average cost minimization)
    cost_term = gp.quicksum(c[j] * q[t, j] for t in hours for j in generators)
    emission_term = gp.quicksum(emissions[j] * q[t, j] for t in hours for j in generators)
    m.setObjective(0.65 * cost_term + 0.35 * emission_term, GRB.MINIMIZE)

    return m, q
