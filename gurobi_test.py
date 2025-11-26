import gurobipy as gp

try:
    m = gp.Model("test")

    x = m.addVar(name="x")

    m.setObjective(x, gp.GRB.MAXIMIZE)

    m.optimize()

    print("Gurobi werkt ✔️")
    print("Oplossing:", x.X)

except gp.GurobiError as e:
    print("Gurobi fout:", e)

except Exception as e:
    print("Andere fout:", e)
