import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_emission_history

# --- LOAD YOUR DATA -----------------------------------------------------
df = load_emission_history()

# If df contains only emission columns:
mu = df.mean(axis=0).to_numpy()          # mean per column
sigma = df.std(axis=0, ddof=1).to_numpy()  # sample std per column

print("mu:", mu) 
print("sigma:", sigma)

# Plot the mu and sigma ---------------------------------------------------
J = len(mu)
generators = np.arange(1, J+1)

plt.figure(figsize=(12, 6))
plt.bar(generators - 0.2, mu, width=0.4, label="Mean (μ)")
plt.bar(generators + 0.2, sigma, width=0.4, label="Std dev (σ)")

plt.xticks(generators)
plt.xlabel("Generator j")
plt.ylabel("Value")
plt.title("Mean and Standard Deviation of Emissions per Generator")
plt.legend()
plt.tight_layout()
plt.show()