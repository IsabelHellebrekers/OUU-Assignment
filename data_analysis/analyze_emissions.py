import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.stats import probplot

from load_data import load_emission_history

# --- LOAD YOUR DATA -----------------------------------------------------
df = load_emission_history()

emission_cols = [col for col in df.columns if col.startswith("Emission_rate_g")]
E = df[emission_cols].astype(float).to_numpy()   # shape (N, J)

# --- Compute mean (mu) and sample std (sigma) ---------------------------
mu = E.mean(axis=0)            # (J,)
sigma = E.std(axis=0, ddof=1)  # (J,)

def emissions_mean() :
    return mu

print("mu:", mu)
print("sigma:", sigma)

# --- Residuals ----------------------------------------------------------
residuals = E - mu  # (N, J)

# --- Standardized residuals, careful with sigma = 0 ---------------------
std_residuals = np.full_like(E, np.nan, dtype=float)  # start with NaNs

nonzero_mask = sigma > 0
std_residuals[:, nonzero_mask] = residuals[:, nonzero_mask] / sigma[nonzero_mask]

J = E.shape[1]

ncols = 4
nrows = math.ceil(J / ncols)

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4 * ncols, 3 * nrows),
                         sharex=False, sharey=False)

axes = axes.ravel()

for j in range(J):
    ax = axes[j]
    data = std_residuals[:, j]

    if sigma[j] == 0:
        # Degenerate case: no variability, emissions always zero
        ax.text(0.5, 0.5, "σ=0\n(no variability)",
                ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Histogram
        ax.hist(data, bins=30, density=True, alpha=0.6)

        # Standard normal pdf over same range
        x = np.linspace(np.nanmin(data), np.nanmax(data), 200)
        ax.plot(x, norm.pdf(x), linewidth=1)

    ax.set_title(f"g{j+1}")

# Turn off any unused subplots
for k in range(J, nrows * ncols):
    axes[k].axis("off")

fig.suptitle("Standardized residuals + N(0,1) pdf for all generators",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.title("Standardized emission residuals")
plt.tight_layout()
plt.savefig("figures/residuals_emissions.png")
plt.close()


# QQ Plot 

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4 * ncols, 3 * nrows),
                         sharex=False, sharey=False)

axes = axes.ravel()

for j in range(J):
    ax = axes[j]
    data = std_residuals[:, j]

    if np.all(np.isnan(data)):
        ax.text(0.5, 0.5, "σ=0\n(no variability)",
                ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        probplot(data, dist="norm", plot=ax)

    ax.set_title(f"g{j+1}")

for k in range(J, nrows * ncols):
    axes[k].axis("off")

fig.suptitle("Q–Q plots vs N(0,1) for all generators",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.title("Q-Q plots vs N(0,1) for all generators")
plt.tight_layout()
plt.savefig("figures/qq_plots_emissions.png")
plt.close()
