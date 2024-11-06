import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma
from tqdm import tqdm

# Set up seed for reproducibility
np.random.seed(42)

# Define cities in the Bay Area and create synthetic economic data
cities = [
    "Oakland", "Castro Valley", "Berkeley", "Alameda", "San Leandro", 
    "Richmond", "Emeryville", "Fremont", "Berryessa"
]

# Generate synthetic data: assume GDP values in billions, employment rates, and growth rates
data = pd.DataFrame({
    "City": cities,
    "GDP": np.random.normal(50, 10, len(cities)),  # GDP in billions
    "Employment_Rate": np.random.normal(0.95, 0.02, len(cities)),  # Employment rate between 90-100%
    "Population_Growth": np.random.normal(0.02, 0.005, len(cities)),  # Population growth rate ~2%
})

# Set hyperpriors for the overall Bay Area (second level in hierarchy)
bay_area_mu_gdp = 50
bay_area_sigma_gdp = 10

# Define ranges for parameter estimation (reduced resolution for efficiency)
mu_gdp_range = np.linspace(45, 55, 50)
sigma_gdp_range = np.linspace(5, 15, 50)

# Posterior storage for each parameter
posterior_gdp = np.zeros((len(mu_gdp_range), len(sigma_gdp_range), len(cities)))

# Likelihood calculation function
def calculate_likelihood(mu, sigma, observed_value):
    return norm(mu, sigma).pdf(observed_value)

# Calculate posterior for GDP for each city
for k, city in enumerate(data["City"]):
    for i, mu in enumerate(tqdm(mu_gdp_range, desc=f"Calculating Posterior for {city} - GDP")):
        for j, sigma in enumerate(sigma_gdp_range):
            prior_gdp = norm(bay_area_mu_gdp, bay_area_sigma_gdp).pdf(mu) * invgamma(3).pdf(sigma)
            likelihood_gdp = calculate_likelihood(mu, sigma, data["GDP"][k])
            posterior_gdp[i, j, k] = prior_gdp * likelihood_gdp

# Normalize posterior to sum to 1 for probability interpretation
posterior_gdp /= np.sum(posterior_gdp, axis=(0, 1), keepdims=True)

# Plot posterior for GDP of each city
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.ravel()  # Flatten the array for iteration

for k, city in enumerate(cities):
    ax = axs[k]
    cont = ax.contourf(sigma_gdp_range, mu_gdp_range, posterior_gdp[:, :, k], cmap="viridis")
    ax.set_title(f"Posterior Distribution for {city} GDP")
    ax.set_xlabel("Standard Deviation (Sigma) for GDP")
    ax.set_ylabel("Mean (Mu) for GDP")
    fig.colorbar(cont, ax=ax, label="Posterior Probability Density")

plt.tight_layout()
plt.show()
