import numpy as np
from generation import generate_compare
from function_qmle_dp import qmle_dp_estimate,qmle_dp_inference
from function_wangsong import lse_dp_estimate,lse_dp_gaussian_estimate

# =====================================================================
# 1. Parameter Initialization & Simulation Setup
# =====================================================================
np.random.seed(1234)
R = 100                 # Number of simulation replications
N = 1000                # Sample size (number of nodes in the network)
p = 2                   # Dimension of covariates
rho = 0.2               # True spatial autoregressive parameter
beta = np.array([0.3, 0.3]).reshape((p, 1))  # True covariate coefficients
tbeta = np.append(beta, [rho])               # True parameter vector: [beta1, beta2, rho]

# Storage lists for parameter estimates across R replications
guji = []           # Wang & Song Laplace DP estimates (Pure epsilon-DP)
guji_gaussian = []  # Wang & Song Gaussian DP estimates (Ablated to (epsilon, delta)-DP)
tguji = []          # Non-DP Least Squares estimates (Baseline utility)
guji_our = []       # Our QMLE Gradient DP estimates

print(f"Starting simulation: N={N}, Replications={R}")

for r in range(R):
    # Generate synthetic network data
    X, Y, W = generate_compare(N, p, beta, rho, 'PowerLaw')
    
    # Privacy budget allocation
    epsi = 1
    delta = 10 / N**1.1 

    # Wang & Song (Functional Mechanism with Laplace Noise)
    # Calibrated to L1-sensitivity for pure epsilon-DP
    hrbeta, _, _ = lse_dp_estimate(X, Y, W, epsi, c3=0.8*np.log(N))
    guji.append(hrbeta.flatten())
    
    # Wang & Song Ablation (Functional Mechanism with Gaussian Noise)
    # Calibrated to L2-sensitivity for approximate (epsilon, delta)-DP
    hrbeta_gaussian, _, _ = lse_dp_gaussian_estimate(X, Y, W, epsi, delta, c3=0.8*np.log(N))
    guji_gaussian.append(hrbeta_gaussian.flatten())
    

    # Our Proposed Method (Iterative Gradient Perturbation QMLE)
    hrbeta_our, hsig, T, var_b, var_r, cl2 = qmle_dp_estimate(X, Y, W, epsi, delta)
    guji_our.append(hrbeta_our.flatten())
    
    # Print progress dynamically
    print('\r Current progress: {:^3.0f}%'.format(((r + 1) / R * 100)), end='')

# Convert lists to numpy arrays for vectorized metric computation
guji = np.array(guji)
guji_gaussian = np.array(guji_gaussian)
guji_our = np.array(guji_our)

# =====================================================================
# 2. Evaluation Metrics Definition
# =====================================================================
def calc_rmse(preds, true):
    # Root Mean Square Error
    return np.sqrt(np.mean((preds - true)**2, axis=0))

def calc_bias(preds, true):
    # Empirical Bias
    return np.abs(np.mean(preds - true, axis=0))

def calc_se(preds):
    # Empirical Standard Error (using sample standard deviation, ddof=1)
    return np.std(preds, axis=0, ddof=1)

# Compute metrics for all evaluated methods
rmse_laplace = calc_rmse(guji, tbeta)
rmse_gaussian = calc_rmse(guji_gaussian, tbeta)
rmse_our = calc_rmse(guji_our, tbeta)

bias_laplace = calc_bias(guji, tbeta)
bias_gaussian = calc_bias(guji_gaussian, tbeta)
bias_our = calc_bias(guji_our, tbeta)

se_laplace = calc_se(guji)
se_gaussian = calc_se(guji_gaussian)
se_our = calc_se(guji_our)

# =====================================================================
# 3. Print Summary of Simulation Results
# =====================================================================
print("\n" + "="*60)
print("Simulation Results Summary")
print("="*60)

print(f"\nTrue Parameters: {tbeta.flatten()}")
print(f"Parameter Dimensions: beta1, beta2, rho")

print("\n--- Wang & Song Laplace DP (ε=1.0) ---")
print(f"RMSE:      {rmse_laplace}")
print(f"Bias:      {bias_laplace}")
print(f"Std. Err:  {se_laplace}")

print("\n--- Wang & Song Gaussian DP (ε=1.0, δ=10/N^1.1) ---")
print(f"RMSE:      {rmse_gaussian}")
print(f"Bias:      {bias_gaussian}")
print(f"Std. Err:  {se_gaussian}")

print("\n--- Our Gradient DP Method (ε=1.0, δ=10/N^1.1) ---")
print(f"RMSE:      {rmse_our}")
print(f"Bias:      {bias_our}")
print(f"Std. Err:  {se_our}")