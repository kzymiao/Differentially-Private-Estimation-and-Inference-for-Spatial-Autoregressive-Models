import numpy as np
import random
import scipy
from scipy import stats
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from random import randint, sample
import networkx as nx
import time
import copy
import warnings
import pandas as pd
from generation import generate
from function_qmle import qmle_estimate, qmle_inference, f_lambda_ture_sig
from function_qmle_dp import qmle_dp_estimate, qmle_dp_inference

warnings.filterwarnings("ignore")

np.random.seed(123)
R = 500
Distri_e = ['N']
Type_w = ['PowerLaw','Dyad', 'SBM']  # 'PowerLaw', 'Dyad', 'SBM'
p = 2
distri_e = 'N'

beta = np.array([0.5, 0.5]).reshape((p, 1))
rho = 0.2
tbeta = np.append(beta, [rho])  # shape: (p+1,)

columns = ['noise(E,epsilon)', 'N', 'type', 
           'RMSE_beta1', 'RMSE_beta2', 'RMSE_rho', 
           'bias_beta1', 'bias_beta2', 'bias_rho',
           'tRMSE_beta1', 'tRMSE_beta2', 'tRMSE_rho', 
           'tbias_beta1', 'tbias_beta2', 'tbias_rho', 
           'estimate_time_dp', 'estimate_time_nondp', 'W-1-norm']
results_df = pd.DataFrame(columns=columns)


def calc_rmse(preds, true):
    """Root Mean Square Error (vectorized over parameters)"""
    return np.sqrt(np.mean((preds - true) ** 2, axis=0))


def calc_bias(preds, true):
    """Empirical Bias (vectorized over parameters)"""
    return np.abs(np.mean(preds - true, axis=0))


for ii in range(len(Type_w)):
    type_w = Type_w[ii]
    
    N = 3000
    epsi = 2
    delta = 10 / N ** 1.1
    pro = 0.5
    
    # Storage for DP estimates: shape (R, p+1)
    guji = np.zeros((R, p + 1))
    # Storage for non-DP estimates: shape (R, p+1)
    tguji = np.zeros((R, p + 1))
    
    # Timing and coverage probability accumulators
    time_dp_total = 0
    time_nondp_total = 0
    CP = np.zeros(p + 1)   # Coverage probability for DP
    tCP = np.zeros(p + 1)  # Coverage probability for non-DP
    
    for r in range(R):
        X, Y, W = generate(N, p, beta, rho, type_w)
        
        # ========== DP Estimation ==========
        starttime = time.time()
        hrbeta, hsig, T, var_b, var_r, cl2 = qmle_dp_estimate(X, Y, W, epsi * pro, delta * pro)
        endtime = time.time()
        time_dp_total += (endtime - starttime)
        guji[r, :] = hrbeta.flatten()
        
        # DP Inference for coverage probability
        SE = qmle_dp_inference(X, Y, W, hrbeta, hsig, var_b, var_r, cl2, epsi * (1 - pro), delta * (1 - pro))
        CI1 = hrbeta - 1.96 * SE[0:p+1]
        CI2 = hrbeta + 1.96 * SE[0:p+1]
        for i in range(p + 1):
            if tbeta[i] >= CI1[i] and tbeta[i] <= CI2[i]:
                CP[i] += 1
        
        # ========== Non-DP Estimation ==========
        starttime = time.time()
        thrbeta, thsig = qmle_estimate(X, Y, W, T)
        endtime = time.time()
        time_nondp_total += (endtime - starttime)
        tguji[r, :] = thrbeta.flatten()
        
        # Non-DP Inference for coverage probability
        tSE = qmle_inference(X, Y, W, hrbeta, hsig)
        tCI1 = thrbeta - 1.96 * tSE[0:p+1]
        tCI2 = thrbeta + 1.96 * tSE[0:p+1]
        for i in range(p + 1):
            if tbeta[i] >= tCI1[i] and tbeta[i] <= tCI2[i]:
                tCP[i] += 1
        
        # Progress print
        print('\r Current progress: {:^3.0f}%'.format(((r + 1) / R * 100)), end='')
    
    # ========== Compute Evaluation Metrics (Vectorized) ==========
    # DP metrics
    rmse_dp = calc_rmse(guji, tbeta)
    bias_dp = calc_bias(guji, tbeta)
    
    # Non-DP metrics
    rmse_nondp = calc_rmse(tguji, tbeta)
    bias_nondp = calc_bias(tguji, tbeta)
    
    # Coverage probabilities
    CP_ratio = CP / R
    tCP_ratio = tCP / R
    
    # Average computation times
    avg_time_dp = time_dp_total / R
    avg_time_nondp = time_nondp_total / R
    
    # Network density
    density = np.mean([len(np.nonzero(W.toarray())[0]) / N / (N - 1) for _ in range(R)])
    
    # ========== Print Summary (similar to reference code) ==========
    print("\n" + "=" * 60)
    print("Simulation Results Summary")
    print("=" * 60)
    print(f"\nTrue Parameters: {tbeta.flatten()}")
    print(f"Parameter Dimensions: " + ", ".join([f"beta{i+1}" for i in range(p)] + ["rho"]))
    
    print("\n--- DP Method (ε=1.0, δ=10/N^1.1) ---")
    print(f"RMSE:      {rmse_dp}")
    print(f"Bias:      {bias_dp}")
    print(f"CP:        {CP_ratio}")
    
    print("\n--- Non-DP Method ---")
    print(f"RMSE:      {rmse_nondp}")
    print(f"Bias:      {bias_nondp}")
    print(f"CP:        {tCP_ratio}")
    
    print(f"\n--- Computation Time ---")
    print(f"DP Estimate Time (avg):      {avg_time_dp:.4f} sec")
    print(f"Non-DP Estimate Time (avg):  {avg_time_nondp:.4f} sec")
    
    print(f"\n--- Network Properties ---")
    print(f"W-1-norm:  {np.linalg.norm(W.toarray(), ord=1, axis=None):.4f}")
    print(f"Density:   {density:.4f}")
    
