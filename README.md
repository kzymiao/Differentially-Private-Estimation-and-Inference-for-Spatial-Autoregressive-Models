#Differentially Private Estimation and Inference for Spatial Autoregressive Models

> Kong, Z., Tao, Y., Huang, D., and Cai, Z. (2025). Differentially Private Estimation and Inference for Spatial Autoregressive Models

## Overview
This repository provides the Python implementation for a novel Differentially Private Quasi-Maximum Likelihood Estimator (DP-QMLE) for SAR models. This repository includes the code for private parameter estimation (Algorithm 1), differentially private statistical inference and confidence interval construction (Algorithm 2), as well as comparative baseline methods from recent literature.


## Repository Structure

|File name| Description |
|-------------|---------------|
|**`generation.py`**| Handles the generation of synthetic network topologies (Power-Law, Dyad Independence, and Stochastic Block Models) and generates SAR data according to assumptions. |
|**`function_qmle.py`**| Contains the non-private standard QMLE estimation and inference functions, serving as the utility baseline. |
|**`function_qmle_dp.py`**| Contains the core proposed methods: iterative DP-QMLE parameter estimation (qmle_dp_estimate, Algorithm 1) and the differentially private statistical inference procedure (qmle_dp_inference, Algorithm 2). |
|**`function_wangsong.py`**| Implements the baseline methods from Wang & Song (2024), including Least Squares Estimation (LSE) with Functional Mechanism using both Laplace and Gaussian noise distributions. |
|**`main.py`**| The main script to run comprehensive Monte Carlo simulations for the proposed DP-QMLE method. It evaluates and outputs performance metrics including Root Mean Square Error (RMSE), Bias, Coverage Probability (CP), and computation time. |
|**`main_compare.py`**| A comparative simulation script designed to benchmark the proposed DP-QMLE method against the Wang & Song (2024) methods under equivalent privacy budgets. |