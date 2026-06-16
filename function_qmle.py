
import numpy as np
import scipy
from scipy import stats
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from random import randint, sample
import networkx as nx
import time
import warnings

def f_lambda_ture_sig(hbeta, hrho, Y, X, W):  
    """
    Calculate the first-order derivatives (score/gradient) and estimate the variance sigma^2 
    for the standard SAR QMLE method.
    
    Parameters:
    :hbeta: The estimate of beta in the current iterative step.
    :hrho: The estimate of rho in the current iterative step.
    :Y: The observed response vector.
    :X: The covariate matrix.
    :W: The spatial weight matrix.
    
    Returns:
    :f: The combined gradient vector for beta and rho.
    :hsig: The estimated error variance (sigma^2).
    """
    N = len(Y)
    p = len(X[1])
    
    # Calculate the spatial filter matrix: S = I - rho * W
    hS = np.eye(N) - hrho * W.toarray()
    hS1 = np.linalg.inv(hS)
    
    # Calculate the residuals: V = S * Y - X * beta
    V = hS.dot(Y) - (X.dot(hbeta)).reshape(N, 1)
    
    # Estimate the error variance sigma^2: (V^T * V) / N
    hsig = (V.T.dot(V) / N)[0, 0]
    s1 = 1.0 / hsig         
    
    # Calculate the first-order derivative (gradient) of the log-likelihood w.r.t rho
    f_r = -((W.toarray().dot(hS1)).trace()) + 1 / hsig * V.T.dot(W.toarray()).dot(Y)
    
    # Calculate the first-order derivative (gradient) of the log-likelihood w.r.t beta
    f_b = 1 / hsig * X.T.dot(V)
    
    f = np.vstack((f_b, f_r))
    return f, hsig


def qmle_estimate(X, Y, W, T):
    """
    Obtain the QMLE parameter estimates using a gradient ascent algorithm.
    
    Parameters:
    :X: The covariate matrix.
    :Y: The observed response vector.
    :W: The spatial weight matrix.
    :T: Maximum number of iterations.
    
    Returns:
    :hrbeta: The final estimated parameter vector [beta; rho].
    :hsig: The final estimate of the error variance sigma^2.
    """
    N = len(Y)
    p = len(X[1])
    
    # Initialize parameters: all set to 0.01
    hrbeta = (0.01 * np.ones(p + 1)).reshape((p + 1, 1))
    K = T
    k = 0
    e = 10**(-8) # Tolerance for the stopping criterion
    
    f, hsig = f_lambda_ture_sig(hrbeta[0:p,], hrbeta[p,], Y, X, W)
    
    while np.linalg.norm(f) > e and k <= K:
        lr = 1 / N 
        
        hrbeta += lr * f
        k += 1
        
        f, hsig = f_lambda_ture_sig(hrbeta[0:p,], hrbeta[p,], Y, X, W)
    
    return hrbeta, hsig


def qmle_inference(X, Y, W, hrbeta, hsig):
    """
    Perform statistical inference for the SAR model by calculating standard errors using the sandwich covariance matrix estimator.
    
    Parameters:
    :X: The covariate matrix.
    :Y: The observed response vector.
    :W: The spatial weight matrix.
    :hrbeta: The estimated parameter vector [beta; rho].
    :hsig: The estimated error variance sigma^2.
    
    Returns:
    :SE1: The standard errors for the estimated parameters [beta; rho].
    """
    N = len(Y)
    p = len(X[1])
    W = W.toarray()
    

    S = np.eye(N) - hrbeta[p,][0] * W
    S1 = np.linalg.inv(S)
    G = W.dot(S1)
    Gs = G + np.transpose(G)
    
    # Calculate residuals based on final estimates
    V = S.dot(Y) - (X.dot(hrbeta[0:p,])).reshape(N, 1)
    
    # Construct the Information Matrix (Expected Hessian) -> Sigm2
    CO1 = 1.0 / (hsig * N) * np.transpose(X).dot(X)
    CO2 = 1.0 / (hsig * N) * np.transpose(G.dot(X).dot(hrbeta[0:p,])).dot(G.dot(X).dot(hrbeta[0:p,])) + 1.0 / N * (G.dot(Gs)).trace()
    CO3 = 1.0 / (hsig * N) * np.transpose(X).dot(G.dot(X).dot(hrbeta[0:p,]))
    
    CO = np.vstack((CO1, np.transpose(CO3)))
    Sigm2 = np.hstack((CO, np.vstack((CO3, CO2))))
    
    # Calculate excess kurtosis adjustment for robust standard errors
    mu_e = np.mean(np.array(V)**4)
    A1 = -G / hsig
    Delta_rr = (mu_e - 3 * hsig**2) / N * sum([A1[i][i]**2 for i in range(len(A1))])
    
    # Construct the variance of the score vector  -> Sigm1
    Sigm1 = np.hstack((np.vstack((CO1, np.transpose(CO3))), np.vstack((CO3, (CO2 + Delta_rr)))))
    
    # Calculate the covariance matrix: Cov = Sigm2^{-1} * Sigm1 * Sigm2^{-1}
    CO_matrix = np.linalg.inv(Sigm2) @ Sigm1 @ np.linalg.inv(Sigm2)
    
    CO_diag = [CO_matrix[i][i] for i in range(p + 1)]
    
    # Calculate the standard errors
    SE1 = 1.0 / N**0.5 * np.sqrt((np.array(CO_diag)).reshape((p + 1, 1)))
    return SE1
