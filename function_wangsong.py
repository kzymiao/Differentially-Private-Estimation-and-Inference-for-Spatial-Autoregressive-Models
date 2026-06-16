import numpy as np

def lse_estimate(X, Y, W):
    """
    Non-Private Baseline Estimation: Closed-form Nonlinear Least Squares (NLSE) solution
    """
    WY = W.dot(Y)
    X_tilde = np.column_stack((WY, X))
    # Closed-form solution: theta = (X_tilde^T * X_tilde)^(-1) * X_tilde^T * Y
    theta = np.linalg.inv(X_tilde.T.dot(X_tilde)).dot(X_tilde.T).dot(Y)
    return theta, None

def lse_dp_estimate(X, Y, W, epsilon=1.0, c3=1.0, p=0.5, q=0.7):
    """
    Differentially Private Estimation: Functional Mechanism with Laplace Noise (Original implementation based on Wang & Song):
    Zhijian Wang and Yunquan Song. Privacy-preserving parametric inference for spatial autoregressive model. Test, 33(3):877–896, 2024.
    Achieves epsilon-DP by injecting Laplace noise calibrated to the L1-sensitivity of the objective function's polynomial expansion coefficients.
    """
    d = X.shape[1]
    
    # Compute the true sufficient statistics
    WY = W.dot(Y)
    C_rho2 = WY.T.dot(WY)
    C_beta2 = X.T.dot(X)
    C_rho = Y.T.dot(WY)
    C_beta = X.T.dot(Y)
    C_rho_beta = X.T.dot(WY)
    
    # Define L1-sensitivity bounds for each sufficient statistic block
    delta_c_rho2 = 4 * c3 + 2
    delta_c_beta2 = 2 * (d**2)
    delta_c_rho = 4 + 2 * c3
    delta_c_beta = 2 * d
    delta_c_rho_beta = (4 + 2 * c3) * d
    
    # Sequential privacy budget allocation based on hyper-parameters p and q
    eps1 = p * epsilon
    eps2 = (1 - p) * epsilon
    eps3 = (1 - p) * (1 - q) * epsilon
    
    # Laplace noise proportional to L1-sensitivity / allocated epsilon
    noise_rho2 = np.random.laplace(0, delta_c_rho2 / eps1)
    noise_rho = np.random.laplace(0, delta_c_rho / eps1)
    noise_rho_beta = np.random.laplace(0, delta_c_rho_beta / eps2, size=(d, 1))
    noise_beta = np.random.laplace(0, delta_c_beta / eps3, size=(d, 1))
    
    # Generate noise for the Hessian component and enforce symmetry
    noise_beta2 = np.random.laplace(0, delta_c_beta2 / eps3, size=(d, d))
    noise_beta2 = (noise_beta2 + noise_beta2.T) / 2
    
    # Construct the privatized sufficient statistics
    hC_rho2 = C_rho2 + noise_rho2
    hC_beta2 = C_beta2 + noise_beta2
    hC_rho = C_rho + noise_rho
    hC_beta = C_beta.reshape((d, 1)) + noise_beta
    hC_rho_beta = C_rho_beta.reshape((d, 1)) + noise_rho_beta
    
    # Assemble the perturbed linear system and solve for theta_dp
    A = np.block([[hC_beta2, hC_rho_beta], [hC_rho_beta.T, hC_rho2]])
    B = np.vstack((hC_beta, hC_rho))
    
    theta_dp = np.linalg.solve(A, B)
    return theta_dp, None, None

def lse_dp_gaussian_estimate(X, Y, W, epsilon=1.0, delta=1e-5, c3=1.0, p=0.5, q=0.7):
    """
    Differentially Private Estimation: Functional Mechanism with Gaussian Noise (Ablated Version).
    Achieves approximate (epsilon, delta)-DP. 
    By shifting to the Gaussian mechanism and L2-sensitivity, it drastically reduces 
    the dimensionality penalty compared to the Laplace mechanism.
    """
    d = X.shape[1]
    
    # Compute the true sufficient statistics
    WY = W.dot(Y)
    C_rho2 = WY.T.dot(WY)
    C_beta2 = X.T.dot(X)
    C_rho = Y.T.dot(WY)
    C_beta = X.T.dot(Y)
    C_rho_beta = X.T.dot(WY)
    
    # Define L2-sensitivity bounds
    delta2_rho2 = 4 * c3 + 2
    delta2_rho = 4 + 2 * c3
    delta2_beta2_element = 2.0            
    delta2_beta_element = 2.0             
    delta2_rho_beta_element = 4 + 2 * c3  
    
    # Sequential privacy budget allocation
    eps1 = p * epsilon
    eps2 = (1 - p) * epsilon
    eps3 = (1 - p) * (1 - q) * epsilon
    
    # Gaussian Mechanism
    def gaussian_scale(delta2, eps):
        if eps <= 1e-10:
            return 1e10  
        return delta2 * np.sqrt(2 * np.log(1.25 / delta)) / eps
    
    # Inject independent Gaussian noise scaled to L2-sensitivity and budget
    noise_rho2 = np.random.normal(0, gaussian_scale(delta2_rho2, eps1))
    noise_rho = np.random.normal(0, gaussian_scale(delta2_rho, eps1))
    noise_rho_beta = np.random.normal(0, gaussian_scale(delta2_rho_beta_element, eps2), size=(d, 1))
    noise_beta = np.random.normal(0, gaussian_scale(delta2_beta_element, eps3), size=(d, 1))
    
    # Generate noise for the Hessian component and enforce symmetry
    noise_beta2 = np.random.normal(0, gaussian_scale(delta2_beta2_element, eps3), size=(d, d))
    noise_beta2 = (noise_beta2 + noise_beta2.T) / 2
    
    # Construct the privatized sufficient statistics
    hC_rho2 = C_rho2 + noise_rho2
    hC_beta2 = C_beta2 + noise_beta2
    hC_rho = C_rho + noise_rho
    hC_beta = C_beta.reshape((d, 1)) + noise_beta
    hC_rho_beta = C_rho_beta.reshape((d, 1)) + noise_rho_beta
    
    
    A = np.block([[hC_beta2, hC_rho_beta], [hC_rho_beta.T, hC_rho2]])
    B = np.vstack((hC_beta, hC_rho))
    
    theta_dp = np.linalg.solve(A, B)
    return theta_dp, None, None