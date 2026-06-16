import numpy as np
import scipy
from scipy import stats
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from random import randint, sample
import networkx as nx
import time
import warnings

def project_pd_cone(matrix, eps_H=1e-7):
        """Helper function to project a symmetric matrix onto the PD cone."""
        # Compute eigenvalues and eigenvectors for the symmetric matrix
        evals, evecs = np.linalg.eigh(matrix)
        # Truncate eigenvalues below the strictly positive threshold eps_H
        evals = np.maximum(evals, eps_H)
        # Reconstruct and return the positive definite matrix
        return evecs @ np.diag(evals) @ evecs.T

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


def qmle_dp_estimate(X,Y,W,epsi,delta):
    """
    [Algorithm 1: Differentially Private SAR Model (Estimation)]
    Obtain estimator by DP-Gradient Descent algorithm.
    """
    N=len(Y)
    p=len(X[1])
    
    # Input: Initial value \theta^0
    hrbeta=(0.01*np.ones(p+1)).reshape((p+1,1))

    k=0
    e=10**(-8)
    
    # Initial variance computation \sigma(0)^2
    hS= np.eye(N)-hrbeta[p,]*W.toarray()
    hS1=np.linalg.inv(hS)
    V=hS.dot(Y)-(X.dot(hrbeta[0:p])).reshape(N,1)
    hsig=(V.T.dot(V)/N)[0,0]
    s1=1.0/hsig    

    L=1.5 
    
    # Input: Feasibility parameters (truncation boundaries) c_\beta and c_\rho
    cb=np.sqrt(p)
    cr=0.5
    
    # Input: Truncation level R for dependent variable y 
    cy=hsig*np.sqrt(2*np.log(N))
    
    # Apply truncation: \Pi_R(y)
    Y=np.clip(Y, -cy, cy)
    
    cx=3*np.sqrt(p)
    cl2=0.5*np.log(N)/hsig
    
    # Input: Step size \eta^0
    # Note: eta contains 1/N, effectively dividing the gradient sum by N as required by Alg 1.
    eta=3/(4*N*cl2)
    
    # Input: Number of iterations T
    T=0.8*np.log(N)**2
    
    w1=0.8*np.log(N)
    
    # Input: Noise scales B_\rho and B_\beta (Sensitivities)
    sen_r=(2*(1+w1)*cr/(1-cr)**2+12*cy**2*w1/hsig+4*cy*cx*cb/hsig+4*w1*cy*cx*cb/hsig+4*cy**2/hsig)
    sen_b=4*cx*cy/hsig+8*cx*cy*w1/hsig+4*cx**2*cb/hsig
    
    # Calculate variances for Gaussian noise v_t^\beta and v_t^\rho
    var_b=2*sen_b**2*eta**2*np.log(2*T/delta)*T**2/(N**2*epsi**2)
    var_r=2*sen_r**2*eta**2*np.log(2*T/delta)*T**2/(N**2*epsi**2)
    
    f,hsig=f_lambda_ture_sig(hrbeta[0:p,],hrbeta[p,],Y,X,W)
    

    while k<=T:

        lr=eta/(hsig)
        
        # Generate independent Gaussian noise v_t^\beta and v_t^\rho
        v=np.vstack((np.random.normal(0,var_b**0.5,p).reshape(p,1),np.random.normal(0,var_r**0.5,1).reshape(1,1)))
        

        hrbeta+=lr*f+v
        
        if(np.linalg.norm(hrbeta[0:p,],ord=2,axis=None)>cb):
            hrbeta[0:p,]=cb*hrbeta[0:p,]/np.linalg.norm(hrbeta[0:p,],ord=2,axis=None)
            
        if(abs(hrbeta[p,])>cr):
            hrbeta[p,]=cr
            
        k+=1
        
        f,hsig=f_lambda_ture_sig(hrbeta[0:p,],hrbeta[p,],Y,X,W)
    
    return hrbeta,hsig,T,var_b,var_r,cl2


def qmle_dp_inference(X,Y,W,hrbeta,hsig,var_b,var_r,cl2,epsi,delta):
    """
    [Algorithm 2: Differentially Private Confidence Interval for SAR Model]
    Computes corrected standard errors for hypothesis testing.
    """
    N=len(Y)
    p=len(X[1])
    
    # Privacy budget allocation for the inference step
    epsi=epsi/4
    W=W.toarray()
    
    S=np.eye(N)-hrbeta[p,][0]*W
    S1=np.linalg.inv(S)
    G=W.dot(S1)
    Gs=G+np.transpose(G)
    STW=S.T@W
    STS=S.T@S
    WW=np.transpose(S).dot(W)+np.transpose(W).dot(S)
    
    V=S.dot(Y)-(X.dot(hrbeta[0:p,])).reshape(N,1)

    cx=3*np.sqrt(p)
    cw=0.8*np.log(N)
    cs=0.8*np.log(N)
    cr=0.5
    cb=np.sqrt(p)
    cy=hsig*np.sqrt(2*np.log(N))
    
    # Obtain Differentially Private Estimate of \sigma^2 (\hat{\sigma}^2)
    ssig=(cy+2*cy*cw*np.log(N)+cx*cb)*np.sqrt(2*np.log(1.25/delta))/(N*epsi)
    hsig=np.abs(hsig+np.random.normal(0,ssig,1)[0])
    
    
    drho=4*(1+cw)/(1-cr)**3/(hsig*N)+2*cx*cs/(1-cr)/(hsig*N)
    dbeta=4*cx**2/(hsig*N)
    dbetarho=cx**2*cb*cs/(1-cr)/(hsig*N)
    
    # Noise scales for the Hessian components
    sdrho=np.sqrt(2*np.log(1.25/delta))*drho/(N*epsi)
    sdbeta=np.sqrt(2*np.log(1.25/delta))*dbeta/(N*epsi)
    sdbetarho=np.sqrt(2*np.log(1.25/delta))*dbetarho/(N*epsi)  
    
    # Obtain DP Estimate of \tilde{\Sigma}_2(\theta^{(K)})
    # Inject symmetric Gaussian noise into the Hessian matrix blocks
    upper_triangle = np.random.normal(0, sdbeta , size=(p, p)) 
    upper_triangle = np.triu(upper_triangle, k=0)  
    symmetric_matrix = upper_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))
    
    CO1=1.0/(hsig*N)*np.transpose(X).dot(X)
    CO11=CO1+symmetric_matrix
    
    CO2=1.0/(hsig*N)*np.transpose(G.dot(X).dot(hrbeta[0:p,])).dot(G.dot(X).dot(hrbeta[0:p,]))+1.0/N*(G.dot(Gs)).trace()
    CO21=CO2+(np.random.normal(0,sdrho,1).reshape(1,1))
    
    CO3=1.0/(hsig*N)*np.transpose(X).dot(G.dot(X).dot(hrbeta[0:p,]))
    CO31=CO3+(np.random.normal(0,sdbetarho,p).reshape(p,1))
    
    CO=np.vstack((CO11,np.transpose(CO31)))
    Sigm2=np.hstack((CO,np.vstack((CO31,CO21)))) # This is \tilde{\Sigma}_2
    
    # Obtain DP Estimate of \tilde{\Sigma}_1(\theta^{(K)}) 
    mu_e=np.mean(np.array(V)**4)
    A1=-G/hsig
    
    Delta_rr=(mu_e-3*hsig**2)/N*sum([A1[i][i]**2 for i in range(len(A1))])
    
    # Inject symmetric noise into \tilde{\Sigma}_1
    upper_triangle = np.random.normal(0, np.sqrt(2*np.log(1.25/delta))*cl2/(N*epsi) , size=(p+1, p+1))
    upper_triangle = np.triu(upper_triangle, k=0)  
    symmetric_matrix = upper_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))
    Sigm1=np.hstack((np.vstack((CO1,np.transpose(CO3))),np.vstack((CO3,(CO2+Delta_rr)))))+symmetric_matrix

    # Apply projection to both perturbed covariance components
    Sigm1 = project_pd_cone(Sigm1)
    Sigm2 = project_pd_cone(Sigm2)

    # Compute DP asymptotic variance estimate: \tilde{\Sigma}_+ = \tilde{\Sigma}_2^{-1} \tilde{\Sigma}_1 \tilde{\Sigma}_2^{-1}
    CO=np.linalg.inv(Sigm2)@Sigm1@np.linalg.inv(Sigm2)
    CO=[CO[i][i] for i in range(p+1)]
    
    # Corrected standard error estimate \widehat{SE}_j = \sqrt{\tilde{\Sigma}_{+,jj}/n + var_j}
    SE1=np.sqrt(np.array(CO).reshape((p+1,1))/N+2*np.vstack(((var_b)*np.ones((p,1)).reshape(p,1),(var_r))).reshape((p+1,1)))
    
    return SE1