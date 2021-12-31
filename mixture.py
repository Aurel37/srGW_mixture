import numpy as np

def F(dist):
    """cumulative function of dist
    
    Paremeters:
        dist (np.array) : 
    """
    n = len(dist)
    dist_cumul = [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            dist_cumul[i] = dist[i]
        else:
            dist_cumul[i] = dist[i] + dist_cumul[i-1]
    return np.array(dist_cumul)
    
    
def discrete_sampler(X, dist):
    """sample X according to the discrete
    distribution dist
    """
    dist_cumul = F(dist)
    u = np.random.random(1)
    j = 0
    m = len(dist_cumul)
    while u > dist_cumul[j]:
        j += 1
        if j == m-1:
            u = 0
    return X[j]

def gaussian_mixture(n, d, alpha, mu, sigma, color = False):
    """Sample a Gaussian mixture of n observations with dimension d with parameters alpha, mu, sigma
    
    Parameters:
        n (int) : number of observations
        d (int) : dimension of the observations
        alpha (np.array) : array that represents the density of Z_i
        mu (np.array) : array of mean of the gaussians
        sigma (list(np.matrix)) : list of variance matrices
        color (Boolean) : if True return an array of the form [x_i, j] 
                            with j the cluster x_i belongs to
                        (useful for doing plots)
                          if Flase return only the x_i
                          
    """
    m = len(alpha)
    X = [] # result, at the end X = [X_1,...,X_n]
    J = [i for i in range(m)]
    for i in range(n):
        # sample the Z_i "to find which gaussian j to sample"
        j = discrete_sampler(J, alpha)
        # sample the "gaussian j"
        
        # first sample a normal vector of dimension d
        U = np.random.normal(0,1,d)   
        if d == 1:
            L = np.sqrt(sigma[j])
            # return the gaussian with parameters mu[j], sigma[j]
            if color:
                # the sampled value belongs to the j-th cluster
                X.append([mu[j] + L*U, j])
            else:
                X.append(mu[j] + L*U)
        else:
            # perform the cholesky decomposition in dimension d>1
            L = np.linalg.cholesky(sigma[j])
            # return the gaussian with parameters mu[j], sigma[j]
            if color:
                # the sampled value belongs to the j-th cluster
                X.append([mu[j] + L@U, j])
            else:
                X.append(mu[j] + L@U)
    return X