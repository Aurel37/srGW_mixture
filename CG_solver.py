import numpy as np
from numba import jit
from scipy.optimize import linprog
from update_progress import update_progress

class CG_solver:
    """cg solver
    """
    def __init__(self, T_0, C, C_bar, h, D):
        self.T = T_0
        self.C = C
        self.C_bar = C_bar
        self.N, self.K = C.shape
        self.h = h
        self.D = D
        
    @jit(forceobj = True)
    def L(self, C_1, C_2, T):
        """
        compute L(C,C_bar)\otimes T see 'Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.'
        """
        one = np.array([1 for i in range(len(C_1))])

        h_bar = T.T@one
        constC_1 = np.dot(np.dot(np.square(C_1), self.h.reshape(-1, 1)),
                     np.ones(len(C_2)).reshape(1, -1))
        constC_2 = np.dot(np.ones(len(C_1)).reshape(-1, 1),
                     np.dot(h_bar.reshape(1, -1), np.square(C_2).T))
        L = -np.dot(C_1, T).dot(C_2.T)
        return L + constC_1 + constC_2
    
    def gradient(self, T):
        """compute gradient w.r.t. T of equation [25] in 'Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, 
        and Nicolas Courty. Semi-relaxed gromov-wasserstein divergence with applications on graphs, 2022.'
        """
        return self.L(self.C, self.C_bar, self.T) + self.L(self.C.T, self.C_bar.T, self.T)
    
    @jit(forceobj = True)
    def direction(self, G):
        """Compute the direction descent 
        """
        n, k = G.shape
        A_sum = np.zeros((k, k))
        A_sum[:][0] = np.array([1 for i in range(k)]) 
        b_eq = np.array([0 for i in range(k)])
        b_eq[0] = 1
        X = []
        for i in range(n):
            x_i = linprog(G[i] + self.D[i], A_ub=-np.eye(k), b_ub=np.array([0 for i in range(k)]), A_eq = A_sum, b_eq = b_eq).x
            X.append(self.h[i]*x_i)
        return np.array(X)
        #return np.array([g_i/np.linalg.norm(g_i) for g_i in G])

    @jit(forceobj = True)
    def optimal_step(self, X):
        """compute the optimal gamma
        """
        b = np.trace(self.L(self.C, self.C_bar, X-self.T).T@self.T + self.L(self.C, self.C_bar, self.T).T@(X-self.T) + self.D.T@(X-self.T))
        a = np.trace(self.L(self.C, self.C_bar, X-self.T).T@(X-self.T))
        gamma = 0.01
        if a > 0:
            gamma = min(1, max(0.01, -b/(2*a)))
        else:
            if a+b < 0:
                gamma = 1
        return gamma
    
    def run(self, epsilon):
        T_0 = 1000*self.T
        Max_ITER = 1000
        iter_n = 0
        old_loss = 0
        curr_loss = np.sum(self.L(self.C, self.C_bar, self.T) * self.T)
        crit = 1
        # CG algorithm
        while crit > epsilon**2 and iter_n < Max_ITER:
            old_loss = curr_loss
            T_0 = self.T
            G = self.gradient(self.T)
            X = self.direction(G)
            gamma = self.optimal_step(X)
            self.T = (1-gamma)*self.T + gamma*X
            iter_n += 1
            curr_loss = np.sum(self.L(self.C, self.C_bar, self.T) * self.T)
            crit = np.abs(curr_loss - old_loss)/np.abs(old_loss)
        return self.T