import numpy as np
import math
import sys
from scipy.integrate import solve_ivp

class ModelParams():
    """
    Parameters struct
    Arguments:
        age_structure -- vector containing the lower bound for each age class
        B             -- average birth rate per capita T -> \\R
        V             -- effective vaccination rate (includes vaccine efficacy) T -> k
        d             -- death rates, T -> k
        gamma         -- recovery rates
        C             -- intergenerational contact rate matrix (Who Interacts With Who Matrix)
        N             -- population size, T -> \\R
    """
    def __init__(self, age_strucure, B, V, d, gamma, C, N):
        self.age_strucure = age_strucure
        self.B = B
        self.V = V 
        self.d = d
        self.gamma = gamma
        self.C = C
        self.N = N

        ### k age classes
        self.k = np.size(age_strucure)

class SIRVModel(object):
    """
    SIRV model class
    This class is stateless post-initialisation!
    Methods:
        run                 -- run the simulation
    Arguments:
        model_params        -- model parameters
        force_of_infection  -- function T -> k x k, potentially time dependent force of infection
        population_profile  -- 'NORMALISE' or 'BALANCE' Balance will recompute synthetic death rates. Default:'Normalise'
    """
    def __init__(self, model_params, force_of_infection, population_profile='NORMALISE'):
        super(SIRVModel, self).__init__()
        self.model_params = model_params
        self.force_of_infection = force_of_infection
        self.__build__()

    def __build__(self):
        ### Initialise the model
        mp = self.model_params

        ### export precomputed matrices
        self.C = mp.C
        self.k = mp.k

        ### convert vectors to diagonal matrices
        self.b = lambda t: np.reshape((mp.B(t))* np.eye(1,self.k,0),(-1))
        self.V_mat = lambda t: np.diag(mp.V(t))
        self.gamma_mat = np.diag(mp.gamma)

        ### precompute aging matrix from age structure vector
        a = mp.age_strucure
        a_shift = np.pad(mp.age_strucure[1:self.k], (0,1), 'edge')
        age_class_sizes = (a_shift-a)[0:self.k-1]

        #death_rate = 1.0/(mp.N/mp.B-np.sum(1.0/age_class_sizes))
        #assert death_rate > 0, "Death rate not positive, parameters entered probably correspond to a growing population profile"

        #age transition matrix 
        self.A = np.diag(np.pad(1.0/age_class_sizes, (0,1), 'constant', constant_values=(0)), k=0) 
        self.I = np.eye(mp.k)
        self.L = np.diag(np.ones(mp.k-1), k=-1)

        self.d_mat = lambda t: np.diag(mp.d(t))# np.eye(1,self.k,self.k-1)*death_rate

    def __dt__(self, t, y):
        ### Diff equation in matrix form
        b = self.b
        V = self.V_mat
        d = self.d_mat
        C = self.C
        A = self.A
        beta = self.force_of_infection   
        gamma = self.gamma_mat
        
        I=self.I

        ### extract compartments from state vector
        K_max = int(len(y)/4)   
        
        s = y[0:K_max]
        i = y[K_max:(K_max*2)]
        r = y[(K_max*2):(K_max*3)]
        v = y[(K_max*3):]         

        ### SIRV equations here
        ds = (I-V(t))@b(t) - d(t)@s - s*(beta(t)@C@i)
        di = s*(beta(t)@C@i) - (d(t)+gamma)@i
        dr = gamma@i - d(t)@r
        dv = V(t)@b(t) - d(t)@v
        ### end SIRV equations

        return np.hstack([ds,di,dr,dv])

    def __age__(self, y, t):

        A=self.A
        L=self.L
        I=self.I
        V=self.V_mat
        
        ### Apply discrete aging
        K_max = int(len(y)/4)   

        ### extract compartments from state vector        
        s = y[0:K_max]
        i = y[K_max:(K_max*2)]
        r = y[(K_max*2):(K_max*3)]
        v = y[(K_max*3):]

        ### multiply by aging matrix A
        s = s+(L-I-V(t)@L)@A@s 
        i = i+(L-I)@A@i
        r = r+(L-I)@A@r
        v = v+(L-I)@A@v+V(t)@L@A@s

        n = np.sum(s+i+r+v)

        s=(1/n)*s
        i=(1/n)*i
        r=(1/n)*r
        v=(1/n)*v

        return np.hstack([s, i, r, v])

    def run(self, ivs, t_max, method = 'RK45', eval_per_year=-1, t_year_scale = 1.0):
        """
        Runs Model
        Arguments:
            ivs          -- Initial conditions
            t_max        -- Run model until t_max preferably an integer!
            method       -- Approximation method passed to integrator (default: RK45)
            eval_per_year -- Time steps per year. Use -1 for adaptive step size (default: -1)
            t_year_scale -- Year scale length (default: one year)
        Returns: 
            Y_t    -- [S;I;R;V] x T
            T      -- Time steps
        """
        T=[0]
        Y_t = ivs
        Y0 = ivs

        first_run = True

        while (T[-1] < t_max):
            t0 = T[-1]
            tend = min(t_max,T[-1]+1*t_year_scale)
            t_span = (t0, tend)
            ### Solve for one unit on year scale
            if (eval_per_year <0):
                sol_1_year = solve_ivp(self.__dt__, t_span = t_span, y0 = Y0, method = method)
            else:
                sol_1_year = solve_ivp(self.__dt__, t_span = t_span, y0 = Y0, method = method, t_eval = np.linspace(t0, tend, num=eval_per_year))

            if first_run:
               Y_t = sol_1_year.y
               first_run = False
               T = sol_1_year.t

            else:
                Y_t = np.hstack([Y_t, sol_1_year.y])
                T = np.hstack([T, sol_1_year.t])

            ### Apply aging (or other delta functions)
            Y0 = self.__age__(Y_t[:,-1], T[-1])
        return (Y_t, T) 

    def get_age_matrix(self):
        return (self.L-self.I)@self.A


