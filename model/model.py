import numpy as np
from scipy.integrate import solve_ivp

class ModelParams():
    """
    Arguments:
        age_structure -- vector containing the lower bound of each age class
        B             -- birth rate
        V             -- effective vaccination rate (includes vaccine efficacy)
        d             -- death rates
        gamma         -- recovery rates
        C             -- intergenerational contact rate matrix
    """
    def __init__(self, age_strucure, B, V, d, gamma, C):
        self.age_strucure = age_strucure
        self.B = B
        self.V = V 
        self.d = d
        self.gamma = gamma
        self.C = C

        ### k age classes
        self.k = np.size(age_strucure)

class SIRVModel(object):
    """
    Arguments:
        model_params        -- model parameters
        force_of_infection  -- T -> kxk, potentially time dependent force of infection
    """
    def __init__(self, model_params, force_of_infection):
        super(SIRVModel, self).__init__()
        self.model_params = model_params
        self.force_of_infection = force_of_infection
        self.__build__()

    def __build__(self):
        ### Initialise the model
        mp = self.model_params

        self.C = mp.C
        self.k = mp.k

        self.B_mat = np.diag(mp.B)
        self.V_mat = np.diag(mp.V)
        self.d_mat = np.diag(mp.d)
        self.gamma_mat = np.diag(mp.gamma)

        a = mp.age_strucure
        a_shift = np.pad(age_strucure[1:self.k], (0,1), 'edge')
        age_class_sizes = (a_shift-a)[0:self.k-1]

        #age transition matrix 
        self.A = np.eye(self.k) - np.diag(np.pad(1.0/age_class_sizes, (0,1), 'constant', constant_values=(0)), k=0) + np.diag(1.0/age_class_sizes, k=-1)

    def __dt__(self, t, y):
        ### Diff equation in matrix form
        B = self.B_mat
        V = self.V_mat
        d = self.d_mat
        C = self.C
        A = self.A
        beta = self.force_of_infection   
        gamma = self.gamma_mat
     
        
        K_max = int(len(y)/4)   
        
        s = y[0:K_max]
        i = y[K_max:(K_max*2)]
        r = y[(K_max*2):(K_max*3)]
        v = y[(K_max*3):]         

        ds = B - (V+d)@s - s*(beta(t)@C@i)
        di = s*(beta(t)@C@i) + (d+gamma)@i
        dr = gamma@i - d@r
        dv = V@s - d@v

        return [ds,di,dr,dv]

    def __age__(self, y):
        ### Apply discrete aging
        s = y[0:K_max]
        i = y[K_max:(K_max*2)]
        r = y[(K_max*2):(K_max*3)]
        v = y[(K_max*3):]

        s = A@s   
        i = A@i
        r = A@r
        v = A@v

    return [s, i, r, v]

    def run(self, ivs, t_max, method = 'RK45', t_year_scale = 1.0):
        """
        Runs Model
        Arguments:
            ivs    -- Initial conditions
            tmax   -- Run model until tmax preferably an integer!
            method -- Approximation method passed to integrator
        Returns: 
            Y_t    -- [S;I;R;V] x T
            T      -- Time steps
        """
        T = [0]
        Y_t = ivs
        Y0 = ivs

        while (T[-1] < t_max):
            sol_1_year = solve_ivp(self.__dt__, t_span = (T[-1], min(t_max,(T[-1]+1)*t_year_scale)), y0 = Y0, method = method)
            Y0 = self.__age__(Y_t[:,-1])
            Y_t = np.hstack(Y_t, sol_1_year.y)
            T = np.hstack(T, sol_1_year.t)
        return (Y_t, T)