import numpy as np
import math
import sys
from scipy.integrate import solve_ivp

class ModelParams():
    """
    Parameters struct
    Arguments:
        age_structure -- vector containing the lower bound for each age class
        B             -- birth rate
        V             -- effective vaccination rate (includes vaccine efficacy)
        d             -- death rates
        gamma         -- recovery rates
        C             -- intergenerational contact rate matrix (Who Interacts With Who Matrix)
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
    SIRV model class
    This class is stateless post-initialisation!
    Methods:
        run                 -- run the simulation
    Arguments:
        model_params        -- model parameters
        force_of_infection  -- function T -> k x k, potentially time dependent force of infection
    """
    def __init__(self, model_params, force_of_infection):
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
        self.b = mp.B * np.eye(1,self.k,0)
        self.d_mat = np.diag(mp.d)
        self.gamma_mat = np.diag(mp.gamma)

        ### precompute aging matrix from age structure vector
        a = mp.age_strucure
        a_shift = np.pad(mp.age_strucure[1:self.k], (0,1), 'edge')
        age_class_sizes = (a_shift-a)[0:self.k-1]

        #age transition matrix 
        self.A = np.diag(np.pad(1.0/age_class_sizes, (0,1), 'constant', constant_values=(0)), k=0) 
        self.V_mat = np.diag(mp.V)

        self.I = np.eye(mp.k)
        self.L = np.diag(np.ones(mp.k-1, k=-1))

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
        L=self.L

        ### extract compartments from state vector
        K_max = int(len(y)/4)   
        
        s = y[0:K_max]
        i = y[K_max:(K_max*2)]
        r = y[(K_max*2):(K_max*3)]
        v = y[(K_max*3):]         


        ### SIRV equations here
        ds = (I-V)@b - d@s - s*(beta(t)@C@i)
        di = s*(beta(t)@C@i) - (d+gamma)@i
        dr = gamma@i - d@r
        dv = V@b - d@v
        ### end SIRV equations

        return np.hstack([ds,di,dr,dv])

    def __age__(self, y):

        A=self.A
        L=self.L
        I=self.I
        V=self.V

        ### Apply discrete aging
        K_max = int(len(y)/4)   

        ### extract compartments from state vector        
        s = y[0:K_max]
        i = y[K_max:(K_max*2)]
        r = y[(K_max*2):(K_max*3)]
        v = y[(K_max*3):]

        ### multiply by aging matrix A
        s = (L-I-V@L)@A@s 
        i = (L-I)@A@i
        r = (L-I)@A@r
        v = (L-I)@A@v+V@L@A@s
        
        return np.hstack([s, i, r, v])

    def run(self, ivs, t_max, method = 'RK45', t_year_scale = 1.0):
        """
        Runs Model
        Arguments:
            ivs          -- Initial conditions
            tmax         -- Run model until tmax preferably an integer!
            method       -- Approximation method passed to integrator (default: RK45)
            t_year_scale -- Year scale length (default: one year)
        Returns: 
            Y_t    -- [S;I;R;V] x T
            T      -- Time steps
        """
        T=[0]
        Y_t = ivs
        Y0 = ivs

        first_run = True

        ###Initialise progress bar
        num_it = int(math.ceil((t_max/t_year_scale)))
        progress_bar_width = min(num_it,40)
        tick_when = int(num_it/progress_bar_width)

        sys.stdout.write("[")
        #sys.stdout.flush()
        #sys.stdout.write("\b" * (progress_bar_width+1))

        while (T[-1] < t_max):

            ### Solve for one unit on year scale
            sol_1_year = solve_ivp(self.__dt__, t_span = (T[-1], min(t_max,T[-1]+1*t_year_scale)), y0 = Y0, method = method)

            if first_run:
               Y_t = sol_1_year.y
               first_run = False
               T = sol_1_year.t

            else:
                Y_t = np.hstack([Y_t, sol_1_year.y])
                T = np.hstack([T, sol_1_year.t])

            ### Apply aging (or other delta functions)
            Y0 = self.__age__(Y_t[:,-1])

            ###Show progress
            if(int(T[-1])%tick_when==0):
                sys.stdout.write("-")
                sys.stdout.flush()
        ### Close progress bar
        sys.stdout.write("]\n")
        return (Y_t, T) 