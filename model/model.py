import numpy as np
from scipy.integrate import solve_ivp

class ModelParams():
    """
    Arguments:
    age_structure -- vector containing the lower bound of each age class
    B             -- birth rate
    V             -- vaccination rates
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

        ### compute age transition rates

        a = age_strucure
        a_shift = np.pad(age_strucure[1:self.k], (0,1), 'edge')
        age_class_sizes = (a_shift-a)[0:self.k-1]

        #age transition matrix 
        self.A = np.eye(self.k) - np.diag(np.pad(1.0/age_class_sizes, (0,1), 'constant', constant_values=(0)), k=0) + np.diag(1.0/age_class_sizes, k=-1)

class SIRVModel(object):
    """docstring for SIRVModel"""
    def __init__(self, model_params, force_of_infection):
        super(SIRVModel, self).__init__()
        self.model_params = model_params
        self.force_of_infection = force_of_infection