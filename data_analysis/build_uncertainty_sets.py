from load_data import load_generator_parameters
from analyze_emissions import emissions_mean

import numpy as np
import picos as pic
from picos import Problem, RealVariable


# data 
parameters = load_generator_parameters() 

print(parameters.head())

T = 24 
J = 22

c        = np.asarray(parameters["marginal_cost"], dtype=float)
k        = np.asarray(parameters["max_capacity"], dtype=float)
gamma_u  = np.asarray(parameters["ramp_up"], dtype=float)
gamma_d  = np.asarray(parameters["ramp_down"], dtype=float)
mu_e     = np.asarray(parameters)




