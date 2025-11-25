
'''
    This code demonstrates the operation of the nuts_sampler program. 
    To create a sample from a distribution defined by energy, you need 
    to create a function that returns -E and grad(-E) at the point
    In this example, a distribution is constructed from a multidimensional 
    normal distribution with a random covariance matrix.
'''

import numpy as np
from .nuts_sampler import MassMatrixAdaptation, NutsSampler

def generate_random_covariance_matrix(dim):
    A = np.random.randn(dim, dim)
    return A @ A.T + np.eye(dim) * 0.1

# creating a function that defines the parameters of the distribution
# it should return two values: -E and -grad(E)
def log_multinormal_distribution(x, mean, cov):
    diff = x - mean
    cov_inv_diff = np.linalg.solve(cov, diff)
    return -0.5 * diff.T @ cov_inv_diff, -cov_inv_diff

if __name__ == "__main__":
    np.random.seed(42)
    
    dim = 10
    mean_vector = np.zeros(dim)
    covariance_matrix = generate_random_covariance_matrix(dim)
    
    
    nuts = NutsSampler(
        lambda x: log_multinormal_distribution(x, mean_vector, covariance_matrix),
        dim
    )
    result = nuts.create_sample(
        mass_matrix_mode=MassMatrixAdaptation.LOW_RANK,
        draws=100,
        tune=500,
        chains = 2
    )
    # save bin file
    # nuts.statistics.to_netcdf("results.nc")

    nuts.statistics.save_to_log()
       