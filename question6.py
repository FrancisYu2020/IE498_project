import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
# reuse CRR_Binomial and black scholes model from previous questions
from utils import Binomial, European_option_price

def deAmericanize(option, K, T, S, target_price, r, q, sigma_low, sigma_high, epsilon):
    # binary search on 10 base log uniform space
    sigma_mid = np.exp((np.log(sigma_low) + np.log(sigma_high)) / 2)
    curr_price = European_option_price(option, S, r, q, K, T, sigma_mid)
    print(f'Current sigma = {sigma_mid}, price diff = {abs(curr_price - target_price)}')
    if curr_price < target_price - epsilon:
        return deAmericanize(option, K, T, S, target_price, r, q, sigma_mid, sigma_high, epsilon)
    elif curr_price > target_price + epsilon:
        return deAmericanize(option, K, T, S, target_price, r, q, sigma_low, sigma_mid, epsilon)
    else:
        return sigma_mid

if __name__ == '__main__':
    option = 'C'
    S = 100
    K = 100
    T = 1
    r = 0.05
    q = 0
    sigma = 0.2
    N = 1000
    
    # parameter search space
    sigma_low = 0.0001
    sigma_high = 1
    epsilon=1e-6 # error tolerance

    american_call_price = Binomial(option, K, T, S, sigma, r, q, N, 'A')[0]
    
    calibrated_sigma = deAmericanize(option, K, T, S, american_call_price, r, q, sigma_low, sigma_high, epsilon)
