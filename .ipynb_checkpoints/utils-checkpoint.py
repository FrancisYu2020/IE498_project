import numpy as np
import time
from scipy.stats import norm

def risk_neutral_probability(r, q, delta, u, d):
    return (np.exp((r - q) * delta) - d) / (u - d)

def Binomial(Option, K, T, S0, sigma, r, q, N, Exercise):
    '''
    Binomial model implementation for project question 1
    '''
    # start the timer
    start_time = time.time()

    # function to compute f(n, j)
    delta = T / N
    u = np.exp(sigma * np.sqrt(delta))
    d = np.exp(-sigma * np.sqrt(delta))
    p = risk_neutral_probability(r, q, delta, u, d)

    def payoff(n, j):
        if Option == 'C':
            # use call payoff function
            return max(0, S0 * u**j * d**(n - j) - K)
        elif Option == 'P':
            # use put payoff function
            return max(0, K - S0 * u**j * d**(n - j))
        else:
            raise NameError(f"Unrecognized option type {Option}!")

    # set up the memory buffer for dynamic programming
    # only use two vectors for memory efficiency
    memory = np.ones((2, N + 1))
    # initialize the leave node values
    for j in range(N + 1):
        memory[N % 2, j] = payoff(N, j)

    # main part for dynamic programming
    for n in range(N - 1, -1, -1):
        for j in range(n + 1):
            t0, t1 = n % 2, (n + 1) % 2
            if Exercise == 'A':
                memory[t0, j] = max(payoff(n, j), np.exp(- r * delta) * (p * memory[t1, j + 1] + (1 - p) * memory[t1, j]))
            elif Exercise == 'E':
                memory[t0, j] = np.exp(- r * delta) * (p * memory[t1, j + 1] + (1 - p) * memory[t1, j])
            else:
                raise NameError(f'Unrecognized option style {Exercise}!')
    
    option_price = memory[0, 0]
    end_time = time.time()

    return option_price, end_time - start_time

def black_scholes_call_price(S0, r, q, K, T, sigma):
    '''
    European Black-Scholes formula to compute call price
    Helper function for question 2
    '''
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(- q * T) * norm.cdf(d1) - K * np.exp(- r * T) * norm.cdf(d2)

def payoff(Option, K, S):
    assert Option in ['P', 'C'], "Only P and C accepted for option parameter"
    if Option == 'P':
        return max(0, K - S)
    else:
        return max(0, S - K)