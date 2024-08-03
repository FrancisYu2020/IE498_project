from utils import Binomial, black_scholes_call_price, black_scholes_put_price
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def richardson_extrapolation(price_N, price_2N):
    '''
    BBSR uses Richardson extrapolation
    '''
    return 2 * price_2N - price_N

def BBSR(Option, K, T, S0, sigma, r, q, N, Exercise):
    # Calculate option prices using binomial trees with N and 2N steps
    price_N = Binomial(Option, K, T, S0, sigma, r, q, N, Exercise)[0]
    price_2N = Binomial(Option, K, T, S0, sigma, r, q, 2 * N, Exercise)[0]
    
    # Apply Richardson extrapolation
    BBSR_price = richardson_extrapolation(price_N, price_2N)
    
    return BBSR_price

r = 0.05
q = 0.04
K = 100
T = 1
S0 = 100
sigma = 0.2

N = 200 # we simulate binmial model with N from 1 - 100
xs = np.arange(1, N + 1, 1)
# bs_price = black_scholes_call_price(S0, r, q, K, T, sigma)

# fontsize = 15
# plt.plot(xs, [bs_price] * len(xs), label='black scholes call price')
# plt.plot(xs, [BBSR('C', K, T, S0, sigma, r, q, n, 'E') for n in tqdm(xs)], label='BBSR call price', alpha=0.5)
# plt.xlabel('N', fontsize=fontsize)
# plt.ylabel('call price', fontsize=fontsize)
# plt.title('BBSR Call Price vs. Black Scholes Modeled Call Price', fontsize=fontsize)
# plt.legend(fontsize=fontsize)
# plt.savefig('BBSR_call.png')
# error = np.abs(BBSR('C', K, T, S0, sigma, r, q, N, 'E') - bs_price)
# print(f'Black-Scholes option price: {bs_price}, the error is {error}')

bs_price = black_scholes_put_price(S0, r, q, K, T, sigma)

fontsize = 15
plt.plot(xs, [bs_price] * len(xs), label='black scholes put price')
plt.plot(xs, [BBSR('P', K, T, S0, sigma, r, q, n, 'E') for n in tqdm(xs)], label='BBSR put price', alpha=0.5)
plt.xlabel('N', fontsize=fontsize)
plt.ylabel('call price', fontsize=fontsize)
plt.title('BBSR Put Price vs. Black Scholes Modeled Put Price', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.savefig('BBSR_put.png')
error = np.abs(BBSR('P', K, T, S0, sigma, r, q, N, 'E') - bs_price)
print(f'Black-Scholes option price: {bs_price}, the error is {error}')