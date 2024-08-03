from utils import Binomial, black_scholes_call_price, black_scholes_put_price
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import sys
sys.setrecursionlimit(20000)

start_time = time.time()

r = 0.05
q = 0.04
K = 100
T = 1
S0 = 100
sigma = 0.2

N = 400 # we simulate binmial model with N from 1 - 100
xs = np.arange(1, N + 1, 1)
# bs_price = black_scholes_call_price(S0, r, q, K, T, sigma)

# fontsize = 15
# plt.plot(xs, [bs_price] * len(xs), label='black scholes call price')
# plt.plot(xs, [Binomial('C', K, T, S0, sigma, r, q, n, 'E')[0] for n in tqdm(xs)], label='binomial modeled call price', alpha=0.5)
# plt.xlabel('binomial step N', fontsize=fontsize)
# plt.ylabel('call price', fontsize=fontsize)
# plt.title('Binomial Model Estimated Call Price vs. Black Scholes Modeled Call Price', fontsize=fontsize)
# plt.legend(fontsize=fontsize)
# plt.savefig('question2.png')
# error = np.abs(Binomial('C', K, T, S0, sigma, r, q, N, 'E')[0] - bs_price)
# print(f'Black-Scholes option price: {bs_price}, the error is {error}')

bs_price = black_scholes_put_price(S0, r, q, K, T, sigma)

fontsize = 15
plt.plot(xs, [bs_price] * len(xs), label='black scholes put price')
plt.plot(xs, [Binomial('P', K, T, S0, sigma, r, q, n, 'E')[0] for n in tqdm(xs)], label='binomial modeled put price', alpha=0.5)
plt.xlabel('binomial step N', fontsize=fontsize)
plt.ylabel('put price', fontsize=fontsize)
plt.title('Binomial Model Estimated Put Price vs. Black Scholes Modeled Put Price', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.savefig('question2_put.png')
error = np.abs(Binomial('P', K, T, S0, sigma, r, q, N, 'E')[0] - bs_price)
print(f'Black-Scholes option price: {bs_price}, the error is {error}')



end_time = time.time()
print(f'Time elapsed: {end_time - start_time} seconds')