from utils import Binomial, payoff
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import sys
import os
import multiprocessing as mp
# sys.setrecursionlimit(20000)

start_time = time.time()

# useful helper functions
def binomial_wrapper(args):
    Option, K, T, S0, sigma, r, q, N, Exercise = args
    ret = Binomial(Option, K, T, S0, sigma, r, q, N, Exercise)[0]
    return ret

def binomial_deviation_wrapper(args):
    Option, K, T, S0, sigma, r, q, N, Exercise = args
    ret = Binomial(Option, K, T, S0, sigma, r, q, N, Exercise)[0] - payoff(Option, K, S0)
    return (S0, ret)

# global parameters
option = 'C'
K = 100
T = 1
sigma = 0.2
r = 0.05
q1 = 0.04
q2 = 0.08

# # part 1: Find the good time steps
# def compute_delta(args):
#     r, q, K, sigma, T, S0, tolerance = args
#     delta = 0.01  # initial delta should be smaller than 1 / 12
#     diff = np.inf
#     while diff >= tolerance:
#         delta /= 1.1
#         curr_price = Binomial(option, K, T, S0, sigma, r, q, int(T / delta), 'A')[0]
#         next_price = Binomial(option, K, T, S0, sigma, r, q, int(T / delta) + 1, 'A')[0]
#         diff = np.abs(curr_price - next_price)
#         if diff < tolerance:
#             return S0, delta
#     return S0, delta

# def find_delta(r, q, K, sigma, results_path='results.txt', write_mode='w', tolerance=1e-3):
#     if os.path.exists(results_path) and write_mode == 'w':
#         print('Already saved results for question 3')
#         with open(results_path, 'r') as f:
#             print(f.read())
#         return

# #     tasks = [(r, q, K, sigma, T, S0, tolerance) for T in np.linspace(1/12, 1, 12) for S0 in range(70, 135, 5)]
#     tasks = [(r, q, K, sigma, T, S0, tolerance) for T in np.linspace(1/12, 1, 12) for S0 in range(100, 160, 1)]
#     selected_delta = np.inf

#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         for result in tqdm(pool.imap_unordered(compute_delta, tasks), total=len(tasks)):
#             S0, delta = result
#             selected_delta = min(selected_delta, delta)
#             tqdm.write(f'S0 = {S0} converges at delta = {delta}')

#     with open(results_path, write_mode) as f:
#         f.write(f'selected delta for q = {q}: {selected_delta} ')

# # # find delta when q = 0
# find_delta(r, q1, K, sigma, 'question4-delta1.txt')
# find_delta(r, q2, K, sigma, 'question4-delta2.txt')


# Part 2: option price vs. S0 plot for 12-month put
# delta = 7.5e-3
# N = int(T / delta)
    
# S = np.arange(60, 150)

# args1 = [(option, K, T, S0, sigma, r, q1, N, 'A') for S0 in S]
# args2 = [(option, K, T, S0, sigma, r, q2, N, 'A') for S0 in S]

# with mp.Pool(processes=80) as pool:
#     results1 = list(tqdm(pool.imap(binomial_wrapper, args1), total=len(args1), desc=f"Processing q = {q1}"))

# # Perform parallel computation with progress bar for args2
# with mp.Pool(processes=80) as pool:
#     results2 = list(tqdm(pool.imap(binomial_wrapper, args2), total=len(args2), desc=f"Processing q = {q2}"))

# print(results1)
# print(results2)
# print([payoff(option, K, S0) for S0 in S])

# fontsize = 15
# plt.clf()
# plt.title("American Call Price vs. Initial Stock Price", fontsize=fontsize)
# plt.plot(S, results1, label=f'q = {q1}')
# plt.plot(S, results2, label=f'q = {q2}')
# plt.plot(S, [payoff(option, K, S0) for S0 in S], label='Call Intrinsic Value', linestyle='dashed')
# plt.legend(fontsize=fontsize)
# plt.xlabel('S0', fontsize=fontsize)
# plt.ylabel('Call Price', fontsize=fontsize)
# plt.savefig('question4_C-S0.png')

# part 3 report and plot early exercise boundary for American put
# def find_boundary(l1, u1, l2, u2, T, delta):
#     N = int(T / delta)
    
#     S1 = np.linspace(l1, u1, int((u1 - l1) * 100) + 1)
#     S2 = np.linspace(l2, u2, int((u2 - l2) * 100) + 1)
    
#     args1 = [(option, K, T, S0, sigma, r, q1, N, 'A') for S0 in S1]
#     args2 = [(option, K, T, S0, sigma, r, q2, N, 'A') for S0 in S2]
    
#     with mp.Pool(processes=80) as pool:
#         results1 = list(tqdm(pool.imap(binomial_deviation_wrapper, args1), total=len(args1), desc=f"Processing q = {q1}"))
    
#     # Perform parallel computation with progress bar for args2
#     with mp.Pool(processes=80) as pool:
#         results2 = list(tqdm(pool.imap(binomial_deviation_wrapper, args2), total=len(args2), desc=f"Processing q = {q2}"))
    
#     results1 = np.array(results1)
#     results2 = np.array(results2)
#     mask1 = results1[:, 1] < 0.005
#     mask2 = results2[:, 1] < 0.005
# #     print(results1)
# #     print(results2)
#     print(results1[mask1][0], results2[mask2][0])
#     print()
#     return results1[mask1][0][0], results2[mask2][0][0]

# star1, star2 = [], []
# Ts = np.linspace(1/12, 1, 12)
# for T in Ts:
#     print(f'Current T = {T}')
#     try:
#         s1, s2 = find_boundary(100, 160, 100, 160, T, 7e-4) # coarse-grained search
#         print(s1, s2)
#         s1, s2 = find_boundary(s1 - 0.5, s1 + 0.5, s2 - 0.5, s2 + 0.5, T, 7e-5) # fine-grained search
#         print(f'S* = {s1}, {s2}')
#         star1.append(s1)
#         star2.append(s2)
#     except:
#         star1.append(0)
#         star2.append(0)
#         print(f'Error happens in T = {T}, please verify it again later')

# star1 = [,,,,,,,,,,, , 81.40]
# star2 = [,,,,,,,,,,, , 74.31]
star1 = [125.05, 128.71, 132.24, 135.44, 138.26, 140.79, 143.06, 145.13, 147.04, 148.82, 150.46, 152.02]
star2 = [110.51, 113.93, 116.26, 118.07, 119.56, 120.83, 121.94, 122.93, 123.81, 124.61, 125.34, 126.02]
print()
print()
print(star1)
print(star2)
plt.clf()
fontsize = 15
plt.plot(range(len(star1)), star1, label=f'q = {q1}')
plt.plot(range(len(star1)), star2, label=f'q = {q2}')
plt.xlabel('time to maturity T', fontsize=fontsize)
plt.ylabel('early exercise boundary', fontsize=fontsize)
plt.xticks(range(len(star1)), [f'{i}/12' for i in range(1, len(star1) + 1)], fontsize=10)
plt.title('American Call Option Early Exercise Boundary \n vs. Time to Maturity', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig('question4-eeb.png')

end_time = time.time()
print(f'Time elapsed: {end_time - start_time} seconds')