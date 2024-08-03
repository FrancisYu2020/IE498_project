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
    return (S0, ret)

def binomial_deviation_wrapper(args):
    Option, K, T, S0, sigma, r, q, N, Exercise = args
    ret = Binomial(Option, K, T, S0, sigma, r, q, N, Exercise)[0] - payoff(Option, K, S0)
    return (S0, ret)

# global parameters
option = 'P'
K = 100
T = 1
sigma = 0.2
r = 0.05
q1 = 0
q2 = 0.04


# part 1: Find the good time steps
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
#     tasks = [(r, q, K, sigma, T, S0, tolerance) for T in np.linspace(1/12, 1, 12) for S0 in range(75, 95, 1)]
#     selected_delta = np.inf

#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         for result in tqdm(pool.imap_unordered(compute_delta, tasks), total=len(tasks)):
#             S0, delta = result
#             selected_delta = min(selected_delta, delta)
#             tqdm.write(f'S0 = {S0} converges at delta = {delta}')

#     with open(results_path, write_mode) as f:
#         f.write(f'selected delta for q = {q}: {selected_delta} ')

# # # find delta when q = 0
# find_delta(r, q1, K, sigma, 'question3-delta1.txt')
# find_delta(r, q2, K, sigma, 'question3-delta2.txt')


# Part 2: option price vs. S0 plot for 12-month put
# delta = 7.5e-5
# N = int(T / delta)
    
# S = np.arange(70, 150)

# args1 = [(option, K, T, S0, sigma, r, q1, N, 'A') for S0 in S]
# args2 = [(option, K, T, S0, sigma, r, q2, N, 'A') for S0 in S]

# with mp.Pool(processes=80) as pool:
#     results1 = list(tqdm(pool.imap(binomial_wrapper, args1), total=len(args1), desc="Processing q = 0"))

# # Perform parallel computation with progress bar for args2
# with mp.Pool(processes=80) as pool:
#     results2 = list(tqdm(pool.imap(binomial_wrapper, args2), total=len(args2), desc="Processing q = 0.04"))

# print(results1)
# print(results2)
# print([payoff(option, K, S0) for S0 in S])

# fontsize = 15
# plt.title("American Put Price vs. Initial Stock Price", fontsize=fontsize)
# plt.plot(S, results1, label='q = 0')
# plt.plot(S, results2, label='q = 0.04')
# plt.plot(S, [payoff(option, K, S0) for S0 in S], label='Put Intrinsic Value', linestyle='dashed')
# plt.legend(fontsize=fontsize)
# plt.xlabel('S0', fontsize=fontsize)
# plt.ylabel('Put Price', fontsize=fontsize)
# plt.savefig('question3_P-S0.png')

# part 3 report and plot early exercise boundary for American put
# def find_boundary(l1, u1, l2, u2, T, delta):
#     N = int(T / delta)
    
#     S1 = np.linspace(l1, u1, int((u1 - l1) * 100) + 1)
#     S2 = np.linspace(l2, u2, int((u2 - l2) * 100) + 1)
    
#     args1 = [(option, K, T, S0, sigma, r, q1, N, 'A') for S0 in S1]
#     args2 = [(option, K, T, S0, sigma, r, q2, N, 'A') for S0 in S2]
    
#     with mp.Pool(processes=80) as pool:
#         results1 = list(tqdm(pool.imap(binomial_deviation_wrapper, args1), total=len(args1), desc="Processing q = 0"))
    
#     # Perform parallel computation with progress bar for args2
#     with mp.Pool(processes=80) as pool:
#         results2 = list(tqdm(pool.imap(binomial_deviation_wrapper, args2), total=len(args2), desc="Processing q = 0.04"))
    
#     results1 = np.array(results1)
#     results2 = np.array(results2)
#     mask1 = results1[:, 1] < 0.005
#     mask2 = results2[:, 1] < 0.005
# #     print(results1)
# #     print(results2)
#     print(results1[mask1][-1], results2[mask2][-1])
#     print()
#     return results1[mask1][-1][0], results2[mask2][-1][0]

# star1, star2 = [], []
# Ts = np.linspace(1/12, 1, 12)
# for T in Ts:
#     print(f'Current T = {T}')
#     try:
#         s1, s2 = find_boundary(70, 95, 70, 95, T, 5.73e-3) # coarse-grained search
#         print(s1, s2)
#         s1, s2 = find_boundary(s1 - 0.5, s1 + 0.5, s2 - 0.5, s2 + 0.5, T, 5.73e-4) # fine-grained search
#         print(f'S* = {s1}, {s2}')
#         star1.append(s1)
#         star2.append(s2)
#     except:
#         star1.append(0)
#         star2.append(0)
#         print(f'Error happens in T = {T}, please verify it again later')

star1 = [91.32, 88.92, 87.36, 86.19, 85.24, 84.46, 83.78, 83.2, 82.67, 82.21, 81.78, 81.40]
star2 = [88.89, 85.47, 83.21, 81.5, 80.1, 78.94, 77.93, 77.04, 76.26, 75.55, 74.9, 74.31]
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
plt.title('American Put Option Early Exercise Boundary \n vs. Time to Maturity', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig('question3-eeb.png')



# N = 10000

# mem = {}
# lower = 85
# upper = 95.01
# S = np.linspace(lower, upper, int((upper - lower) * 100))
# def find_S_star(l, r, q, epsilon=0.005):
#     '''
#     l: smallest N in the search space
#     r: largest N in the search space
#     '''
# #     print(Binomial(option, K, T, S[r], sigma, r, q, N, 'A')[0])
#     dr = np.abs(Binomial(option, K, T, S[r], sigma, r, q, N, 'A')[0] - payoff(option, K, S[r]))
#     dl = np.abs(Binomial(option, K, T, S[l], sigma, r, q, N, 'A')[0] - payoff(option, K, S[l]))
#     print(S[l], dl, S[r], dr)
#     if dr < epsilon:
#         return -1
#     elif dl > epsilon:
#         return -1
#     elif l == r:
#         return S[l]
#     elif l == r - 1:
#         return S[r] if dr < epsilon else S[l]
#     mid = (l + r) // 2
#     d = np.abs(Binomial(option, K, T, S[mid], sigma, r, q, mid, 'A')[0] - payoff(option, K, S[mid]))
#     if d >= epsilon:
#         return find_S_star(l, mid, q)
#     else:
#         return find_S_star(mid, r, q)

# star = find_S_star(0, len(S) - 1, q1)
# print(star)
# if star == -1:
#     exit()
# print(Binomial(option, K, T, star, sigma, r, q1, N, 'A')[0] - payoff(option, K, star))
# print(Binomial(option, K, T, star + 0.01, sigma, r, q1, N, 'A')[0] - payoff(option, K, star))

end_time = time.time()
print(f'Time elapsed: {end_time - start_time} seconds')