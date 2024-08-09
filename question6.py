import numpy as np
import pandas as pd
from tqdm import tqdm
# reuse CRR_Binomial and black scholes model from previous questions
from utils import Binomial, European_option_price
# import pdb
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def deAmericanize(option, K, T, S, target_price, r, q, sigma_low, sigma_high, epsilon):
    count = 0
    while 1:
        # binary search on 10 base log uniform space
        sigma_mid = (sigma_low + sigma_high) / 2
        curr_price = European_option_price(option, S, r, q, K, T, sigma_mid)
        # print(f'Current sigma = {sigma_mid}, price diff = {abs(curr_price - target_price)}')
        count += 1
        if count > 1000:
            # the search interval length would be of sigma_high / 2**20, very small
            # meaning that the sigma is likely to be out of boundary, binary search fails
            return -1
        if curr_price < target_price - epsilon:
            sigma_low = sigma_mid
        elif curr_price > target_price + epsilon:
            sigma_high = sigma_mid
        else:
            return sigma_mid

if __name__ == '__main__':
    
    option = 'C' # C / P
    
    results_path = f'project_{option}_results.csv'
    if os.path.exists(results_path):
        # load precomputed results
        data = pd.read_csv(results_path)
    else:
        data = pd.read_csv('ProjectOptionData.csv')
    
        # prune untraded S and anomaly
        data[f' [{option}_LAST]'] = data[f' [{option}_LAST]'].str.strip()
        data[f' [{option}_LAST]'].replace('', '0', inplace=True)
        data = data[data[f' [{option}_LAST]'] != '0'].reset_index(drop=True)

        S = data[f' [UNDERLYING_LAST]'] # 219.19
        K = data[f' [STRIKE]']
        T = data[f' [DTE]'] / 365
        option_price = data[f' [{option}_ASK]'].astype(float)
        r = 0.05
        q = 0
        #     sigma = 0.2
        N = 100
    
        # parameter search space
        sigma_low = 1e-10
        sigma_high = 1e10
        epsilon=1e-6 # error tolerance
        
        calibrated_sigmas = []
        for i in tqdm(range(len(data))):
        #         american_call_price = Binomial(option, K[i], T[i], float(S[i]), sigma, r, q, N, 'A')[0]
    
            calibrated_sigma = deAmericanize(option, K[i], T[i], float(S[i]), option_price[i], r, q, sigma_low, sigma_high, epsilon)
            calibrated_sigmas.append(calibrated_sigma)
#             tqdm.write(f'K = {K[i]}, T = {T[i]}, S = {S[i]}, ask = {option_price[i]}, calibrated sigma = {calibrated_sigma}')
        data['sigma'] = calibrated_sigmas
        data = data[data['sigma'] != -1]
        data = data[[f' [UNDERLYING_LAST]', ' [STRIKE]', ' [DTE]', f' [{option}_ASK]', 'sigma']]
        data.to_csv(results_path)
        
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(data[' [STRIKE]'], data[' [DTE]'], data['sigma'], cmap='viridis')
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Time to Maturity (T day)')
    ax.set_zlabel('Implied Volatility (Sigma)')
    ax.set_title('Implied Volatility Surface')
    plt.savefig('question6.png')

