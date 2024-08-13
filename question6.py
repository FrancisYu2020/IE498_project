import numpy as np
import pandas as pd
from tqdm import tqdm
# reuse CRR_Binomial and black scholes model from previous questions
from utils import Binomial, European_option_price
# import pdb
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import torch
import torch.nn as nn

def deAmericanize(option, K, T, S, target_price, r, q, sigma_low, sigma_high, epsilon):
    count = 0
    while 1:
        # binary search on 10 base log uniform space
        sigma_mid = (sigma_low + sigma_high) / 2
        curr_price = European_option_price(option, S, r, q, K, T, sigma_mid)
        # print(f'Current sigma = {sigma_mid}, price diff = {abs(curr_price - target_price)}')
#         count += 1
#         if count > 1000:
#             # the search interval length would be of sigma_high / 2**20, very small
#             # meaning that the sigma is likely to be out of boundary, binary search fails
#             return -1
        if curr_price < target_price - epsilon:
            sigma_low = sigma_mid
        elif curr_price > target_price + epsilon:
            sigma_high = sigma_mid
        else:
            return sigma_mid

class SVI_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.))
        self.b = nn.Parameter(torch.tensor(0.))
        self.rho = nn.Parameter(torch.tensor(0.))
        self.m = nn.Parameter(torch.tensor(0.))
        self.sigma = nn.Parameter(torch.tensor(0.))
    
    def forward(self, k):
        var = self.a + self.b * (self.rho * (k - self.m) + torch.sqrt((k - self.m)**2 + self.sigma**2))
        return var

def get_SVI_param_from_model(model):
    return np.round(np.array([model.a.item(), model.b.item(), model.rho.item(), model.m.item(), model.sigma.item()]), 4)

if __name__ == '__main__':
    
    option = 'C' # C / P
    
    results_path = f'project_{option}_results.csv'
    if os.path.exists(results_path):
        # load precomputed results
        data = pd.read_csv(results_path)
    else:
        data = pd.read_csv('ProjectOptionData.csv')
        
        print(f'Original data size: {len(data)}')
        # prune untraded S and anomaly
        data[f' [{option}_LAST]'] = data[f' [{option}_LAST]'].str.strip()
        data[f' [{option}_LAST]'].replace('', '0', inplace=True)
        data = data[data[f' [{option}_LAST]'] != '0']
        # we only use TTM <= 1 year data and K/S moneyness between 0.5 - 1.5
        data = data[(data[' [DTE]'] <= 365) & (data[' [STRIKE]'] / data[' [UNDERLYING_LAST]'] <= 1.5) & (data[' [STRIKE]'] / data[' [UNDERLYING_LAST]'] >= 0.5)]
        data = data.reset_index(drop=True)
        print(f'Fileter data size: {len(data)}')
        
        S = data[' [UNDERLYING_LAST]'] # 219.19
        K = data[' [STRIKE]']
        data['TTM'] = T = data[' [DTE]'] / 365
        option_price = (data[f' [{option}_ASK]'].astype(float) + data[f' [{option}_BID]'].astype(float)) / 2 # use the middle price
        data[f'{option}_PRICE'] = option_price
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
        data['SIGMA'] = calibrated_sigmas
        data = data[data['SIGMA'] != -1]
        data['K/S'] = data[' [STRIKE]'] / data[' [UNDERLYING_LAST]'] # K/S moneyness
        data = data[[f' [UNDERLYING_LAST]', 'K/S', 'TTM', f'{option}_PRICE', 'SIGMA']]
        data.to_csv(results_path)
    
    # fit svi model
    svi_results_path = f'project_{option}_svi.csv'
    if os.path.exists(svi_results_path):
        data = pd.read_csv(svi_results_path)
    else:
        EPOCHS = 200
        criterion = nn.L1Loss()
        grouped = data.groupby('TTM')
        for group_name, group_df in grouped:
            print(f"Fitting model when TTM = {group_name}")
            model = SVI_Model()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # using SGD for optimization
            model.train()
            k = torch.log(torch.from_numpy(group_df['K/S'].values)) # log-moneyness
            var = torch.from_numpy(group_df['SIGMA'].values)**2 # variance is sigma**2
        
            loss = torch.tensor(float('inf'))
            for epoch in range(EPOCHS):
                pred_var = model(k)
                loss = criterion(pred_var, var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f'In Epoch [{epoch + 1} / {EPOCHS}], loss (mean difference between model predicted variance and market variance) = {loss.item()}')
            model.eval()
            data.loc[group_df.index, 'PRED_SIGMA'] = model(k).detach().numpy()
            print(f'At TTM = {group_name}, (a, b, m, ρ, σ) = ({get_SVI_param_from_model(model)})')
            print("-" * 40)
        data.to_csv(svi_results_path)
            
    # plot the implied volatility surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    market_surf = ax.plot_trisurf(data['K/S'], data['TTM'], data['SIGMA'], cmap='viridis', alpha=0.2, label='market surface')
    svi_surf = ax.plot_trisurf(data['K/S'], data['TTM'], data['PRED_SIGMA'], cmap='plasma', alpha=0.5, label='SVI surface')
    ax.set_xlabel('K/S Moneyness')
    ax.set_ylabel('Time to Maturity (Year)')
    ax.set_zlabel('Implied Volatility (Sigma)')
    ax.set_zlim(0,2)
    option_name = {'C':'Call', 'P':'Put'}
    ax.set_title(f'{option_name[option]} Option Implied Volatility Surface')
    
    fig.colorbar(market_surf, label='market surface')
    fig.colorbar(svi_surf, pad=0.2, label='SVI surface')
    fig.tight_layout()
    plt.savefig('question6.png')
    

