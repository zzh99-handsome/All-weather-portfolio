import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ASSET_RISK = {
    '沪深300': 1/8,
    '中证1000': 1/8,
    'T': 3/8,
    '黄金': 1/8,
    '原油': 1/8,
    '铜': 1/16,
    '螺纹钢': 1/16
}

def calculate_std():
    df_bond_index=pd.read_excel('price/中债国债总财富7-10年指数_1.xlsx')
    df_t00=pd.read_excel('price/T00.CFE.xlsx')
    return df_bond_index['bond'].std(), df_t00['T00'].std()

def get_risk_parity_weight():
    df = get_price_df()
    for column in ASSET_RISK.keys():
        df[column] = np.log(df[column]).diff()
    bond_std, t00_std = calculate_std()
    weight_df = df.describe().T[['std']]/bond_std*t00_std
    weight_df['annual_vol'] = weight_df['std'] * np.sqrt(252)
    weight_df['risk'] = pd.Series(ASSET_RISK)
    weight_df['weight'] = weight_df['risk'] / weight_df['annual_vol']
    weight_df['pct'] = weight_df['weight'].apply(lambda x: x / np.sum(weight_df['weight']))
    return weight_df


def get_price_df():
    df = pd.read_excel('price/沪深300.xlsx')
    for asset in list(ASSET_RISK.keys())[1:]:
        df = pd.merge(df, pd.read_excel(f'price/{asset}.xlsx'), how='outer', on='日期')
    df.sort_values(by='日期', inplace=True)
    return df


def generate_benchmark_pnl():
    df = get_price_df()
    rp_weight = get_risk_parity_weight()['pct'].to_dict()
    print(rp_weight)
    df['total_pct_change'] = 0
    for column in ASSET_RISK.keys():
        df[column + '_pct_change'] = df[column].pct_change()
        df[column + '_weighted_pct_change'] = df[column + '_pct_change'] * rp_weight[column]
        df['total_pct_change'] += df[column + '_weighted_pct_change']
    df.dropna(how='any', inplace=True)
    df['total_pct_change + 1'] = df['total_pct_change'] + 1
    df['nv'] = df['total_pct_change + 1'].cumprod()
    #df.dropna(subset=['nv'], inplace=True)
    plt.plot(df['日期'],df['nv'], color='blue', label='nv')
    plt.xlabel('date')
    plt.ylabel('nv')
    plt.legend()
    plt.show()
    return df


if __name__ == '__main__':
    df = generate_benchmark_pnl()
    df.to_excel('全天候组合.xlsx', index=False)