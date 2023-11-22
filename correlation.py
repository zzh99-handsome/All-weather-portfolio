import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import warnings
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def test_oil():
    exchange_rate_df = pd.read_excel('price/美元中间价.xlsx')
    oil_df = pd.read_excel("price/原油结算价.xlsx")
    df = pd.merge(left=exchange_rate_df, right=oil_df, how='inner', on='date')
    df.dropna(inplace=True)
    df['Brent_adjusted'] = df['Brent'] * df['exchange_rate']
    df['bias'] = df['Brent_adjusted'] - df['INE']
    get_regression_result(df['Brent_adjusted'], df['INE'], 'Brent_adjusted', 'INE')


def test_bond():
    bond_index = pd.read_excel('price/中债国债总财富7-10年指数.xlsx')
    bond_index1 = pd.read_excel('price/中债国债总财富7-10年指数_1.xlsx')
    r007 = pd.read_excel('price/银行间质押式回购加权利率_7天.xlsx')
    T00 = pd.read_excel('price/T00.CFE.xlsx')
    df = pd.merge(left=bond_index, right=r007, how='inner').merge(T00, how='inner')
    df1=pd.merge(left=bond_index, right=r007, how='left')
    df1=pd.merge(left=df1, right=T00, on='date', how='left')
    df2 = pd.merge(left=bond_index1, right=r007, how='left')
    df2 = pd.merge(left=df2, right=T00, on='date', how='left')
    df.dropna(inplace=True)
    #df2 = df2[::-1]
    #df2 = modify_bond(df2)[::-1]
    base_bond = df['bond'].values[0]
    base_t00 = df['T00'].values[0]
    df1=modify_bond(df1)
    # df1['bond'] = df1['bond'] / df['bond'].values[0] * df['T00'].values[0]
    # df1['interest'] = df1['R007'] / 365
    # df1['accum_interest'] = df1['interest'].cumsum()
    # df1['bond_adjusted'] = df1['bond'] - df1['accum_interest']
    # df2['bond'] = df2['bond'] / base_bond * base_t00
    # df2['interest'] = df2['R007'] / 365
    # df2['accum_interest'] = df2['interest'].cumsum()
    # df2['bond_adjusted'] = df2['bond'] - df2['accum_interest']
    #df3=df2.copy()
    #df3.dropna(how='any',inplace=True)
    #df1=modify_bond(df1,base_bond,base_t00)
    df1_1=df1.loc[0:3304]
    df1_2=df1.loc[3305:]
    df1_2.dropna(inplace=True)
    model = get_regression_result(df1_2.loc[:,'bond_adjusted'], df1_2.loc[:,'T00'], 'bond_adjusted','T00')
    predicted_bond = predict_historical_bond(model, df1_1)
    #df2.dropna(subset=['bond_adjusted'],inplace=True)
    df1_1.loc[:,'T00'] = predicted_bond
    df=pd.concat([df1_1,df1_2])
    draw_bond_price(df)
    df_t00=pd.DataFrame({'日期': df['date'],'T': df['T00']})
    df_t00.to_excel('price/T.xlsx', index=False)



def get_regression_result(series_x, series_y, x_label, y_label):
# Reshape the input series to 2D arrays
    series_x = pd.DataFrame(series_x)
    series_y = pd.DataFrame(series_y)

    # Reshape the data for LinearRegression
    series_x_reshaped = series_x.values.reshape(-1, 1)
    series_y_reshaped = series_y.values.reshape(-1, 1)

    # Create and fit the Linear Regression model
    model = LinearRegression()
    model.fit(series_x_reshaped, series_y_reshaped)

    # Print the coefficients, intercept, R-squared, and Pearson correlation coefficient
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("R-squared:", r2_score(series_y_reshaped, model.predict(series_x_reshaped)))

    corr, p_value = pearsonr(series_x.values.flatten(), series_y.values.flatten())
    print("Pearson correlation coefficient:", corr)
    return model
    
def predict_historical_bond(model, df):
    df.dropna(subset=['bond_adjusted'],inplace=True)
    return model.predict(df['bond_adjusted'].values.reshape(-1, 1))



    

def draw_oil_price():
    exchange_rate_df = pd.read_excel('price/美元中间价.xlsx')
    oil_df = pd.read_excel("price/原油结算价.xlsx")
    df = pd.merge(left=exchange_rate_df, right=oil_df, how='inner', on='date')
    df.dropna(inplace=True)
    df['Brent_adjusted'] = df['Brent'] * df['exchange_rate']
    df['bias'] = df['Brent_adjusted'] - df['INE']
    plt.plot(df['date'], df['Brent_adjusted'], color='blue', label='Brent_adjusted')
    plt.plot(df['date'], df['INE'], color='red', label='INE')
    plt.legend()
    plt.show()

def draw_bond_price(df):
    plt.plot(df['date'], df['bond_adjusted'], color='blue', label='bond_adjusted')
    plt.plot(df['date'], df['T00'], color='red', label='T00')
    plt.legend()
    plt.show()

# def modify_bond(df, base_bond, base_t00):
#     df['interest']=df['R007']/365/100
#     df['interest_discount']=(1-df['interest']).cumprod()
#     df['bond_adjusted']=df['bond']*df['interest_discount']
#     df['T00']=df['bond_adjusted'].apply(lambda x: x/df.loc[df['date']=='2015-03-20','bond_adjusted']*df.loc[df['date']=='2015-03-20','T00'])
#     return df

def modify_bond(df):
    df['interest'] = df['R007'] / 365 / 100
    df['bond_return'] = df['bond'].pct_change() - df['interest']
    df['bond_nv'] = (1 + df['bond_return']).cumprod()
    df['bond_adjusted'] = df['bond_nv'].apply(lambda x: x / df.loc[df['date'] == '2015-03-20', 'bond_nv'] * df.loc[df['date'] == '2015-03-20', 'T00'])
    return df


if __name__ == '__main__':
    test_oil()
    test_bond()
