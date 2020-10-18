import pandas as pd

df=pd.read_csv('yearly_trading_data_transformed_Momentum_by_sectors.csv')
seperate_size = 10
for i in range(seperate_size):
    if i==seperate_size-1:
        tmp = df.iloc[int(df.shape[0]/seperate_size)*(i):,:]
    else:
        tmp = df.iloc[int(df.shape[0]/seperate_size)*(i):int(df.shape[0]/seperate_size)*(i+1),:]
    tmp.to_csv(f'yearly_trading_data_transformed_Momentum_by_sectors_{i}.csv',index=False)