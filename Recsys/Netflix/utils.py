import os 
import numpy as np 
import pandas as pd
import torch

def preprocess(dpath='archive-2'):
    dpath = 'archive-2'
    data_list = []
    for i in os.listdir(dpath):
        if 'combined_data' in i:
            data_list.append(i)
            
    df = pd.DataFrame({'Cust_ID','Rating','Timestamp'}) 

    for data in data_list:
        temp_df = pd.read_csv(os.path.join(dpath, data), header = None, names = ['Cust_ID', 'Rating','Timestamp'], usecols = [0,1,2])
        temp_df['Rating'] = temp_df['Rating'].astype(float)
        df = pd.concat([df, temp_df])
        print("Loaded: ", data)

    df.index = np.arange(0,len(df))

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
        # numpy approach
        temp = np.full((1,i-j-1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    # Account for last record and corresponding length
    # numpy approach
    last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
    movie_np = np.append(movie_np, last_record)

    print('Movie numpy: {}'.format(movie_np))
    print('Length: {}'.format(len(movie_np)))

    df = df[pd.notnull(df['Rating'])]
    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_ID'] = df['Cust_ID'].astype(int)
    print(df.iloc[::5000000, :])

    df = df[['Cust_ID', 'Movie_Id', 'Rating','Timestamp']]

    user2idx = {j:i for i,j in enumerate(df['Cust_ID'].unique())}
    item2idx = {j:i for i,j in enumerate(df['Movie_Id'].unique())}
    df['Cust_ID'] = df['Cust_ID'].map(user2idx)
    df['Movie_Id'] = df['Movie_Id'].map(item2idx)

    df.to_csv('preprocessed_df_with_timestamp.csv', index=False)
    
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2)+1e-6)
