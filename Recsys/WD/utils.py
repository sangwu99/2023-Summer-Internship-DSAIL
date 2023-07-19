import os 
import pandas as pd 
from sklearn import preprocessing


def age_map(x):
    x = int(x)
    if x < 20:
        return '10'
    elif x >= 20 and x < 30:
        return '20'
    elif x >= 30 and x < 40:
        return '30'
    elif x >= 40 and x < 50:
        return '40'
    elif x >= 50 and x < 60:
        return '50'
    else:
        return '60'
    
def load_dataset(args):
    
    df = pd.read_csv(os.path.join(args.dpath,'u.data'), sep='\t', header=None)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    user2idx = {j:i for i,j in enumerate(df.user_id.unique())}
    item2idx = {j:i for i,j in enumerate(df.item_id.unique())}

    df['user_id'] = df['user_id'].map(user2idx)
    df['item_id'] = df['item_id'].map(item2idx)

    movies_df = pd.read_csv(os.path.join(args.dpath,'u.item'), sep='|', header=None, encoding='latin-1')
    movies_df.columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
                        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                        'Thriller', 'War', 'Western']

    users_df = pd.read_csv(os.path.join(args.dpath,'u.user'), sep='|', encoding='latin-1', header=None)
    users_df.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

    users_df['age'] = users_df['age'].apply(age_map)

    movies_df.drop(['movie_title', 'release_date', 'video_release_date', 'IMDb_URL'], axis=1, inplace=True)
    movies_df['movie_id'] = movies_df['movie_id'].map(item2idx)
    users_df['user_id'] = users_df['user_id'].map(user2idx)

    df.rename(columns={'item_id':'movie_id'}, inplace=True)

    df = pd.merge(df, movies_df,how='left', on = 'movie_id')
    df = pd.merge(df, users_df, how='left',on = 'user_id')

    df.drop(['timestamp', 'zip_code'], axis=1, inplace=True)
    le = preprocessing.LabelEncoder() 
    df['gender'] = le.fit_transform(df['gender'])
    df['age'] = le.fit_transform(df['age'])
    df['occupation'] = le.fit_transform(df['occupation'])
    df['rating'] = [int(i/4) for i in df.rating]
    
    return df