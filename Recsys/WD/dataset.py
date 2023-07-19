import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset , DataLoader


class MovieLensWD(Dataset):
    def __init__(self, df, deep_columns, need_dummies):
        self.df = df 
        self.X = df.drop(['rating'], axis=1)
        
        self.deep_df = self.df[deep_columns]
        self.deep = self.deep_df.values
        
        self.wide_df = self.df.drop(need_dummies, axis=1)
        self.wide = self.wide_df.to_numpy(dtype='float32')
        
        self.y = df['rating'].values
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        return self.wide[index], self.deep[index], self.y[index]
    
    def wide_dim(self):
        return len(self.wide_df.columns)
    
    def deep_dims(self):
        embedding_input = [] 
        for column in self.deep_df.columns:
            embedding_input.append(self.deep_df[column].nunique())

        return embedding_input


def load_and_split_dataset(df, args):
    need_dummies = []
    for column in df.columns:
        if df[column].nunique() > 2:
            need_dummies.append(column)
            
    deep_columns = df.drop(columns=['rating'],axis=1).columns
    
    wide_df = pd.get_dummies(df, columns=need_dummies)

    for column in need_dummies:
        wide_df[column] = df[column]
    
    train_X, test_X= train_test_split(wide_df, test_size=0.2, random_state=42)
    
    df_dataset = MovieLensWD(wide_df, deep_columns, need_dummies)

    train_dataset_wd = MovieLensWD(train_X, deep_columns, need_dummies)
    test_dataset_wd = MovieLensWD(test_X, deep_columns, need_dummies)
    
    wide_dim = df_dataset.wide_dim()
    deep_dims = df_dataset.deep_dims()
    
    train_dataloader_wd = DataLoader(train_dataset_wd, batch_size=args.batch_size, shuffle=True)
    test_dataloader_wd = DataLoader(test_dataset_wd, batch_size=args.batch_size, shuffle=True)

    return train_dataloader_wd, test_dataloader_wd, wide_dim, deep_dims
    