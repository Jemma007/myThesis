import argparse
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm


raw_log_path = './dataset/data/log_standard_4_08_to_4_21_pure.csv'
user_feature_path = './dataset/data/user_features_pure.csv'
video_feature_path = './dataset/data/video_features_basic_pure.csv'
save_path = './dataset/data/save/'
category_features = (['user_id', 'weekday', 'hourmin', 'user_active_degree', 'is_lowactive_period', 'is_video_author',
                     'follow_user_num_range', 'friend_user_num_range', 'register_days_range'] +
                     [f'onehot_feat{i}' for i in range(18)] +
                     ['video_id', 'author_id', 'video_type', 'upload_type', 'visible_status', 'tag'])
continuous_features = ['duration_ms', 'server_width', 'server_height']
user_features = (['user_id','user_active_degree', 'is_lowactive_period', 'is_video_author',
                     'follow_user_num_range', 'friend_user_num_range', 'register_days_range'] +
                [f'onehot_feat{i}' for i in range(18)])
labels = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'is_not_hate']

class mtlDataSet(data_utils.Dataset):
    def __init__(self, data):
        self.feature = data[0]
        self.label = data[1]

    def __getitem__(self, index):
        feature = self.feature[index]
        label = torch.FloatTensor(self.label[index])
        return feature, label

    def __len__(self):
        return len(self.feature)

def concat_features(raw_log_path, user_feature_path, video_feature_path):
    raw_df = pd.read_csv(raw_log_path)
    user_df = pd.read_csv(user_feature_path)
    video_df = pd.read_csv(video_feature_path)
    raw_df = raw_df.merge(user_df, on=['user_id'])
    raw_df = raw_df.merge(video_df, on=['video_id'])
    # 处理时间
    raw_df['hourmin'] = raw_df['hourmin'] // 100
    # 加入星期几特征
    raw_df['date'] = pd.to_datetime(raw_df['date'], format='%Y%m%d')
    raw_df['weekday'] = raw_df['date'].dt.day_name()
    # 处理视频时长
    raw_df['duration_ms'] = raw_df['duration_ms'] // 100

    # 处理tag
    raw_df['tag'] = raw_df['tag'].apply(lambda x:str(x).split(',')[0])

    # 处理hate标签
    raw_df['is_not_hate'] = 1 - raw_df['is_hate']

    # 处理离散变量
    le = dict()
    for fea in category_features:
        le[fea] = LabelEncoder()
        raw_df[fea] = le[fea].fit_transform(raw_df[fea])

    # 处理连续变量
    mms = MinMaxScaler(feature_range=(0, 1))
    raw_df[continuous_features] = mms.fit_transform(raw_df[continuous_features])

    # 处理成字典
    final_columns = category_features + continuous_features
    categorical_feature_dict, continuous_feature_dict = {}, {}
    for idx, col in tqdm(enumerate(final_columns)):
        if col in category_features:
            raw_df[col] = raw_df[col].map(int)
            categorical_feature_dict[col] = (len(raw_df[col].unique()), idx)
        else:
            continuous_feature_dict[col] = (0, idx)
    return raw_df, categorical_feature_dict, continuous_feature_dict, le


def split_train_test_by_time(df):
    df = df.sort_values('time_ms', ascending=True).reset_index(drop=True)
    final_columns = category_features + continuous_features + labels
    df = df[final_columns]
    # print(df.head())
    train_length = int(df.shape[0]*0.8)
    val_length = int(df.shape[0]*0.1)
    train_df = df[:train_length]
    val_df = df[train_length:train_length + val_length]
    test_df = df[train_length + val_length:]
    train_df.to_csv(save_path + 'train_data.csv', index=False)
    val_df.to_csv(save_path + 'val_data.csv', index=False)
    train_df.to_csv(save_path + 'test_data.csv', index=False)
    del df
    # print(train_df.head())
    return train_df, val_df, test_df

def load_data(train_df, val_df, test_df, args):
    train_dataset = (train_df.loc[:, ~train_df.columns.isin(labels)].values, train_df.loc[:, train_df.columns.isin(labels)].values)
    val_dataset = (val_df.loc[:, ~val_df.columns.isin(labels)].values, val_df.loc[:, val_df.columns.isin(labels)].values)
    test_dataset = (test_df.loc[:, ~test_df.columns.isin(labels)].values, test_df.loc[:, test_df.columns.isin(labels)].values)
    train_dataset = mtlDataSet(train_dataset)
    val_dataset = mtlDataSet(val_dataset)
    test_dataset = mtlDataSet(test_dataset)

    # dataloader
    train_dataloader = get_train_loader(train_dataset, args)
    val_dataloader = get_val_loader(val_dataset, args)
    test_dataloader = get_test_loader(test_dataset, args)
    return train_dataloader, val_dataloader, test_dataloader

def get_train_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, pin_memory=True)
    return dataloader

def get_val_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True)
    return dataloader

def get_test_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    return dataloader
def process_data(args):
    raw_df, categorical_feature_dict, continuous_feature_dict, le = concat_features(raw_log_path, user_feature_path, video_feature_path)
    train_df, val_df, test_df = split_train_test_by_time(raw_df)
    train_dataloader, val_dataloader, test_dataloader = load_data(train_df, val_df, test_df, args)
    return train_dataloader, val_dataloader, test_dataloader, categorical_feature_dict, continuous_feature_dict, user_features, labels, le

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--is_parallel', type=bool, default=False)
    args = parser.parse_args()
    process_data(args)