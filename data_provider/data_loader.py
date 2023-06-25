import pandas as pd
import numpy as np
import copy
import geopandas as gpd
import transbigdata as tbd
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features

class Taxi_data(Dataset):
    def __init__(self, data_path='/home/Paradise/AI_EX/Data/TaxiData.csv',
                 json_path='/home/Paradise/AI_EX/Data/shenzhenshi.json', 
                 data_len=80000,
                 seq_len=24,
                 seed=3407,
                 thresh_hold_len=100,
                 pre_len=1,
                 split_ratio=[0.7,0.2,0.1],
                 split='train',
                 interval=0,
                 tasks='speed',
                 freq='s'):
                
        self.seq_len = seq_len
        self.pre_len=pre_len
        self.tasks=tasks
        self.split_ratio=split_ratio
        self.split=split
        
        ############# 数据准备阶段#######################
        if data_len !=None:
           data_raw=pd.read_csv(data_path,header = None,nrows = data_len)
        else:
            data_raw=pd.read_csv(data_path,header = None)
        data=copy.deepcopy(data_raw)
        data.columns = ['Car','Ti','Lng','Lat','OpenStatus','Speed']
        city=gpd.read_file(json_path)
        #  除去出深圳的数据
        data = tbd.clean_outofshape(data, city, col=['Lng', 'Lat'], accuracy=400)
        #除去 乘客 0 -1 的突变  
        data = tbd.clean_taxi_status(data, col=['Car', 'Ti', 'OpenStatus'])
        grouped=data.groupby('Car')
        group_len=len(grouped)
        data_list=[]
        if data_len !=None:
            for i in range(group_len):
                    data_list.append((grouped.get_group(22223+i)).values)
        else:
            for i in range(group_len+4):
                if 22223+i in [24357,25093,28321,31249]:
                    pass
                else:
                    data_list.append((grouped.get_group(22223+i)).values)
                
        new_list=[]
        if interval !=0:
            for elem in data_list:
                # 将时间字符串转换为时间戳
                timestamps = np.array([int(pd.Timestamp(t).timestamp()) for t in elem[:, 1]])

                # 每x分钟一个间隔，计算数据应被划分成的间隔数量
                
                num_intervals = int((timestamps[-1] - timestamps[0]) / interval / 60)

                # 划分数据并采样每个时间间隔中的最后一个样本
                result = []
                for i in range(num_intervals):
                    start_time = timestamps[0] + i * interval * 60
                    end_time = start_time + interval * 60
                    indexes = np.where(np.logical_and(timestamps >= start_time, timestamps < end_time))[0]  # 在时间区间内的样本索引
                    if len(indexes) > 0:
                        max_index = indexes[-1]  # 选择时间在区间内的最后一个样本
                        sample = elem[max_index]
                        result.append(sample)
                # 将结果转换为 ndarray
                result_array = np.array(result)
                new_list.append(result_array)
            
                # 通过列表解析式过滤出长度大于我们所设阈值的 numpy 数组
                new_data_list = [arr for arr in new_list if len(arr) >= thresh_hold_len]
                
        else:# 通过列表解析式过滤出长度大于我们所设阈值的 numpy 数组
             new_data_list = [arr for arr in data_list if len(arr) >= thresh_hold_len]
        ################################################
        for i in range(len(new_data_list)):
            new_data_list[i][:,2]-=114
            new_data_list[i][:,3]-=22
        
        ori_data=[]
        for i in range(len(new_data_list)):
            time_fea=time_features(pd.to_datetime(new_data_list[i][:,1]), freq=freq).transpose(1, 0) 
            self.time_fea_dim=time_fea.shape[-1]
            raw_data=np.concatenate((time_fea, new_data_list[i][:,2:]), axis=-1)
            new_data = raw_data[raw_data[:,-1] != 0]
            ori_data.append(new_data)
        
        self.features=[]
        self.targets=[]
        self.tar_time_mark=[]
        for elem in ori_data:
            if len(elem)>seq_len+pre_len:
                for k in range(len(elem)-seq_len-pre_len):
                    self.features.append(elem[k:seq_len+k])
                    self.targets.append(elem[seq_len+k:seq_len+k+pre_len])
        
        # 首先将数据集划分为训练集和测试集
        train_data, self.test_features = train_test_split(self.features, test_size=split_ratio[1], random_state=seed)
        # 然后从训练集中再次划分出一部分作为验证集
        self.train_features,self.val_features = train_test_split(train_data, test_size=split_ratio[2]/(split_ratio[0]+split_ratio[2]), random_state=seed)
        
        # 首先将数据集划分为训练集和测试集
        train_label, self.test_targets = train_test_split(self.targets, test_size=split_ratio[1], random_state=seed)
        # 然后从训练集中再次划分出一部分作为验证集
        self.train_targets,self.val_targets = train_test_split(train_label, test_size=split_ratio[2]/(split_ratio[0]+split_ratio[2]), random_state=seed)
        
    def __len__(self):
        if self.split=='train':
            return len(self.train_features)
        elif self.split =='val':
            return len(self.val_features)
        else :
            return len(self.test_features)
    
    def __getitem__(self, idx):
        
        if self.split=='train':
            
            seq_x= torch.from_numpy(self.train_features[idx][:,self.time_fea_dim:].astype(np.float32))
            seq_y=torch.from_numpy(self.train_targets[idx][:,self.time_fea_dim:].astype(np.float32))
            seq_x_mark= torch.from_numpy(self.train_features[idx][:,:self.time_fea_dim].astype(np.float32))
            seq_y_mark=torch.from_numpy(self.train_targets[idx][:,:self.time_fea_dim].astype(np.float32))
            return seq_x, seq_y, seq_x_mark, seq_y_mark
 
        elif self.split =='val':
            
            seq_x= torch.from_numpy(self.val_features[idx][:,self.time_fea_dim:].astype(np.float32))
            seq_y=torch.from_numpy(self.val_targets[idx][:,self.time_fea_dim:].astype(np.float32))
            seq_x_mark= torch.from_numpy(self.val_features[idx][:,:self.time_fea_dim].astype(np.float32))
            seq_y_mark=torch.from_numpy(self.val_targets[idx][:,:self.time_fea_dim].astype(np.float32))
            return seq_x, seq_y, seq_x_mark, seq_y_mark
    
        else :
            seq_x= torch.from_numpy(self.test_features[idx][:,self.time_fea_dim:].astype(np.float32))
            seq_y=torch.from_numpy(self.test_targets[idx][:,self.time_fea_dim:].astype(np.float32))
            seq_x_mark= torch.from_numpy(self.test_features[idx][:,:self.time_fea_dim].astype(np.float32))
            seq_y_mark=torch.from_numpy(self.test_targets[idx][:,:self.time_fea_dim].astype(np.float32))
            return seq_x, seq_y, seq_x_mark, seq_y_mark
