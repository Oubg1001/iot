import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as scio
from torch.utils.data import DataLoader
from multiprocessing import Pool
np.seterr(divide='ignore',invalid='ignore')
from scipy.signal import savgol_filter
import hdf5storage as h5



class LoadData(Dataset):  # 这个就是把读入的数据处理成模型需要的训练数据和测试数据，一个一个样本能读取出来
    def __init__(self, train_mode,interval):


        self.train_mode = train_mode
        self.train_len = 42749
        self.validation_len = 6107
        self.test_len = 12214
        self.history_len = 15
        self.future_len = interval

        # self.train_mode = train_mode
        # self.train_len = 71631
        # self.validation_len = 10233
        # self.test_len = 20466
        # self.history_len = 15
        # self.future_len = interval



    def __len__(self):  # 表示数据集的长度
        """
        :return: length of dataset (number of samples).
        """
        if self.train_mode == "train":
            return self.train_len
        elif self.train_mode == "validation":
            return self.validation_len
        elif self.train_mode == "test":
            return self.test_len
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index): #  index = [0, Len - 1] 根据上一步的len函数来取index，再根据index取item


        if self.train_mode=="train":
            mode='train'
        elif self.train_mode=='validation':
            mode='validation'
        elif self.train_mode=='test':
            mode='test'


        data_original =scio.loadmat( mode + '/track_slide/' + str(index) + '.mat')['data']
        # for v in range(data_original.shape[0]):
        #     data_original[v,:,3]=savgol_filter(data_original[v,:,3], 5, 3, mode='nearest')


        data_x = data_original[:,0:self.history_len,[3,4]]
        data_x = torch.tensor(data_x, dtype=torch.float)
        data_y = data_original[:,self.history_len:self.history_len + self.future_len,3]
        data_y = torch.tensor(data_y, dtype=torch.float).unsqueeze(2)



        cell_data_mask=scio.loadmat(mode+'/cell_data_mask/'+str(index)+'.mat')['data']
        cell_data_mask=cell_data_mask[:,0:self.history_len,0]
        cell_data_mask=torch.tensor(cell_data_mask, dtype=torch.float)
        cell_data_mask = cell_data_mask.unsqueeze(2)

        # cell_data_mask=cell_data_mask[:,0:self.history_len,:]
        # cell_data_mask=torch.tensor(cell_data_mask, dtype=torch.float)






        v_graph=scio.loadmat('/home/lq/桌面/ou/gat+tcn/'+mode+'/v_graph/'+str(index)+'.mat')['data']
        v_graph=v_graph[0:self.history_len,:,:]
        v_graph = torch.tensor(v_graph, dtype=torch.float)


        return {"data_x": data_x, "data_y": data_y,'v_graph':v_graph,'cell_data_mask':cell_data_mask} #组成词典返回




if __name__ == '__main__':
    train_data = LoadData(train_mode="train",interval=5)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
    print(train_data[0]["data_x"].size())
    print(train_data[0]["data_y"].size())
    print(train_data[0]["v_graph"].size())
    print(train_data[0]["cell_data_mask"].size())




























