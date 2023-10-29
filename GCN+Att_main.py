from GCN_Att_net import *
import os
import torch.optim as optim
from GCN_Att_dataloader import LoadData
from torch.utils.data import DataLoader
from utils import Evaluation  # 三种评价指标以及可视化类
from utils import visualize_result
import h5py
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt



interval=10
params_path = 'params/'
params_name = 'net_params-'+str(interval)+'s.pth'


Epoch_list=[100,00]





def main():
    LR = 0.001
    dev = torch.device("cuda:0")
    args = {}
    args['in_v'] = 7
    args['in_n'] = 15
    args['in_time'] = 15
    args['c_in_size'] = 1
    args['v_in_size'] = 2
    args['ou_time'] = interval
    args['ou_size'] = 1
    args['input_embedding_size'] = 128
    args['encoder_size'] = 128
    args['att_size'] = 128
    args['dyn_embedding_size'] = 128
    args['decoder_size'] = 256
    args['dev'] = dev




    train_data = LoadData(train_mode="train",interval=interval)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=12)
    validation_data = LoadData(train_mode="validation",interval=interval)
    validation_loader = DataLoader(validation_data, batch_size=128, shuffle=False, num_workers=12)
    test_data = LoadData(train_mode="test",interval=interval)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=12)


    model = CV_net(args).to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LR)
    # optimizer = optim.SGD(params=model.parameters(), lr=LR,momentum=0.99)


    if os.path.exists((params_path+params_name)):
        model.load_state_dict(torch.load(params_path+params_name))  # 直接调用训练好的模型参数

    Epoch=Epoch_list[0]
    for epoch in range(Epoch):
        model.train()  # 打开训练模式
        train_loss = 0
        train_MAE, train_MAPE, train_RMSE = [], [], []  # 定义三种指标的列表
        for data in train_loader:
            data_x = data['data_x'].to(dev)
            data_y = data['data_y'].to(dev)
            v_graph = data['v_graph'].to(dev)
            cell_data_mask= data['cell_data_mask'].to(dev)
            optimizer.zero_grad()
            y_pred = model(data_x, v_graph,cell_data_mask)
            y_pred = y_pred[:,0]
            data_y = data_y[:,0]


            loss = criterion(y_pred, data_y)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
            optimizer.step()
            train_loss += loss.item()
            performance = compute_performance(y_pred,data_y)


            train_MAE.append(performance[0])
            train_MAPE.append(performance[1])
            train_RMSE.append(performance[2])
        print("Epoch: {:04d},  Train Loss: {:02.4f}  Performance:  MAE:{:2.2f}   MAPE:{:2.4f}    RMSE:{:2.2f}".format(epoch,   train_loss / len(train_data), np.mean(train_MAE), np.mean(train_MAPE), np.mean(train_RMSE)))

        model.eval()  # 打开测试模式
        with torch.no_grad():  # 关闭梯度
            validation_MAE, validation_MAPE, validation_RMSE = [], [], []  # 定义三种指标的列表
            val_loss = 0
            for data in validation_loader:
                data_x = data['data_x'].to(dev)
                data_y = data['data_y'].to(dev)
                v_graph = data['v_graph'].to(dev)
                cell_data_mask = data['cell_data_mask'].to(dev)
                y_pred = model(data_x, v_graph, cell_data_mask)


                y_pred = y_pred[:, 0]
                data_y = data_y[:, 0]

                loss = criterion(y_pred, data_y)
                val_loss += loss.item()
                performance = compute_performance(y_pred,data_y)

                validation_MAE.append(performance[0])
                validation_MAPE.append(performance[1])
                validation_RMSE.append(performance[2])
            print("val Loss: {:02.4f}  Performance:  MAE:{:2.2f}   MAPE:{:2.4f}    RMSE:{:2.2f}".format(val_loss / len(validation_data), np.mean(validation_MAE), np.mean(validation_MAPE), np.mean(validation_RMSE)))

            model_save_judge(val_loss / len(validation_data),model)


    if os.path.exists((params_path+params_name)):
        model.load_state_dict(torch.load(params_path+params_name))  # 直接调用训练好的模型参数




    model.eval()  # 打开测试模式
    with torch.no_grad():  # 关闭梯度
        MAE, MAPE, RMSE = [], [], []  # 定义三种指标的列表
        # TIME=[]
        k=0
        for data in test_loader:
            data_x = data['data_x'].to(dev)
            data_y = data['data_y'].to(dev)
            v_graph = data['v_graph'].to(dev)
            cell_data_mask= data['cell_data_mask'].to(dev)
            y_pred = model(data_x, v_graph,cell_data_mask)


            y_pred = y_pred[:, 0]
            data_y = data_y[:, 0]

            performance = compute_performance(y_pred,data_y)
            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

            # -----------------------huatu----------------------------
            #
            # plt.figure()
            # plt.grid(True, linestyle="-.", linewidth=1)
            # node_id = 0
            # steps = 5000
            # his = data_x[node_id, 0, :, 0]
            #
            #
            # for step in range(steps):
            #     pre_plt = y_pred[node_id + step, :, 0]
            #
            #     if step==0:
            #         dx = [14 + step + t for t in range(len(pre_plt) + 1)]
            #         dy=torch.cat((data_x[node_id+step,0,-1,0].unsqueeze(0),pre_plt),dim=0)
            #         tar_plt = data_y[node_id + step, :, 0]
            #     elif step%interval==0:
            #         dx = dx + [i + dx[-1] + 1 for i in range(len(pre_plt))]
            #         dy=torch.cat((dy,y_pred[node_id + step , :, 0]), dim=0)
            #         tar_plt = torch.cat((tar_plt, data_y[node_id + step, :, 0]), dim=0)
            #
            #
            #
            #     if data_x[node_id+step+1,0,0,0]!=data_x[node_id+step,0,1,0]:
            #         break
            #
            # tar_plt=torch.cat((his,tar_plt),dim=0).cpu().numpy()
            #
            # #-----------------------to csv-------------------------------
            # t=np.array([(i+1)*0.1 for i in range(len(tar_plt))])
            # t=pd.DataFrame(t,columns=['ms'])
            # tar_plt = pd.DataFrame(tar_plt, columns=['km/h']) * (0.3068 * 3.6)
            # dy = pd.DataFrame(torch.cat((his, dy[1:]), dim=0).cpu().numpy(), columns=['km/h']) * (0.3068 * 3.6)
            # tar=pd.concat([t,tar_plt],axis=1)
            # pre=pd.concat([t,dy],axis=1)
            # if os.path.exists(params_path+'data'+str(interval)) == False:
            #     os.mkdir(params_path+'data'+str(interval))
            # tar.to_excel(params_path+'data'+str(interval)+'/'+'tar'+str(k)+'.xlsx',index=None)
            # pre.to_excel(params_path+'data'+str(interval)+'/'+'pre'+str(k)+'.xlsx',index=None)
            #-----------------------to csv-------------------------------


            # plt.plot(np.array([t for t in range(len(tar_plt))]), tar_plt, ls="-", marker=" ", color="b")
            # plt.plot(np.array(dx), dy.cpu().numpy(), ls="-", marker=" ", color="r")
            # plt.legend(["tar", "pre"], loc="upper right")
            # if os.path.exists(params_path+'fig'+str(interval)) == False:
            #     os.mkdir(params_path+'fig'+str(interval))
            # plt.savefig(params_path+'fig'+str(interval)+'/'+str(k)+'.jpg')
            k+=1

            # -----------------------huatu----------------------------





        print("Performance:  MAE:{:2.2f}   MAPE:{:2.4f}    RMSE:{:2.2f}".format(np.mean(MAE),np.mean(MAPE),np.mean(RMSE)))




def adjust_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def compute_performance(prediction,target):  # 计算模型性能


    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    # 对三种评价指标写了一个类，这个类封装在另一个文件中，在后面
    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标
    performance = [mae, mape, rmse]
    return performance  # 返回评价结果


def model_save_judge(val_loss,model):
    now_loss = val_loss
    if os.path.exists(params_path + 'performance' + str(interval) + '.csv'):
        read = pd.read_csv(params_path + 'performance' + str(interval) + '.csv', names=['loss'])
        min_loss = float(read['loss'][len(read) - 1])
    else:
        min_loss = 99999

    if now_loss <= min_loss:
        min_loss = now_loss
        torch.save(model.state_dict(), (params_path + params_name))

    performance = np.array([min_loss]).reshape(-1, 1)
    write = pd.DataFrame(data=performance, columns=['loss'])
    write.to_csv(params_path + 'performance' + str(interval) + '.csv', index=False, columns=['loss'])


if __name__ == '__main__':
    main()