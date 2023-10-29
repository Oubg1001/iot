import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from tcn_model import TemporalConvNet









#
class GCN_Att(nn.Module):

    def __init__(self, in_features, out_features,dev):
        super(GCN_Att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features,self.out_features)
        self.a = nn.Linear(2*self.out_features,1)

        self.W.weight.data.normal_(0, 0.01)
        self.a.weight.data.normal_(0, 0.01)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.dp=nn.Dropout(0.3)


    def forward(self, input, adj):
        h =self.W(input) # shape [B,N,out_features]
        N = h.size()[1]
        B = h.size()[0]

        a_input = torch.cat([h.repeat(1,1,N).view(B,N * N, -1), h.repeat(1,N,1)], dim=2).view(B,N,-1,2 * self.out_features) # shape[B,N,N,2*out_features]
        a_input=a_input.view(B,-1,2*self.out_features)  # shape[B,N*N,2*out_features]
        e = self.leakyrelu(self.a(a_input))  # shape[B,N*N,1]
        e = e.view(B,N,-1)

        zero_vec = -1e16*torch.ones_like(e)
        attention = torch.where(adj > 0, e+adj, zero_vec)
        # attention_temp = torch.where(e > 0, e , zero_vec)
        # print(attention_temp[0,:,:])
        attention = F.softmax(attention, dim=-1)
        # print(attention[10,:,:])
        attention=self.dp(attention)
        h_prime1 = torch.matmul(attention, h)  # [B,N,N], [B,N,out_features] --> [B,N,out_features]
        h_prime1=F.elu(h_prime1)


        return h_prime1



# class GCN_Att(nn.Module):
#
#     def __init__(self, in_features, out_features,dev):
#         super(GCN_Att, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#
#         self.W1 = nn.Linear(in_features,self.out_features)
#         self.dp = nn.Dropout(0.5)
#
#
#
#     def forward(self, input, adj):
#         h =self.W1(input) # shape [B,N,out_features]
#         h=F.relu(self.dp(torch.bmm(adj,h)))
#
#
#         return h





class Embedding(nn.Module):
    def __init__(self,in_size,emb_dize,dev):
        super(Embedding, self).__init__()
        self.embedding = nn.Linear(in_size, emb_dize)
        self.device=dev
        self.leakrelu=nn.ReLU()


    def forward(self,x):
        emb=self.leakrelu(self.embedding(x))
        return emb.unsqueeze(2) # B N 1 Emb_size



class V_GA(nn.Module):

    def __init__(self,in_size,hidden_size,heads,dev):
        super(V_GA, self).__init__()

        self.device=dev

        self.att1 = nn.ModuleList([GCN_Att(in_size, hidden_size, self.device) for _ in range(heads)])
        self.fc1 = nn.Linear(heads * hidden_size, hidden_size)



        # self.att1 = GCN_Att(in_size, hidden_size, self.device)




    def forward(self,x,graph):
        #

        att1 = torch.cat([att(x,graph) for att in self.att1],dim=-1) #[B,V,att*n]
        att1=self.fc1(att1) #[B,V,att]
        att1=F.elu(att1)

        # att1=self.att1(x,graph)

        return att1





class c_pro(nn.Module):
    def __init__(self,att_size,device):
        super().__init__()
        self.device = device
        self.att_size = att_size


        # self.conv = nn.Sequential(
        #
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=1,
        #         kernel_size=(10,5),
        #         stride=1,
        #         padding=(1,0),
        #     ),
        #     nn.BatchNorm2d(num_features=1),
        #     nn.ReLU(),
        # )


        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(5,5),
                stride=1,
                padding=(1,1),
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )



        self.trans = nn.Linear(13, att_size)  #cell=15
        # self.trans = nn.Linear(14, att_size)  #cell=16
        # self.trans = nn.Linear(15, att_size)  # cell=17
        # self.trans = nn.Linear(16, att_size)  # cell=18
        # self.trans = nn.Linear(17, att_size)  # cell=19
        # self.trans = nn.Linear(48, att_size)  # cell=50
        # self.trans = nn.Linear(98, att_size)  # cell=100
        # self.trans = nn.Linear(198, att_size)  # cell=200
        # self.trans = nn.Linear(298, att_size)  # cell=300
        # self.trans = nn.Linear(18, att_size)  # cell=20
        # self.trans = nn.Linear(23, att_size)  # cell=25
        # self.trans = nn.Linear(28, att_size)  # cell=30
        # self.trans = nn.Linear(3, att_size)  # cell=5
        # self.trans = nn.Linear(5, att_size)  # cell=7
        # self.trans = nn.Linear(8, att_size)  # cell=10
        # self.trans = nn.Linear(1, att_size)  # cell=3


    def forward(self, x):
        # x [ B N T D]


        x=x.permute([0,3,1,2]) #[ B D N T]
        x_conv=self.conv(x) #  [ B 1 N T]
        x_conv=x_conv.permute([0,3,2,1])  #  [ B T N 1]
        x_conv=x_conv.view(x_conv.shape[0],x_conv.shape[1],-1)  #  [ B T N*1]

        # print(x_conv.shape)
        op=self.trans(x_conv)  #  [ B T(28) att_size]

        return op



class CV_net(nn.Module):
    def __init__(self, args):
        super(CV_net, self).__init__()


        self.device=args['dev']
        self.in_time=args['in_time']
        self.in_v = args['in_v']
        self.in_n = args['in_n']
        self.c_in_size=args['c_in_size']
        self.v_in_size=args['v_in_size']
        self.ou_time=args['ou_time']
        self.ou_size=args['ou_size']
        self.input_embedding_size=args['input_embedding_size']
        self.att_size=args['att_size']
        self.encoder_size=args['encoder_size']
        self.decoder_size=args['decoder_size']
        self.dyn_embedding_size=args['dyn_embedding_size']


        self.c_pro=c_pro(self.att_size,self.device)
        self.enc=TemporalConvNet(self.att_size,[self.encoder_size]*3)


        self.emb=nn.Linear(self.v_in_size,self.input_embedding_size)


        self.enc_t = TemporalConvNet(self.att_size, [self.encoder_size] * 3)


        self.V_GA = V_GA(self.input_embedding_size, self.att_size, 2, self.device)
        self.v_enc=TemporalConvNet(self.att_size, [self.encoder_size]*3)



        self.dec=TemporalConvNet(self.encoder_size*2, [self.decoder_size]*3)
        self.op=nn.Linear(self.decoder_size,1)





    def forward(self,v_data,v_graph,cell_data_mask):
        # v_data [B V T D]

#-------------------------------网格处理模块---------------------------

        c_out=self.c_pro(cell_data_mask)  #  [B T' att_size]
        c_out=c_out.permute([0,2,1]) #  [B att_size T']
        c_out=self.enc(c_out) #  [B encoder_size T']
        c_out=c_out[:,:,-1] #  [B encoder_size]


# -----------------------------embedding模块---------------------------


        emb=v_data.contiguous().view(-1,v_data.shape[2],v_data.shape[3]) # [B*V,T,D]
        embed=self.emb(emb).squeeze(2) # [B*V,T,input_embedding_size]
        embed=embed.permute([0,2,1]) # [B*V,input_embedding_size,T]




# -----------------------------TCN时间编码模块-------------------------------------------------------------
        enc_t = F.relu(self.enc_t(embed) + embed)  # [B*V,encoder_size,T]
        enc_t=enc_t.view(-1,self.in_v,self.encoder_size,self.in_time) # [B,V,encoder_size,T]
        enc_t=enc_t[:,:,:,-1] # [B,V,encoder_size]


# -----------------------------GAT交互模块------------------------------------------------------------





        V_GA = F.relu(self.V_GA(enc_t, v_graph[:,-1]) + enc_t)  # [B,V,att_size]
        V_GA=V_GA.permute([0,2,1]) # [B,att_size,V]

        # V_GA=V_GA[:,:,0]

        v_list=[v for v in range(self.in_v)]
        V_GA=self.v_enc(V_GA[:,:,v_list[::-1]])
        V_GA=V_GA[:,:,-1] # [B,att_size]




# # ------------------------------TCN解码模块---------------------------


        h=torch.cat((V_GA,c_out),dim=1) #  [B,2*encoder_size,1]
        h=h.unsqueeze(2) #  [B,2*encoder_size,1]
        h=h.repeat(1,1,self.ou_time) #  [B,2*encoder_size,pre_t]
        h = self.dec(h)  # [B,decoder_size,pre_t]
        h=h.permute([0,2,1])# [B,pre_t,decoder_size]
        op=self.op(h) # [B,pre_t,1]
        op=op.unsqueeze(1) # [B,1,pre_t,1]




        return op



if __name__ == '__main__':
    dev = torch.device("cuda:0")
    args = {}
    args['in_v'] = 11
    args['in_n'] = 25
    args['in_time'] = 15
    args['c_in_size'] = 1
    args['v_in_size'] = 2
    args['ou_time'] = 1
    args['ou_size'] = 1
    args['input_embedding_size'] = 128
    args['encoder_size'] = 128
    args['att_size'] = 128
    args['dyn_embedding_size'] = 128
    args['decoder_size'] = 256
    args['dev'] = dev

    model = CV_net(args).to(dev)

    x=torch.randn(16,11,15,2).to(dev)
    graph=torch.randn(16,15,11,11).to(dev)
    cell=torch.randn(16,25,15,1).to(dev)
    out=model(x,graph,cell)

