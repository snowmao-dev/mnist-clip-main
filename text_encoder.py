from torch import nn
import torch
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(num_embeddings=10,embedding_dim=16)# 词嵌入层
        self.dense1=nn.Linear(in_features=16,out_features=64)# 全连接层1
        self.dense2=nn.Linear(in_features=64,out_features=16) #全连接层2
        self.wt=nn.Linear(in_features=16,out_features=8) # 投影层
        self.ln=nn.LayerNorm(8)    # 层归一化

    def forward(self,x):# x形状: (10,)
        # 步骤1: 数字→向量
        x=self.emb(x)
        # 步骤2: 特征学习
        x=F.relu(self.dense1(x)) # (10, 16) → (10, 64)
        x=F.relu(self.dense2(x))# (10, 64) → (10, 16)
        # 步骤3: 空间对齐
        x=self.wt(x)# (10, 16) → (10, 8)
        x=self.ln(x)# 归一化
        return x

if __name__=='__main__':
    text_encoder=TextEncoder()
    x=torch.tensor([1,2,3,4,5,6,7,8,9,0])
    y=text_encoder(x)
    print(y.shape)