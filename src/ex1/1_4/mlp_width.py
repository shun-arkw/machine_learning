import torch.nn as nn
import torch.nn.functional as F
 
class Net(nn.Module):
    def __init__(self, width):   
        super().__init__()
        self.width = width
        '''
        torch.nn.Linearは入力データに対して,線形変換を行う全結合層(affine層,dense層とか)
        第一引数 入力ユニット数  第二引数 出力ユニット数

        '''
        self.fc_input = nn.Linear(28*28, self.width)  # 入力28*28次元, 出力128次元
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc_output = nn.Linear(self.width, 10)
 
    def forward(self, x):
        x = x.view(-1, 28*28)  # 28 x 28の画像を 28*28のベクトルにする
        x = F.relu(self.fc_input(x))
        x = F.relu(self.fc1(x))
        x = self.fc_output(x)
        return x
 