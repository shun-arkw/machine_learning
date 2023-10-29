import torch.nn as nn
import torch.nn.functional as F

#クラス定義(nn.Moduleの継承)
class Net(nn.Module):
    def __init__(self, width, layerNum):
        super().__init__()
        self.width = width
        self.layerNum = layerNum
        #self.fc
        self.fcList = []

        

        self.fc_input = nn.Linear(28 * 28, self.width)

        for i in range(0, self.layerNum):
            self.fc = nn.Linear(self.width, self.width)
            self.fcList.append(self.fc)

        self.fc_output = nn.Linear(self.width, 10)

  
        


    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc_input(x))

        for i in range(0, self.layerNum):
            m = self.fcList[i]
            x = F.relu(m(x))

        x = self.fc_output(x)
        return x


class Net_6(nn.Module):
    def __init__(self):   
        super().__init__()
        self.width = 8000
        self.fc1 = nn.Linear(28*28, 8000)  # 入力28*28次元, 出力128次元
        self.fc2 = nn.Linear(8000, 6000)
        self.fc3 = nn.Linear(6000, 4855)
        self.fc4 = nn.Linear(4855, 3512)
        self.fc5 = nn.Linear(3512, 1000)
        self.fc6 = nn.Linear(1000, 500)
        self.fc7 = nn.Linear(500, 100)
        self.fc8 = nn.Linear(100, 10)
 
    def forward(self, x):
        x = x.view(-1, 28*28)  # 28 x 28の画像を 28*28のベクトルにする
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x