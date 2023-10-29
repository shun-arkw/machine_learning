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






  
