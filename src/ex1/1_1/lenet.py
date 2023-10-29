import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # 初期化の時にレイヤーが準備される部分
    def __init__(self):   
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)          # 畳み込み層
        self.pool = nn.MaxPool2d(2, 2)           # プーリング層
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)    # 全結合層   
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # ネットワークにデータを通す時に機能する
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # 畳み込み -> 活性化 -> プーリング
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)               # テンソルの形を成形
        x = F.relu(self.fc1(x))                  # 全結合 -> 活性化
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x