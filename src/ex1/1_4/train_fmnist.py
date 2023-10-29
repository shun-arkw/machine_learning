# モジュールのインポート
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# `lenet.py`と`cifar10.py`にあるものを読み込んでいる（アンコメントせよ（シャープを外せ））


from fmnist import load_data

import warnings
warnings.filterwarnings("ignore")  # warningを表示しない

def train(net, trainloader, optimizer, criterion, nepochs):
    net.train()  # ネットワークを「訓練モード」にする（おまじない）．

    for epoch in range(nepochs):  
        # --- ここを埋める ---------------
        running_loss = 0.0
        for data in trainloader:  # ミニバッチを一つずつ処理していく
            images, labels = data
            images, labels = images.cuda(), labels.cuda() # *** 画像とラベルをGPUに移す（代入が必要） ***

            # オプティマイザの初期化
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)              # ネットワークへの代入
            loss = criterion(outputs, labels)  # ロスの計算 
            loss.backward()                    # 勾配の計算
            optimizer.step()                   # ネットワークパラメタの更新

            running_loss += loss.item()

        running_loss = running_loss / len(trainloader)
        print(f'[epoch {epoch + 1:2d}] loss: {running_loss:.3f}')

        # ------------------------------
    print('Training completed')
    
def test(net, dataloader):
    net.eval()  # ネットワークを「評価モード」にする（おまじない）．

    correct = 0  # 正解数
    total = 0    # 画像総数

    for data in dataloader:
        # --- ここを埋める ---------------
        images, labels = data
        images, labels = images.cuda(), labels.cuda() # *** 画像とラベルをGPUに移す（代入が必要） ***

        outputs = net(images)

        pred_labels = outputs.argmax(dim=1)  # !各画像に関して最大値のインデックスを取り出す
        

        correct += (pred_labels == labels).sum().item()  # `a += b` は `a = a + b`の意味
        total += len(labels)

        # ------------------------------

    acc = correct / total

    return acc 

# これはおまじない
def set_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nepochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save_model_name', type=str, default='')
    parser.add_argument('--save_csv_name', type = str, default='')   #csvファイルを出力するように
    args = parser.parse_args()
    return args 

