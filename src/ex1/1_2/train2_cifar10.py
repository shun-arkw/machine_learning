# モジュールのインポート
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# `lenet.py`と`cifar10.py`にあるものを読み込んでいる（アンコメントせよ（シャープを外せ））

from vgg import VGG as Net
from cifar10 import load_data

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
    args = parser.parse_args()
    return args 

def main():
    # --- 変更不要 --------------------------------
    # GPU or CPUなのか表示
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)

    # `--nepochs 2` などで与えたパラメタの読み込み
    # `args.nepochs` のようにして使える
    args = set_args()

    
    # --- ここを埋める ---------------
    # --- データをロード ---------------------------
    

    trainloader, testloader, classes = load_data(args.batch_size, use_all=True)

    
    
    
    # --- ネットワークの初期化と学習------------------
    
    


    net = Net('VGG11')  # ネットワークの初期化
    net.cuda()  # *** ネットワークをGPUに移す ***

    #オプティマイザの定義
    optimizer = optim.SGD(net.parameters(), args.lr, momentum=0.9) #確率的勾配法SGD

    #ロス関数の定義
    criterion = nn.CrossEntropyLoss()

    #ネットワークの訓練
    train(net, trainloader, optimizer, criterion, nepochs = args.nepochs)





    # ------------------------------

    # --- ネットワークを評価して正解率を表示 ---------- 
    train_acc = test(net, trainloader)
    test_acc = test(net, testloader)
    print(f'train acc = {train_acc:.3f}')  # ':.3f'とつけると小数点以下3桁までの表示になる
    print(f' test acc = {test_acc:.3f}')  




    # --- 変更不要 --------------------------------
    if args.save_model_name:  # 保存先が与えられている場合保存
        PATH = args.save_model_name
        torch.save(net.state_dict(), PATH)

    #state_dict = torch.load(args.save_model_name)        # 保存したパラメタをロード
    #net.load_state_dict(state_dict)      # ネットワークにパラメタをセット
    #train_acc = test(net, trainloader)
    #test_acc = test(net, testloader)
    #print(f'train acc = {train_acc:.3f}')  # ':.3f'とつけると小数点以下3桁までの表示になる
    #print(f' test acc = {test_acc:.3f}')  


 # おまじない（変更しなくて良い）
if __name__ == '__main__':
    main()  # main()が実行される