# モジュールのインポート
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# `lenet.py`と`cifar10.py`にあるものを読み込んでいる（アンコメントせよ（シャープを外せ））
from vgg import VGG as Net 
from adversarial_attack import fgsm
from adversarial_attack import pgd
from cifar10 import load_data

import warnings
warnings.filterwarnings("ignore")  # warningを表示しない

#敵対的サンプルに対する防御
#学習の時に敵対的サンプルを作成し，これを学習に使う
def train(net, trainloader, optimizer, criterion, nepochs, epsilon):
    net.train()  # ネットワークを「訓練モード」にする（おまじない）．

    for epoch in range(nepochs):  
        # --- ここを埋める ---------------
        running_loss = 0.0
        for data in trainloader:  # ミニバッチを一つずつ処理していく
            images, labels = data
            images, labels = images.cuda(), labels.cuda() # *** 画像とラベルをGPUに移す（代入が必要） ***

            net.eval()  #モデルのパラメータを固定

            #敵対的サンプルを作る
            adv_images = fgsm(net, images, labels, epsilon)
            

            net.train() #モデルのパラメータを可変に
            optimizer.zero_grad()# オプティマイザの初期化

            # forward + backward + optimize
            #最悪ケースでの誤差を，最小化するようにパラメータを調整
            outputs = net(adv_images)          # 敵対的サンプルをネットワークへの代入
            loss = criterion(outputs, labels)  # ロスの計算 
            loss.backward()                    # 勾配の計算
            optimizer.step()                   # ネットワークパラメタの更新

            running_loss += loss.item()

        running_loss = running_loss / len(trainloader)
        print(f'[epoch {epoch + 1:2d}] loss: {running_loss:.3f}')

        # ------------------------------
    print('Training completed')
    
def test_fgsm(net, dataloader, epsilon):
    net.eval()  # ネットワークを「評価モード」にする（おまじない）．

    correct = 0  # 正解数
    total = 0    # 画像総数

    for data in dataloader:
        # --- ここを埋める ---------------
        images, labels = data
        images, labels = images.cuda(), labels.cuda() # *** 画像とラベルをGPUに移す（代入が必要） ***

        adv_images = fgsm(net, images, labels, epsilon) #敵対データを作る
        outputs = net(adv_images)

        pred_labels = outputs.argmax(dim=1)  # !各画像に関して最大値のインデックスを取り出す
        

        correct += (pred_labels == labels).sum().item()  # `a += b` は `a = a + b`の意味
        total += len(labels)

        # ------------------------------

    acc = correct / total

    return acc 

def test_pgd(net, dataloader, epsilon, alpha, n_iter):
    net.eval()

    correct = 0  # 正解数
    total = 0    # 画像総数

    for data in dataloader:
        # --- ここを埋める ---------------
        images, labels = data
        images, labels = images.cuda(), labels.cuda() # *** 画像とラベルをGPUに移す（代入が必要） ***

        adv_images = pgd(net, images, labels, epsilon, alpha, n_iter) #敵対データを作る
        outputs = net(adv_images)

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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nepochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--save_model_name', type=str, default='')
    parser.add_argument('--attack_method', type= str, default='')
    parser.add_argument('--alpha', type= float, default='0.002')
    parser.add_argument('--n_iter', type= int, default='7')
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
    train(net, trainloader, optimizer, criterion, nepochs = args.nepochs, epsilon =args.eps)





    # ------------------------------

    # --- ネットワークを評価して正解率を表示 ---------- 
    '''
    train_acc = test_fgsm(net, trainloader, args.eps)
    test_acc = test_fgsm(net, testloader, args.eps)
    print(f'train acc = {train_acc:.3f}')  # ':.3f'とつけると小数点以下3桁までの表示になる
    print(f' test acc = {test_acc:.3f}')
    '''

    if args.attack_method == 'fgsm':
        train_acc = test_fgsm(net, trainloader, args.eps)
        test_acc = test_fgsm(net, testloader, args.eps)
        print(f'train acc = {train_acc:.3f}')  # ':.3f'とつけると小数点以下3桁までの表示になる
        print(f' test acc = {test_acc:.3f}') 
    
    elif args.attack_method == 'pgd':
        train_acc = test_pgd(net, trainloader, args.eps, args.alpha, args.n_iter)
        test_acc = test_pgd(net, testloader, args.eps, args.alpha, args.n_iter)
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