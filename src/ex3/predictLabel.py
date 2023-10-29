from vgg import VGG as Net
from jkj1a import load_data, load_data_2, train, test, set_args


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")  # warningを表示しない



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
    

    trainloader, testloader, metric = load_data(args.batch_size)

    
    
    
    # --- ネットワークの初期化と学習------------------
    
    # ネットワークの初期化

    
    net = Net('VGG11')

    
    net.eval() #評価モードにするとパラメータを固定　　テストモードではパラメータが可変
    #torch.load()を用いることで，先ほど学習したネットワークのパラメータを読み込む
    state_dict = torch.load('model/model_jkj1a_flippedImage.pth')  

    #load_state_dict()を利用して，新しいネットワークにパラメータを読み込む
    net.load_state_dict(state_dict)
    
    
    net.cuda()  # *** ネットワークをGPUに移す ***

    #オプティマイザの定義
    optimizer = optim.SGD(net.parameters(), args.lr, momentum=0.9) #確率的勾配法SGD

    #ロス関数の定義
    #criterion = nn.CrossEntropyLoss()
    criterion = lambda x,y: torch.dist(x, y, p = 2)
    #criterion = nn.MS

    

    #ネットワークの訓練
    train(net, trainloader, optimizer, criterion, nepochs = args.nepochs)

   

    # ------------------------------

    # --- ネットワークを評価して正解率を表示 ---------- 
    train_acc = test(net, trainloader, args.value)
    test_acc = test(net, testloader, args.value)
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