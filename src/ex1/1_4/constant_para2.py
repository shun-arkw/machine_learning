from mlp_para import Net_6
from fmnist import load_data
from train_fmnist import train, test, set_args

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")  # warningを表示しない

import csv
import time

def main():

    # --- 変更不要 --------------------------------
    # GPU or CPUなのか表示
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)

    # `--nepochs 2` などで与えたパラメタの読み込み
    # `args.nepochs` のようにして使える
    args = set_args()

    with open(args.save_csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["width", "layerNum", "train_acc", "test_acc", "time"])

       
    
            
        # --- ここを埋める ---------------
        # --- データをロード ---------------------------
        

        trainloader, testloader, classes = load_data(args.batch_size, use_all=True)

        
        
        
        # --- ネットワークの初期化と学習------------------
        
        # ネットワークの初期化

        
        net = Net_6()


        
        net.cuda()  # *** ネットワークをGPUに移す ***

        #オプティマイザの定義
        optimizer = optim.SGD(net.parameters(), args.lr, momentum=0.9) #確率的勾配法SGD

        #ロス関数の定義
        criterion = nn.CrossEntropyLoss()


            #学習時間の計測
        torch.cuda.synchronize()
        start = time.time()
        

        #ネットワークの訓練
        train(net, trainloader, optimizer, criterion, nepochs = args.nepochs)

        torch.cuda.synchronize()
        elapsed_time = time.time() - start


        # ------------------------------

        # --- ネットワークを評価して正解率を表示 ---------- 
        train_acc = test(net, trainloader)
        test_acc = test(net, testloader)
        print(f'train acc = {train_acc:.3f}')  # ':.3f'とつけると小数点以下3桁までの表示になる
        print(f' test acc = {test_acc:.3f}')  

        writer.writerow([train_acc, test_acc, elapsed_time])




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