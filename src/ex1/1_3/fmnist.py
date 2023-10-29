import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np

# データロード関数を定義
## 引数batch_sizeはミニバッチの大きさ

def load_data(batch_size, n_train=15000, n_test=2500, use_all=False):

    # クラスのラベル名
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    ## 前処理関数の準備
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))]
        )

    # CIFAR10の準備（ローカルにデータがない場合はダウンロードされる）
    # 訓練用データセット
    trainset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    # 評価用データセット
    testset = FashionMNIST(root='./data', train=False, download=True, transform=transform)


    # --- 学習時間短縮のためデータを選別（本質的でないので気にしなくていい）-------------
    if not use_all:
        trainset.targets = np.asarray(trainset.targets)
        testset.targets = np.asarray(testset.targets)

        classes_id = [trainset.class_to_idx[c] for c in classes]             # クラス名を数値（クラスid）に

        indices = np.where(np.isin(trainset.targets, classes_id))            # 該当クラスの位置
        trainset.data = trainset.data[indices]                               # 該当クラスのデータを抽出
        trainset.targets = trainset.targets[indices]                         # 該当クラスのラベルを抽出
        trainset.targets = [classes_id.index(i) for i in trainset.targets]   # クラスidを0からの連番に  
        
        indices = np.where(np.isin(testset.targets, classes_id))
        testset.data = testset.data[indices]
        testset.targets = testset.targets[indices]
        testset.targets = [classes_id.index(i) for i in testset.targets]

        # trainsetの内，n_train個だけ選ぶ
        trainset, _ = random_split(trainset, [n_train, len(trainset) - n_train])
        # testsetの内，n_test個だけ選ぶ
        testset, _ = random_split(testset, [n_test, len(testset) - n_test])
    # ------------------------------------------------------------------------


    # !ミニバッチに小分けしておく．これを後で使う
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    # !ミニバッチに小分けしておく．これを後で使う
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return (trainloader, testloader, classes)