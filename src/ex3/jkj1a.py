from pathlib import Path
from PIL import Image
import argparse
import os 
import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Subset, Dataset
import numpy as np

def load_data(batch_size, metric='beauty', img_size=32):  # metric is 'beauty' or 'pleasure'

    # 画像への前処理
    transform = transforms.Compose(
            [transforms.Resize(img_size), # original size is 256 x 256
             transforms.ToTensor(),
             ] 
            )
    
    # ラベルに対する前処理（各自で適宜修正）
    target_transform = transforms.Compose(
        [
         transforms.Lambda(torch.tensor),
         #transforms.Lambda(lambda x: x / x.norm())  # xは7次元ベクトル（ヒストグラム）．それの正規化している．
         transforms.Lambda(lambda x: x / torch.sum(x))  # xは7次元ベクトル（ヒストグラム）．各要素が確率を示す．
        ]
    )

    dataset = JKJ1Adataset(
                transform=transform, 
                target_transform=target_transform,
                metric=metric)

    n_test = 50
    n_train = len(dataset) - n_test
    trainset, testset = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloader, testloader, metric

def load_data_2(batch_size, metric='beauty', img_size=32):  # metric is 'beauty' or 'pleasure'

    # 画像への前処理
    transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(p=1),
             transforms.Grayscale(num_output_channels=3),
             transforms.Resize(img_size), # original size is 256 x 256
             transforms.ToTensor(),
            ] 
            )
    
    # ラベルに対する前処理（各自で適宜修正）
    target_transform = transforms.Compose(
        [
         transforms.Lambda(torch.tensor),
         #transforms.Lambda(lambda x: x / x.norm())  # xは7次元ベクトル（ヒストグラム）．それの正規化している．
         transforms.Lambda(lambda x: x / torch.sum(x))  # xは7次元ベクトル（ヒストグラム）．各要素が確率を示す．
        ]
    )

    dataset = JKJ1Adataset(
                transform=transform, 
                target_transform=target_transform,
                metric=metric)

    n_test = 50
    n_train = len(dataset) - n_test
    trainset, testset = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloader, testloader, metric

#ラベルyと予想ラベルの2乗距離を求める．その値が閾値以下である確率．
def test(net, dataloader, value):  #valueは閾値
    net.eval()  # ネットワークを「評価モード」にする（おまじない）．

    correct = 0  # 正解数
    total = 0    # 画像総数

    for data in dataloader:
        # --- ここを埋める ---------------
        images, labels = data
        images, labels = images.cuda(), labels.cuda() # *** 画像とラベルをGPUに移す（代入が必要） ***

        outputs = net(images)

        #pred_labels = outputs.argmax(dim=1)  # !各画像に関して最大値のインデックスを取り出す
        
        pred_labels = outputs

        for label, pred_label in zip(labels, pred_labels):
            dist = float(torch.dist(label, pred_label, p=2))
            
            if abs(dist) < value: #閾値以下ならカウント
                correct += 1  

            total += 1

        # ------------------------------

    acc = correct / total

    return acc 

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
            
            '''
            torch.set_printoptions(edgeitems=1000)
            print(outputs)
            print(labels)
            '''
            
            
            
            loss = criterion(outputs, labels) # ロスの計算 
            #print(loss)
            loss.backward()                    # 勾配の計算
            optimizer.step()                   # ネットワークパラメタの更新

            running_loss += loss.item()

        running_loss = running_loss / len(trainloader)
        print(f'[epoch {epoch + 1:2d}] loss: {running_loss:.3f}')

        # ------------------------------
    print('Training completed')

def set_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nepochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save_model_name', type=str, default='')
    parser.add_argument('--save_csv_name', type = str, default='')   #csvファイルを出力するように
    parser.add_argument('--value', type=float, default=0.3)
    args = parser.parse_args()
    return args 

class JKJ1Adataset(Dataset):
    def __init__(self, 
                 img_dir='data/JKJ1A-dataset/images', 
                 label_dir='data/JKJ1A-dataset/labels',
                 transform=None,
                 target_transform=None,
                 metric='beauty'):
        
        self.img_dir = img_dir
        self.label_dir = os.path.join(label_dir, metric)

        # 画像ファイルのパス一覧を取得する。
        self.images = self.load_images(self.img_dir)
        self.labels = np.load(os.path.join(self.label_dir, 'histogram.npy'))

        self.transform = transform
        self.target_transform = target_transform
        # self.transform = transforms.Compose( 
        #     [transforms.Resize(size), 
        #      transforms.ToTensor(),
        #      ] # original size is 256 x 256
        #     )

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None: 
            label = self.target_transform(label)

        return img, label

    def _get_img_paths(self, img_dir, IMG_EXTENSIONS=[".jpg", ".jpeg", ".png", ".bmp"]):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in IMG_EXTENSIONS
        ]

        return img_paths

    def load_images(self, img_dir):
        img_paths = sorted(self._get_img_paths(img_dir))
        images = [Image.open(path) for path in img_paths]
        return images

    def _get_label_paths(self, label_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        label_dir = Path(label_dir)
        label_paths = [
            p for p in label_dir.iterdir()
        ]

        return label_paths

    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.labels)
