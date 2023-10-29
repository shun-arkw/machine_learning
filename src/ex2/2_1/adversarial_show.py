import numpy as np
import matplotlib.pyplot as plt 
import torchvision


def imshow(img):
    npimg = img.numpy()
    plt.figure(figsize = (10,5))  # 横幅, 縦幅
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 複数の画像を並べて表示する関数
def imshow_list(images): 
    imshow(torchvision.utils.make_grid(images))

def cmp_images(net, images, labels, adv_images, classes, n_show):
    preds1 = net(images).argmax(dim=-1)  # 予測ラベル
    #print(preds1)
    imshow_list(images[:n_show])
    print(' '.join('%5s' % classes[preds1[j]] for j in range(n_show)))

    preds2 = net(adv_images).argmax(dim=-1)  # 予測ラベル
    imshow_list((adv_images)[:n_show])
    print(' '.join('%5s' % classes[preds2[j]] for j in range(n_show)))