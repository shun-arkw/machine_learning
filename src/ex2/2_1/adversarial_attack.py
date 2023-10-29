import torch.nn.functional as F
import numpy as np
import torch


#正しいラベルでの誤差が大きくなるように敵対データを作る
def fgsm(model, x, y, eps):             
    x0 = x.detach().clone()              # 元画像をとっておく
    
    x.requires_grad = True
    outputs = model(x)                   
    loss = F.cross_entropy(outputs, y)   # ロスを計算
    loss.backward()                      # 勾配を計算

    grad = x.grad.detach()               # 勾配
    grad = grad.sign()                   # 全ての値を正負によって+1or-1にする
    x = x0 + eps*grad
    x = x0 + (x-x0).clamp(-eps, eps)     # 各要素の変化の範囲が[-eps, eps]になるよう切り落とす

    x = x.clamp(0, 1)                    # 画素値を[0,1]の範囲に収める
    return x.detach()


#Projected Gradient Descent (PGD)  FGSMより強力な攻撃方法
def pgd(model, x, y, epsilon, alpha, n_iter):
    x0 = x.detach().clone()                  # 元画像をとっておく
    
    for _ in range(n_iter):
        x.requires_grad = True
        outputs = model(x)                   
        loss = F.cross_entropy(outputs, y)   # ロスを計算
        loss.backward()                      # 勾配を計算

        grad = x.grad.data.detach()          # 勾配
        grad = grad.sign()                   # 全ての値を正負によって+1or-1にする
        x = x + alpha * grad
        x = x0 + (x-x0).clamp(-epsilon, epsilon)     # 各要素の変化の範囲が[-eps, eps]になるよう切り落とす

        x = x.clamp(0, 1)                    # 画素値を[0,1]の範囲に収める
        x = x.detach()
        #print(loss.item())
    return x


#指定したラベルに間違えさせる
def targeted_fgsm(model, x, y, eps):             
    x0 = x.detach().clone()              # 元画像をとっておく

    '''
    target_label = np.full(16, y)
    target_label = torch.from_numpy(target_label)
    '''
    y = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    
    x.requires_grad = True
    outputs = model(x)                   
    loss = F.cross_entropy(outputs, y)   # ロスを計算
    loss.backward()                      # 勾配を計算
    print(y)
    grad = x.grad.detach()               # 勾配
    grad = grad.sign()                   # 全ての値を正負によって+1or-1にする
    x = x0 - eps*grad
    x = x0 + (x-x0).clamp(-eps, eps)     # 各要素の変化の範囲が[-eps, eps]になるよう切り落とす

    x = x.clamp(0, 1)                    # 画素値を[0,1]の範囲に収める
    x = x.detach()
    return x



def targeted_pgd(model, x, y, epsilon, alpha, n_iter):
    x0 = x.detach().clone()                  # 元画像をとっておく
    y = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    
    for _ in range(n_iter):
        x.requires_grad = True
        outputs = model(x)                   
        loss = F.cross_entropy(outputs, y)   # ロスを計算
        loss.backward()                      # 勾配を計算

        grad = x.grad.data.detach()          # 勾配
        grad = grad.sign()                   # 全ての値を正負によって+1or-1にする
        x = x - alpha * grad
        x = x0 + (x-x0).clamp(-epsilon, epsilon)     # 各要素の変化の範囲が[-eps, eps]になるよう切り落とす

        x = x.clamp(0, 1)                    # 画素値を[0,1]の範囲に収める
        x = x.detach()
        #print(loss.item())
    return x