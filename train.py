import torch
from dataset import MNIST
from clip import CLIP
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*libiomp5md.dll.*")  # 忽略OpenMP警告

# 文本编码器：Transformer → 简单Embedding
# 图像编码器：ViT/ResNet-50 → 小型ResNet
def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备
    print(f"使用设备: {DEVICE}")

    dataset = MNIST()  # 数据集
    model = CLIP().to(DEVICE)  # 模型

    try:  # 加载模型
        model.load_state_dict(torch.load('model.pth'))
        print("加载已有模型成功")
    except:
        print("未找到已有模型，从头开始训练")
        pass

    optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器

    '''
        训练模型
    '''
    ITER_BATCH_COUNT = 100000  # 迭代次数
    BATCH_SIZE = 64  # 从batch内选出10个不一样的数字
    TARGET_COUNT = 10  # 共10种数字

    # 用于记录损失值
    loss_history = []
    iteration_points = []

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # 数据加载器

    print("开始训练...")# 从batch中选出10个不同数字的样本
    for i in range(ITER_BATCH_COUNT):
        while True:
            imgs, labels = next(iter(dataloader))
            if torch.unique(labels).shape[0] < TARGET_COUNT:  # 未覆盖10种数字
                continue
            # 挑选出10个数字
            target = set()
            indexes = []
            for j in range(BATCH_SIZE):
                if labels[j].item() in target:
                    continue
                target.add(labels[j].item())
                indexes.append(j)
                if len(target) == TARGET_COUNT:
                    break
            imgs = imgs[indexes]
            labels = labels[indexes]
            break

        logits = model(imgs.to(DEVICE), labels.to(DEVICE))

        targets = torch.arange(0, TARGET_COUNT).to(DEVICE)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.permute(1, 0), targets)
        loss = (loss_i + loss_t) / 2

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        # optimizer.zero_grad(): 将模型参数的梯度清零。因为PyTorch默认会累积梯度，所以在每次反向传播之前需要将梯度归零，否则梯度会累加上一次反向传播的梯度。
        # loss.backward(): 执行反向传播，计算损失函数对模型参数的梯度。即根据损失函数，计算每个参数的梯度值。
        # optimizer.step(): 根据反向传播得到的梯度更新模型参数。优化器（如Adam）会按照其优化算法来更新参数。

        # 记录损失值
        if i % 100 == 0:  # 每100次迭代记录一次
            loss_history.append(loss.item())
            iteration_points.append(i)

        if i % 1000 == 0:
            print('iter:{},loss:{}'.format(i, loss))
            torch.save(model.state_dict(), '.model.pth')
            os.replace('.model.pth', 'model.pth')

    print("训练完成!")

    # 训练结束后绘制损失曲线
    try:
        if loss_history:  # 确保有损失数据
            plt.figure(figsize=(10, 6))
            plt.plot(iteration_points, loss_history)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
            plt.savefig('training_loss.png')
            print("损失曲线已保存为 'training_loss.png'")

            # 保存损失数据到文件
            np.save('loss_history.npy', np.array(loss_history))
            np.save('iteration_points.npy', np.array(iteration_points))
            print("损失数据已保存为 'loss_history.npy' 和 'iteration_points.npy'")
        else:
            print("没有记录到损失数据")
    except Exception as e:
        print(f"绘制图表时出错: {e}")
        # 即使绘图失败，也保存损失数据
        np.save('loss_history.npy', np.array(loss_history))
        np.save('iteration_points.npy', np.array(iteration_points))
        print("损失数据已保存为 'loss_history.npy' 和 'iteration_points.npy'")


if __name__ == '__main__':
    # 设置环境变量以避免OpenMP冲突
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()