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


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")

    dataset = MNIST()
    model = CLIP().to(DEVICE)

    try:
        model.load_state_dict(torch.load('model.pth'))
        print("加载已有模型成功")
    except:
        print("未找到已有模型，从头开始训练")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练参数
    EPOCHS = 50  # 训练轮次
    BATCH_SIZE = 64  # 从batch内选出10个不一样的数字
    TARGET_COUNT = 10  # 共10种数字

    # 计算每个epoch需要的迭代次数
    ITER_PER_EPOCH = len(dataset) // TARGET_COUNT

    # 用于记录损失值
    epoch_losses = []

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)

    print("开始训练...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        iter_count = 0

        # 模拟一个epoch：进行ITER_PER_EPOCH次迭代
        for i in range(ITER_PER_EPOCH):
            while True:
                imgs, labels = next(iter(dataloader))
                if torch.unique(labels).shape[0] < TARGET_COUNT:
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iter_count += 1

            # 每100次迭代打印一次进度
            if i % 100 == 0:
                print(f'Epoch: {epoch + 1}/{EPOCHS}, Iter: {i}/{ITER_PER_EPOCH}, Loss: {loss.item()}')

        # 计算该epoch的平均损失
        avg_epoch_loss = epoch_loss / iter_count
        epoch_losses.append(avg_epoch_loss)
        print(f'Epoch: {epoch + 1}/{EPOCHS}, Average Loss: {avg_epoch_loss}')

        # 每个epoch保存一次模型
        torch.save(model.state_dict(), '.model.pth')
        os.replace('.model.pth', 'model.pth')

    print("训练完成!")

    # 绘制损失曲线
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, EPOCHS + 1), epoch_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        print("损失曲线已保存为 'training_loss.png'")

        # 保存损失数据
        np.save('epoch_losses.npy', np.array(epoch_losses))
        print("损失数据已保存为 'epoch_losses.npy'")
    except Exception as e:
        print(f"绘制图表时出错: {e}")
        np.save('epoch_losses.npy', np.array(epoch_losses))


if __name__ == '__main__':
    # 设置环境变量以避免OpenMP冲突
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()