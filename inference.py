'''
CLIP能力演示

1、对图片做分类
2、对图片求相似图片

'''

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 添加这行来解决OpenMP冲突

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import warnings
import random
warnings.filterwarnings("ignore", message=".*libiomp5md.*")

from dataset import MNIST
from clip import CLIP

# 设置要显示的相似图像数量
SIMILAR_IMAGE_COUNT = 5  # 可以修改这个数字

# 设置要测试的数字（可以修改为0-9之间的任何数字）
TARGET_DIGIT = 2  # 可以改为0, 1, 2, 3, 4, 5, 6, 7, 8, 9

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'   # 设备
print(f"使用设备: {DEVICE}")

dataset = MNIST()  # 数据集

model = CLIP().to(DEVICE)  # 模型
try:
    model.load_state_dict(torch.load('model.pth'))
    print("模型加载成功")
except FileNotFoundError:
    print("未找到模型文件，请先训练模型")
    exit()

model.eval()  # 预测模式

'''
1、对图片分类
'''
try:
    # 找到指定数字的一个样本
    found_index = None
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == TARGET_DIGIT:
            found_index = i
            break

    if found_index is None:
        print(f"未找到数字 {TARGET_DIGIT} 的样本")
        exit()

    image, label = dataset[found_index]
    print(f'测试数字: {label} (索引: {found_index})')

    # 保存图像到文件而不是直接显示
    plt.figure()
    plt.imshow(image.permute(1,2,0))
    plt.axis('off')
    plt.savefig('original_image.png')
    plt.close()
    print("原始图像已保存为 'original_image.png'")

    targets = torch.arange(0, 10)  # 10种分类
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(DEVICE), targets.to(DEVICE))  # 1张图片 vs 10种分类

    predicted_label = logits.argmax(-1).item()
    print(f'CLIP分类: {predicted_label}')
    print(f'分类是否正确: {predicted_label == label}')

    '''
    2、图像相似度
    '''
    # 收集与目标数字相同的其他图像
    other_images = []
    other_labels = []
    other_indices = []

    # 随机选择一些样本（包括目标数字和其他数字）
    for i in range(200):  # 检查200个样本
        idx = random.randint(0, len(dataset)-1)
        other_image, other_label = dataset[idx]
        other_images.append(other_image)
        other_labels.append(other_label)
        other_indices.append(idx)

    # 其他图像的向量
    with torch.no_grad():
        other_img_embs = model.img_enc(torch.stack(other_images, dim=0).to(DEVICE))

    # 当前图片的向量
    with torch.no_grad():
        img_emb = model.img_enc(image.unsqueeze(0).to(DEVICE))

    # 计算当前图片和其他所有图片的相似度
    similarities = img_emb @ other_img_embs.T
    values, indices = similarities[0].topk(SIMILAR_IMAGE_COUNT)  # 使用变量而不是固定数字

    # 保存相似图像到文件
    # 根据要显示的图像数量动态调整图形大小
    fig_width = min(20, 3 * SIMILAR_IMAGE_COUNT)  # 限制最大宽度
    plt.figure(figsize=(fig_width, 3))

    # 计算布局：如果图像数量多，使用多行显示
    if SIMILAR_IMAGE_COUNT <= 5:
        ncols = SIMILAR_IMAGE_COUNT
        nrows = 1
    else:
        ncols = 5
        nrows = (SIMILAR_IMAGE_COUNT + 4) // 5  # 向上取整

    for i, idx in enumerate(indices):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(other_images[idx].permute(1,2,0))
        plt.title(f'Label: {other_labels[idx]}\nSimilarity: {values[i].item():.3f}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('similar_images.png')
    plt.close()
    print(f"最相似的 {SIMILAR_IMAGE_COUNT} 张图像已保存为 'similar_images.png'")

    # 加载保存的损失数据
    try:
        loss_history = np.load('loss_history.npy')
        iteration_points = np.load('iteration_points.npy')

        # 检查数据点数量是否足够进行4000次间隔处理
        if len(loss_history) < 2:
            print("数据点不足，无法进行4000次间隔处理")
        else:
            # 计算每4000次迭代的平均loss
            new_loss_history = []
            new_iteration_points = []

            # 确定记录间隔
            record_interval = 4000

            # 找到第一个记录点的迭代次数
            first_iteration = iteration_points[0]

            # 计算需要多少个点来形成4000次间隔
            for i in range(len(iteration_points)):
                if iteration_points[i] % record_interval == 0:
                    new_iteration_points.append(iteration_points[i])

                    # 计算当前点前后一定范围内的平均值
                    start_idx = max(0, i - 1)
                    end_idx = min(len(loss_history), i + 2)
                    avg_loss = np.mean(loss_history[start_idx:end_idx])
                    new_loss_history.append(avg_loss)

            # 如果没有找到4000次间隔的点，使用原始数据
            if not new_iteration_points:
                new_loss_history = loss_history
                new_iteration_points = iteration_points
                print("未找到4000次间隔的数据点，使用原始数据")
            else:
                print(f"使用每{record_interval}次迭代的平均loss数据")

            # 绘制损失曲线
            plt.figure(figsize=(10, 6))
            plt.plot(new_iteration_points, new_loss_history)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'Training Loss (Averaged every {record_interval} iterations)')
            plt.grid(True)
            plt.savefig('training_loss.png')
            plt.close()
            print("损失曲线已保存为 'training_loss.png'")

    except FileNotFoundError:
        print("未找到损失数据文件。请先运行训练代码以生成损失数据。")

except Exception as e:
    print(f"推理过程中出现错误: {e}")