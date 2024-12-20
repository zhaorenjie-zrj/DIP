import sys 
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import Generator, Discriminator
from torch.optim.lr_scheduler import StepLR
def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and ensure values are within [0, 255]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)


import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, 
                    criterion_gan, criterion_l1, device, epoch, num_epochs):
    generator.train()
    discriminator.train()
    running_g_loss = 0.0
    running_d_loss = 0.0
    if epoch<400:
        running_loss = 0.0
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # 将数据移动到设备
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # 清零生成器的梯度
            g_optimizer.zero_grad()

            # 前向传播
            outputs = generator(image_rgb)

            # 每 5 个 epoch 保存一次样本图像
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'train_results', epoch)

            # 计算生成器的对抗损失
            loss = criterion_l1(outputs, image_semantic)

            # 反向传播和优化
            loss.backward()
            g_optimizer.step()

            # 更新运行中的损失
            running_loss += loss.item()

            # 打印损失信息
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    else:    
        # 动态调整生成器L1损失权重，逐步增大
        l1_weight = min(100, 10 * (epoch + 1))  # 每个epoch增大10，最大增到100

        # 逐步降低学习率，每5个epoch衰减一次
        if epoch % 5 == 0 and epoch != 0:
            for param_group in g_optimizer.param_groups:
                param_group['lr'] *= 0.8  # 每5个epoch学习率降低20%
            for param_group in d_optimizer.param_groups:
                param_group['lr'] *= 0.8
        
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # 使用标签平滑：将真实标签设为0.9，生成器的标签设为0.1
            real_labels = torch.full((image_rgb.size(0), 1, 30, 30), 0.9, device=device)  # 真实标签
            fake_labels = torch.full((image_rgb.size(0), 1, 30, 30), 0.0, device=device)  # 假标签

            # 判别器训练步骤，降低判别器更新频率
            if i % 3 == 0:
                d_optimizer.zero_grad()

                # 判别器对真实图像的损失
                real_loss = criterion_gan(discriminator(image_rgb, image_semantic), real_labels)

                # 生成图像并计算判别器对假图像的损失
                fake_images = generator(image_rgb)
                fake_loss = criterion_gan(discriminator(image_rgb, fake_images.detach()), fake_labels)

                # 计算判别器的总损失并更新判别器
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()

                # 为判别器使用梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                d_optimizer.step()

                running_d_loss += d_loss.item()

            # 生成器训练步骤
            g_optimizer.zero_grad()

            # 为生成器使用稍微小于1的标签作为对抗性标签，生成器标签平滑
            generator_labels = torch.full((image_rgb.size(0), 1, 30, 30), 0.1, device=device)  # 平滑的生成器标签
            
            # 生成器的对抗损失
            fake_images = generator(image_rgb)
            g_gan_loss = criterion_gan(discriminator(image_rgb, fake_images), generator_labels)

            # 生成器的L1损失，使用动态调整的L1权重
            g_l1_loss = criterion_l1(fake_images, image_semantic) * l1_weight
            
            # 生成器的总损失
            g_loss = g_gan_loss + g_l1_loss
            g_loss.backward()

            # 为生成器加入梯度裁剪
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            g_optimizer.step()

            running_g_loss += g_loss.item()

            # 输出训练信息
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], '
                    f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

        # 每隔5个epoch保存生成效果
        if epoch % 5 == 0:
            save_images(image_rgb, image_semantic, fake_images, 'train_results', epoch)
def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)
    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    train_dir = 'datasets/facades/train'
    val_dir = 'datasets/facades/val'
    train_list_path = 'train_list.txt'
    val_list_path = 'val_list.txt'
    # 获取训练和验证目录中的所有图像文件名
    train_filenames = [f for f in os.listdir(train_dir) if f.endswith('.jpg') or f.endswith('.png')]
    train_filenames.sort() 
    val_filenames = [f for f in os.listdir(val_dir) if f.endswith('.jpg') or f.endswith('.png')]
    val_filenames.sort()  # 排序保持一致性

    # 创建 train_list.txt 和 val_list.txt 文件
    with open(train_list_path, 'w') as train_file, open(val_list_path, 'w') as val_file:
        for filename in train_filenames:
            train_file.write(os.path.join(train_dir, filename) + '\n')
        for filename in val_filenames:
            val_file.write(os.path.join(val_dir, filename) + '\n')

    print("训练集和验证集文件生成完毕！") 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False, num_workers=4)

    # 初始化生成器和判别器
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 对抗损失使用 BCELoss，图像重构损失使用 L1 损失
    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    # 为生成器和判别器设置各自的优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler_g = StepLR(g_optimizer, step_size=200, gamma=0.2)
    scheduler_d = StepLR(d_optimizer, step_size=200, gamma=0.2)

    # 训练循环
    num_epochs = 1500
    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion_gan, criterion_l1, device, epoch, num_epochs)
        validate(generator, val_loader, criterion_l1, device, epoch, num_epochs)

        # 更新学习率
        scheduler_g.step()
        scheduler_d.step()

        # 每隔 20 个 epoch 保存模型
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/pix2pix_generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/pix2pix_discriminator_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()