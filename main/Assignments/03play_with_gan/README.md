# Play with GAN

## 运行步骤

 运行训练脚本，启动 GAN 模型训练并生成图像：

    ```bash
    python train.py
    ```


---

## 运行结果

<img src="play_with_gan.gif" alt="alt text" width="800">

**解释**：  
我在FCN_network.py网络中定义了一个生成器和判别器，由于直接用Gan进行训练loss下降很慢，所以我只跑200步的生成模型，然后我才开始Gan的训练，在GIF 动画展示了 GAN 模型从随机噪声中生成的图像逐步演化的过程。随着训练的进行，生成的图像越来越清晰，并逐渐接近目标数据分布，但是当跑到1000步的时候有点过拟合。

---
