# Assignment 2 - DIP with PyTorch

### In this assignment, you will implement traditional DIP (Poisson Image Editing) and deep learning-based DIP (Pix2Pix) with PyTorch.

### 1. Implement Poisson Image Editing with PyTorch.
### Running
To run data_possin, run

```basic
python assign2_1.py
```
```

## 2. Pix2Pix implementation.
To pix2, run:
```bash
bash download_facades_dataset.sh
python train.py
```
## Results
### poission 混合：
目前只支持同样大小尺寸的两张图片的poission混合
(https://github.com/zhaorenjie-zrj/DIP/blob/master/main/Assignments/assignment2/data_poission/assign2_1.mp4)

### 全卷积网络:
运行的速度以及收敛的速度与全卷积网络的通道数和批量归一化（Batch Normalization）层有关，我这运行更大的数据集需要很长时间，暂时还没找到好的硬件资源，这是我在那个较小的数据集运行的效果。
(https://github.com/zhaorenjie-zrj/DIP/blob/master/main/Assignments/assignment2/pix2pix/pix2.mp4)
