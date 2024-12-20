import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GaussianParameters:
    positions: torch.Tensor   # (N, 3) 世界坐标位置
    colors: torch.Tensor      # (N, 3) RGB颜色，范围[0,1]
    opacities: torch.Tensor   # (N, 1) 不透明度值，范围[0,1]
    covariance: torch.Tensor  # (N, 3, 3) 协方差矩阵
    rotations: torch.Tensor   # (N, 4) 四元数
    scales: torch.Tensor      # (N, 3) 对数空间尺度


class GaussianModel(nn.Module):
    def __init__(self, points3D_xyz: torch.Tensor, points3D_rgb: torch.Tensor):
        """
        初始化3D高斯点云模型

        参数:
            points3D_xyz: (N, 3) 张量，点的位置
            points3D_rgb: (N, 3) 张量，RGB颜色，范围[0, 255]
        """
        super().__init__()
        self.n_points = len(points3D_xyz)
        
        # 初始化可学习参数
        self._init_positions(points3D_xyz)
        self._init_rotations()
        self._init_scales(points3D_xyz)
        self._init_colors(points3D_rgb)
        self._init_opacities()

    def _init_positions(self, points3D_xyz: torch.Tensor) -> None:
        """从输入点初始化3D位置"""
        self.positions = nn.Parameter(
            torch.as_tensor(points3D_xyz, dtype=torch.float32)
        )

    def _init_rotations(self) -> None:
        """初始化旋转为单位四元数 [w, x, y, z]"""
        initial_rotations = torch.zeros((self.n_points, 4))
        initial_rotations[:, 0] = 1.0  # w=1, x=y=z=0 表示单位四元数
        self.rotations = nn.Parameter(initial_rotations)

    def _init_scales(self, points3D_xyz: torch.Tensor) -> None:
        """基于局部点密度初始化尺度"""
        # 计算K最近邻的平均距离
        K = min(50, self.n_points - 1)
        points = points3D_xyz.unsqueeze(0)  # 添加批次维度
        dists, _, _ = knn_points(points, points, K=K)
        
        # 使用对数空间进行无约束优化
        mean_dists = torch.mean(torch.sqrt(dists[0]), dim=1, keepdim=True) * 2.
        mean_dists = mean_dists.clamp(
            0.2 * torch.median(mean_dists), 
            3.0 * torch.median(mean_dists)
        )  # 防止尺度无限
        print('init_scales', torch.min(mean_dists), torch.max(mean_dists))
        
        log_scales = torch.log(mean_dists)
        self.scales = nn.Parameter(log_scales.repeat(1, 3))

    def _init_colors(self, points3D_rgb: torch.Tensor) -> None:
        """在logit空间中初始化颜色，以便通过sigmoid激活"""
        # 转换到[0,1]范围并应用logit进行无约束优化
        colors = torch.as_tensor(points3D_rgb, dtype=torch.float32) / 255.0
        colors = colors.clamp(0.001, 0.999)  # 防止logit的无穷
        self.colors = nn.Parameter(torch.logit(colors))

    def _init_opacities(self) -> None:
        """在logit空间中初始化不透明度，以便通过sigmoid激活"""
        # 初始化为高不透明度（sigmoid(8.0) ≈ 0.9997）
        self.opacities = nn.Parameter(
            8.0 * torch.ones((self.n_points, 1), dtype=torch.float32)
        )

    def _compute_rotation_matrices(self) -> torch.Tensor:
        """将四元数转换为3x3旋转矩阵"""
        # 将四元数标准化为单位长度
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q.unbind(-1)
        
        # 构建旋转矩阵的各个元素
        R00 = 1 - 2 * y * y - 2 * z * z
        R01 = 2 * x * y - 2 * w * z
        R02 = 2 * x * z + 2 * w * y
        R10 = 2 * x * y + 2 * w * z
        R11 = 1 - 2 * x * x - 2 * z * z
        R12 = 2 * y * z - 2 * w * x
        R20 = 2 * x * z - 2 * w * y
        R21 = 2 * y * z + 2 * w * x
        R22 = 1 - 2 * x * x - 2 * y * y
        
        # 堆叠并重塑为 (N, 3, 3) 形状
        return torch.stack([
            R00, R01, R02,
            R10, R11, R12,
            R20, R21, R22
        ], dim=-1).reshape(-1, 3, 3)

    def compute_covariance(self) -> torch.Tensor:
        """计算所有高斯点的协方差矩阵"""
        # 获取旋转矩阵，形状为 (N, 3, 3)
        R = self._compute_rotation_matrices()
        
        # 将尺度从对数空间转换回原始尺度，确保其为正，形状为 (N, 3)
        scales = torch.exp(self.scales)
        
        # 构造对角尺度矩阵，形状为 (N, 3, 3)
        S = torch.diag_embed(scales)
        
        # 计算协方差矩阵：Cov = R * S * R^T
        # 步骤1：S * R^T
        SRt = torch.bmm(S, R.transpose(1, 2))  # 形状为 (N, 3, 3)
        
        # 步骤2：R * (S * R^T)
        Covs3d = torch.bmm(R, SRt)  # 形状为 (N, 3, 3)
        
        return Covs3d

    def get_gaussian_params(self) -> GaussianParameters:
        """获取所有高斯参数在世界空间中的表示"""
        return GaussianParameters(
            positions=self.positions,
            colors=torch.sigmoid(self.colors),
            opacities=torch.sigmoid(self.opacities),
            covariance=self.compute_covariance(),
            rotations=F.normalize(self.rotations, dim=-1),
            scales=torch.exp(self.scales)
        )

    def forward(self) -> Dict[str, torch.Tensor]:
        """前向传播，返回参数的字典"""
        params = self.get_gaussian_params()
        return {
            'positions': params.positions,
            'covariance': params.covariance,
            'colors': params.colors,
            'opacities': params.opacities
        }