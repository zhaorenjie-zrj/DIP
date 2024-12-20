import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class GaussianParameters:
    positions: torch.Tensor   # (N, 3) 世界坐标位置
    colors: torch.Tensor      # (N, 3) RGB颜色，范围[0,1]
    opacities: torch.Tensor   # (N, 1) 不透明度值，范围[0,1]
    covariance: torch.Tensor  # (N, 3, 3) 协方差矩阵
    rotations: torch.Tensor   # (N, 4) 四元数
    scales: torch.Tensor      # (N, 3) 对数空间尺度


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # 预先计算像素坐标网格
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32, device='cuda'),
            torch.arange(image_width, dtype=torch.float32, device='cuda'),
            indexing='ij'
        )
        # 形状: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))  # (H, W, 2)


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. 将点从世界空间转换到相机空间
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. 获取投影前的深度，用于排序和裁剪
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. 使用相机内参进行透视投影到屏幕空间
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        
        # 4. 计算透视投影的雅可比矩阵
        # 雅可比矩阵 J_proj = d(x', y') / d(X, Y, Z)
        # x' = (fx * X + cx * Z) / Z
        # y' = (fy * Y + cy * Z) / Z
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        # 计算雅可比矩阵的元素
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        J_proj[:, 0, 0] = fx / cam_points[:, 2]  # dx'/dX
        J_proj[:, 0, 2] = -fx * means3D[:, 0] / (cam_points[:, 2] ** 2)  # dx'/dZ
        J_proj[:, 1, 1] = fy / cam_points[:, 2]  # dy'/dY
        J_proj[:, 1, 2] = -fy * means3D[:, 1] / (cam_points[:, 2] ** 2)  # dy'/dZ
        
        # 5. 将协方差矩阵从世界空间转换到相机空间
        # covs_cam = R * covs3d * R^T
        R_expanded = R.unsqueeze(0).repeat(N, 1, 1)  # (N, 3, 3)
        covs_cam = torch.bmm(R_expanded, torch.bmm(covs3d, R_expanded.transpose(1, 2)))  # (N, 3, 3)
        
        # 6. 投影到2D空间
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # 1. 计算像素点与高斯中心的偏移量，形状为 (N, 2, H, W)
        dx = pixels.unsqueeze(0).permute(0, 3, 1, 2) - means2D.unsqueeze(-1).unsqueeze(-1)  # (N, 2, H, W)
        
        # 2. 增加小的epsilon以确保数值稳定性
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)  # (N, 2, 2)
        
        # 3. 计算高斯分布的行列式和逆矩阵
        det = torch.det(covs2D)  # (N,)
        inv_cov = torch.inverse(covs2D)  # (N, 2, 2)
        
        # 4. 计算高斯概率密度
        # 重塑 dx 以进行批量矩阵乘法
        dx_flat = dx.view(N, 2, -1)  # (N, 2, H*W)
        
        # 计算 Σ^{-1} (x - μ)
        inv_cov_flat = inv_cov.view(N, 2, 2)  # (N, 2, 2)
        tmp = torch.bmm(inv_cov_flat, dx_flat)  # (N, 2, H*W)
        
        # 计算 (x - μ)^T Σ^{-1} (x - μ)
        exponent = torch.sum(tmp * dx_flat, dim=1)  # (N, H*W)
        exponent = exponent.view(N, H, W)  # (N, H, W)
        
        # 计算高斯值
        gaussian = torch.exp(-0.5 * exponent) / (2 * np.pi * det).unsqueeze(-1).unsqueeze(-1)  # (N, H, W)
        
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. 投影到2D，得到 means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t.squeeze(-1))
        
        # 2. 深度掩码
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. 按深度排序
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[indices]        # (N, 3)
        opacities = opacities[indices]  # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. 计算高斯值
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. 应用有效掩码
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha 组合设置
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. 计算权重
        # 使用叠加的alpha进行前向积累
        # 权重公式: alpha_i * prod_{j < i} (1 - alpha_j)
        # 为了高效计算，可以从前到后累积(1 - alpha)的乘积
        alphas_flat = alphas.view(N, -1)  # (N, H*W)
        transmittance = torch.cumprod(1 - alphas_flat + 1e-10, dim=0)  # (N, H*W)
        weights = alphas_flat * torch.roll(transmittance, shifts=1, dims=0)
        weights[:, 0] = alphas_flat[:, 0]  # 第一层的权重直接为alpha
        
        weights = weights.view(N, self.H, self.W)  # (N, H, W)
        
        # 8. 最终渲染
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered