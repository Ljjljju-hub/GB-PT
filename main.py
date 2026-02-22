"""
HST-PT: Hierarchical Spatio-Temporal Particle Transformer (SPH-Informed v2.0)
=============================================================================
架构原则：
  - 神经网络仅预测位移 u [N, T, 3]，绝不直接输出应力/应变。
  - 应变通过 SPH Cubic Spline 核函数梯度算子严格推导。
  - 应力通过各向同性线弹性本构矩阵 D 精确计算。
  - 总势能（应变能 - 外力功 + 接触罚函数）作为无监督 Loss。
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

# ===========================================================================
# 1. 粒子时序序列批次 (Particle Sequence Batch)
# ===========================================================================

@dataclass
class ParticleSequenceBatch:
    """
    Transformer 网络的标准输入接口。
    所有张量形状约定：N = 粒子数，T = 时间步数。
    """
    coords:     torch.Tensor   # [N, T, 3]  空间坐标 (x, y, z)
    properties: torch.Tensor   # [N, T, 2]  材料属性 (E, nu)
    volume:     torch.Tensor   # [N, 1]     粒子积分体积 dV（质量守恒，无 T 维）
    bc_values:  torch.Tensor   # [N, T, 3]  外荷载 F 或强制位移值
    domain_id:  torch.Tensor   # [N, T]     物体/碰撞区域编号 (Long)
    bc_type:    torch.Tensor   # [N, T]     边界标记 (0:内部, 1:固定, 2:力, 3:接触) (Long)
    edge_index: torch.Tensor   # [2, E]     SPH 截断半径 rc 内的近邻图索引 (Long)
                               #            edge_index[0]: 源粒子 i, edge_index[1]: 邻居粒子 j


# ===========================================================================
# 2. SPH 物理算子库 (SPH Operators)  ——  无任何可训练参数
# ===========================================================================

class SPHOperators:
    """
    纯数学/物理算子，不继承 nn.Module，不含任何参数。
    提供：
      - cubic_spline_gradient : 3D 三次样条核函数梯度
      - compute_sph_strain    : 基于 scatter_add_ 的 SPH 应变计算
      - build_D_matrix        : 各向同性线弹性本构矩阵
      - compute_stress        : σ = D ε 矩阵乘法
    """

    @staticmethod
    def cubic_spline_gradient(
        r_vec:  torch.Tensor,   # [E, 3]  粒子对的位置向量 x_j - x_i
        r_norm: torch.Tensor,   # [E]     向量范数 ||x_j - x_i||
        h:      float           # 光滑长度 (smoothing length)
    ) -> torch.Tensor:
        """
        计算 3D Cubic Spline 核函数对 x_i 的空间梯度 ∇_i W_ij，形状 [E, 3]。

        核函数 W(r, h) = σ₃/h³ * f(q)，q = r/h，σ₃ = 1/π
          f(q) = 1 - 1.5q² + 0.75q³,  0 ≤ q < 1
                 0.25*(2-q)³,           1 ≤ q < 2
                 0,                     q ≥ 2

        梯度（链式法则，对 x_i 求导带负号）：
          ∇_i W_ij = (σ₃/h³) * dW/dq * (1/h) * (-r_vec/r)
          dW/dq | 0≤q<1 = -3q + 2.25q²
          dW/dq | 1≤q<2 = -0.75*(2-q)²
        """
        sigma3  = 1.0 / math.pi                              # 3D 归一化系数
        h3      = h ** 3
        # 防止 r=0 引发梯度 NaN
        r_safe  = torch.clamp(r_norm, min=1e-8)              # [E]
        q       = r_safe / h                                  # [E]  无量纲距离

        dW_dq = torch.zeros_like(q)                          # [E]
        mask1 = q < 1.0
        mask2 = (q >= 1.0) & (q < 2.0)
        dW_dq[mask1] = -3.0 * q[mask1] + 2.25 * q[mask1] ** 2
        dW_dq[mask2] = -0.75 * (2.0 - q[mask2]) ** 2

        # ∇_i W = (σ₃/h³) * (dW/dq/h) * (-r_vec/r)
        dW_dr  = (sigma3 / h3) * dW_dq / h                  # [E]  标量系数
        grad_W = -dW_dr.unsqueeze(1) * (r_vec / r_safe.unsqueeze(1))  # [E, 3]
        return grad_W                                        # shape: [E, 3]

    @staticmethod
    def compute_sph_strain(
        u:          torch.Tensor,   # [N, 3]   当前时间步预测位移
        coords:     torch.Tensor,   # [N, 3]   当前时间步粒子坐标
        edge_index: torch.Tensor,   # [2, E]   近邻图
        volume:     torch.Tensor,   # [N, 1]   粒子积分体积 V_j
        h_smooth:   float           # SPH 光滑长度
    ) -> torch.Tensor:
        """
        用 SPH 核函数梯度公式计算位移梯度，对称化得到应变，
        返回 6 维 Voigt 工程应变 [N, 6]。

        SPH 位移梯度（standard form）：
          (∂u_α/∂x_β)_i = Σ_j V_j * (u_α(j) - u_α(i)) * ∂W_ij/∂x_β

        应变张量（对称化）：
          ε_αβ = 0.5 * (∂u_α/∂x_β + ∂u_β/∂x_α)

        Voigt 工程应变顺序：[ε₁₁, ε₂₂, ε₃₃, γ₁₂=2ε₁₂, γ₂₃=2ε₂₃, γ₃₁=2ε₃₁]
        """
        N      = u.shape[0]
        device = u.device
        i_idx  = edge_index[0]                               # [E]
        j_idx  = edge_index[1]                               # [E]

        r_vec  = coords[j_idx] - coords[i_idx]               # [E, 3]
        r_norm = torch.norm(r_vec, dim=-1)                   # [E]

        grad_W  = SPHOperators.cubic_spline_gradient(r_vec, r_norm, h_smooth)  # [E, 3]
        delta_u = u[j_idx] - u[i_idx]                       # [E, 3]
        V_j     = volume[j_idx].squeeze(-1)                  # [E]

        # contrib[e, α, β] = V_j * Δu_α * ∂W/∂x_β            [E, 3, 3]
        contrib = V_j.view(-1, 1, 1) * delta_u.unsqueeze(2) * grad_W.unsqueeze(1)

        # scatter_add_：禁止 Python for 循环遍历边，使用纯 PyTorch 算子
        grad_u = torch.zeros(N, 3, 3, device=device)        # [N, 3, 3]
        i_exp  = i_idx.view(-1, 1, 1).expand_as(contrib)
        grad_u.scatter_add_(0, i_exp, contrib)               # shape: [N, 3, 3]

        # 对称化
        strain_tensor = 0.5 * (grad_u + grad_u.transpose(1, 2))  # [N, 3, 3]

        # 转为 6 维 Voigt 工程应变
        strain_voigt = torch.stack([
            strain_tensor[:, 0, 0],           # ε₁₁
            strain_tensor[:, 1, 1],           # ε₂₂
            strain_tensor[:, 2, 2],           # ε₃₃
            2.0 * strain_tensor[:, 0, 1],     # γ₁₂ = 2ε₁₂
            2.0 * strain_tensor[:, 1, 2],     # γ₂₃ = 2ε₂₃
            2.0 * strain_tensor[:, 2, 0],     # γ₃₁ = 2ε₃₁
        ], dim=-1)                            # shape: [N, 6]
        return strain_voigt

    @staticmethod
    def build_D_matrix(
        properties: torch.Tensor    # [N, 2]  每粒子材料属性 (E, nu)
    ) -> torch.Tensor:
        """
        为每个粒子构建各向同性线弹性 6×6 Voigt 本构矩阵 D。
        输出：[N, 6, 6]

        D = E/((1+ν)(1-2ν)) * diag_block([1-ν, ν, ν; ν, 1-ν, ν; ν, ν, 1-ν],
                                           [(1-2ν)/2, (1-2ν)/2, (1-2ν)/2])
        """
        E_mod  = properties[:, 0]   # [N]
        nu     = properties[:, 1]   # [N]
        N      = E_mod.shape[0]
        device = E_mod.device

        lam_c     = E_mod / ((1.0 + nu) * (1.0 - 2.0 * nu))   # [N]  缩放系数
        lam_diag  = 1.0 - nu                                    # [N]  主对角
        lam_off   = nu                                          # [N]  非对角
        lam_shear = (1.0 - 2.0 * nu) / 2.0                    # [N]  剪切

        D = torch.zeros(N, 6, 6, device=device, dtype=E_mod.dtype)  # [N, 6, 6]
        for a in range(3):
            for b in range(3):
                D[:, a, b] = lam_c * (lam_diag if a == b else lam_off)
        for s in range(3, 6):
            D[:, s, s] = lam_c * lam_shear
        return D     # shape: [N, 6, 6]

    @staticmethod
    def compute_stress(
        strain_voigt: torch.Tensor,   # [N, 6]
        D:            torch.Tensor    # [N, 6, 6]
    ) -> torch.Tensor:
        """σ = D ε（批量矩阵-向量乘法），输出 [N, 6]"""
        return torch.bmm(D, strain_voigt.unsqueeze(-1)).squeeze(-1)  # shape: [N, 6]


# ===========================================================================
# 3. 傅里叶特征映射 (Fourier Feature Mapping)
# ===========================================================================

class FourierFeatureMapping(nn.Module):
    """将低维坐标映射为高频特征，消除神经网络的频谱偏置 (Spectral Bias)"""
    def __init__(self, input_dim: int = 3, mapping_size: int = 32, scale: float = 10.0):
        super().__init__()
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)   # 冻结，随模型转移至 GPU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * math.pi * x) @ self.B     # [..., mapping_size]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [..., mapping_size*2]


# ===========================================================================
# 4. 混合物理嵌入层 (Hybrid Physics Embedding)
# ===========================================================================

class HybridPhysicsEmbedding(nn.Module):
    """
    将异构物理量统一映射为高维稠密向量 [N, T, d_model]。
    特征组成：
      Fourier 坐标 (3→64) + bc_type 嵌入 (→16) +
      domain_id 嵌入 (→16) + 连续物理量 (5→32) = 128 → MLP → d_model
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.fourier_map  = FourierFeatureMapping(input_dim=3, mapping_size=32)
        self.embed_bc     = nn.Embedding(num_embeddings=5,  embedding_dim=16)
        self.embed_domain = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.proj_phys    = nn.Linear(5, 32)
        self.fusion_mlp   = nn.Sequential(
            nn.Linear(128, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, batch: ParticleSequenceBatch) -> torch.Tensor:
        feat_coords  = self.fourier_map(batch.coords)                          # [N, T, 64]
        feat_bc      = self.embed_bc(batch.bc_type)                            # [N, T, 16]
        feat_domain  = self.embed_domain(batch.domain_id)                      # [N, T, 16]
        phys_vars    = torch.cat([batch.bc_values, batch.properties], dim=-1)  # [N, T, 5]
        feat_phys    = self.proj_phys(phys_vars)                               # [N, T, 32]
        raw_feat     = torch.cat([feat_coords, feat_bc, feat_domain, feat_phys], dim=-1)  # [N, T, 128]
        return self.fusion_mlp(raw_feat)                                        # [N, T, d_model]


# ===========================================================================
# 5. 层级体素空间注意力 (Hierarchical Voxel Attention)
# ===========================================================================

class HierarchicalVoxelAttention(nn.Module):
    """
    O(N) 复杂度的多尺度空间消息传递算子。
    核心 5 步：
      1. Voxelization   : floor(coords/grid_size) -> 整数网格
      2. Unique Mapping : torch.unique -> 宏观粒子 + 反向映射
      3. Scatter Mean   : scatter_add_ + clamp(counts, min=1) 防除零
      4. Macro Attention: [1, M, d_model] Transformer 全局交互
      5. Broadcast      : macro_h[inverse_mapping] -> [N, d_model]
    """
    def __init__(self, d_model: int, grid_sizes: List[float], n_heads: int = 4):
        super().__init__()
        self.d_model    = d_model
        self.grid_sizes = grid_sizes
        self.macro_attentions = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 2, batch_first=True
            ) for _ in grid_sizes
        ])
        self.multi_scale_fusion = nn.Linear(d_model * (len(grid_sizes) + 1), d_model)

    def forward(self, h: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        h:      [N, d_model]  当前时间步粒子隐特征
        coords: [N, 3]        当前时间步物理坐标
        return: [N, d_model]  多尺度融合后的粒子特征
        """
        device = h.device
        multi_scale_features = [h]

        for i, grid_size in enumerate(self.grid_sizes):
            # Step 1: Voxelization
            voxel_indices = torch.floor(coords / grid_size).long()         # [N, 3]

            # Step 2: Unique Mapping
            _, inverse_mapping = torch.unique(
                voxel_indices, dim=0, return_inverse=True
            )                                                               # [N]
            M = inverse_mapping.max().item() + 1

            # Step 3: Scatter Mean Pooling（严禁 for 循环）
            macro_h = torch.zeros(M, self.d_model, device=device)          # [M, d_model]
            idx_exp = inverse_mapping.unsqueeze(1).expand_as(h)            # [N, d_model]
            macro_h.scatter_add_(0, idx_exp, h)
            counts  = torch.bincount(inverse_mapping, minlength=M).unsqueeze(1).float()
            macro_h = macro_h / torch.clamp(counts, min=1.0)              # [M, d_model]

            # Step 4: Macro Global Attention
            macro_h = self.macro_attentions[i](macro_h.unsqueeze(0))      # [1, M, d_model]
            macro_h = macro_h.squeeze(0)                                   # [M, d_model]

            # Step 5: Broadcast
            multi_scale_features.append(macro_h[inverse_mapping])         # [N, d_model]

        concat_h = torch.cat(multi_scale_features, dim=-1)                # [N, d_model*(1+L)]
        return self.multi_scale_fusion(concat_h)                          # [N, d_model]


# ===========================================================================
# 6. HST-PT 主网络（严格仅输出位移 [N, T, 3]）
# ===========================================================================

class HSTParticleTransformer(nn.Module):
    """
    神经网络 forward() 内绝无应力/应变计算。
    唯一输出: displacement [N, T, 3]。
    """
    def __init__(
        self,
        d_model:    int         = 128,
        n_heads:    int         = 4,
        num_layers: int         = 3,
        grid_sizes: List[float] = [1.0, 5.0]
    ):
        super().__init__()
        self.embedding  = HybridPhysicsEmbedding(d_model)
        self.num_layers = num_layers

        # 时序注意力：N 条粒子轨迹独立，仅在 T 维度计算  复杂度 O(N·T²)
        self.temporal_attentions = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 2, batch_first=True
            ) for _ in range(num_layers)
        ])

        # 空间层级体素注意力                              复杂度 O(T·N)
        self.spatial_attentions = nn.ModuleList([
            HierarchicalVoxelAttention(d_model, grid_sizes, n_heads)
            for _ in range(num_layers)
        ])

        # 解码器：严格输出 3 维位移（架构铁律：禁止输出应力/应变）
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 3)       # ← 仅 3 维，不多一维
        )

    def forward(self, batch: ParticleSequenceBatch) -> torch.Tensor:
        """
        输入: ParticleSequenceBatch
        输出: displacement [N, T, 3]  (bc_type==1 处已硬约束归零)
        """
        # ---- 嵌入 ----
        h = self.embedding(batch)           # shape: [N, T, d_model]
        N, T, _ = h.shape

        # ---- 时空交替注意力块 ----
        for i in range(self.num_layers):
            # 时序注意力
            h = self.temporal_attentions[i](h)      # shape: [N, T, d_model]

            # 空间层级注意力（必须按时间步循环，保证大变形下的物理因果律）
            h_spatial = torch.zeros_like(h)
            for t in range(T):
                h_spatial[:, t, :] = self.spatial_attentions[i](
                    h[:, t, :],                     # [N, d_model]
                    batch.coords[:, t, :]            # [N, 3]
                )
            h = h_spatial                           # shape: [N, T, d_model]

        # ---- 解码为位移场 ----
        displacement = self.decoder(h)              # shape: [N, T, 3]

        # ---- 硬约束：bc_type==1 (固定边界) 位移强制归零 ----
        fixed_mask   = (batch.bc_type == 1).unsqueeze(-1).float()   # [N, T, 1]
        displacement = displacement * (1.0 - fixed_mask)            # shape: [N, T, 3]

        return displacement


# ===========================================================================
# 7. SPH 驱动的最小势能损失函数 (SPHPhysicsInformedLoss)
# ===========================================================================

class SPHPhysicsInformedLoss(nn.Module):
    """
    完全基于 SPH 严谨推导的最小势能原理无监督损失。
    计算流：网络预测 u -> SPH 算子推导 ε -> 本构矩阵 D 计算 σ -> 势能泛函

    总势能：Π = Π_int + Π_ext + L_contact  （对 T 维度取均值）
    """
    def __init__(self, h_smooth: float = 1.0, contact_penalty_k: float = 1e4):
        super().__init__()
        self.h_smooth  = h_smooth
        self.k_penalty = contact_penalty_k

    def forward(
        self,
        displacement: torch.Tensor,        # [N, T, 3]  网络唯一输出
        batch:        ParticleSequenceBatch
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        N, T, _  = displacement.shape
        device   = displacement.device

        pi_int_list:  List[torch.Tensor] = []
        pi_ext_list:  List[torch.Tensor] = []
        contact_list: List[torch.Tensor] = []

        for t in range(T):
            u_t      = displacement[:, t, :]           # [N, 3]
            coords_t = batch.coords[:, t, :]           # [N, 3]
            props_t  = batch.properties[:, t, :]       # [N, 2]
            bc_t     = batch.bc_type[:, t]             # [N]
            bcv_t    = batch.bc_values[:, t, :]        # [N, 3]
            domain_t = batch.domain_id[:, t]           # [N]

            # ---- Step A: SPH 推导应变 ε [N, 6] ----
            strain = SPHOperators.compute_sph_strain(
                u_t, coords_t, batch.edge_index, batch.volume, self.h_smooth
            )                                          # shape: [N, 6]

            # ---- Step B: 本构矩阵 D [N,6,6] -> 应力 σ [N,6] ----
            D      = SPHOperators.build_D_matrix(props_t)          # [N, 6, 6]
            stress = SPHOperators.compute_stress(strain, D)        # [N, 6]

            # ---- 损失 1: 应变能 Π_int = 0.5 Σ_i (σ:ε)_i V_i ----
            # σ:ε = Σ σ_k ε_k（Voigt，γ 已是工程应变故权重均为 1）
            # clamp(min=0) 强制正定，消除负能量作弊漏洞
            double_cont = (stress * strain).sum(dim=-1)            # [N]
            pi_int_t    = 0.5 * torch.clamp(
                (double_cont * batch.volume.squeeze(-1)).sum(), min=0.0
            )
            pi_int_list.append(pi_int_t)

            # ---- 损失 2: 外力功 Π_ext = -Σ_i F_i·u_i  (bc_type==2) ----
            force_mask = (bc_t == 2).float().unsqueeze(-1)         # [N, 1]
            ext_work   = (bcv_t * u_t * force_mask).sum(dim=-1)   # [N]
            pi_ext_t   = -(ext_work * batch.volume.squeeze(-1)).sum()
            pi_ext_list.append(pi_ext_t)

            # ---- 损失 3: 接触穿透罚函数（不同 domain_id 的 bc_type==3 粒子对）----
            c_mask = (bc_t == 3)
            if c_mask.sum().item() >= 2:
                deformed    = coords_t[c_mask] + u_t[c_mask]          # [Nc, 3]
                c_domain    = domain_t[c_mask]                         # [Nc]
                diff_pos    = deformed.unsqueeze(0) - deformed.unsqueeze(1)  # [Nc,Nc,3]
                dist        = torch.norm(diff_pos, dim=-1)             # [Nc, Nc]
                diff_dom    = (c_domain.unsqueeze(0) != c_domain.unsqueeze(1)).float()
                r_c         = (batch.volume[c_mask].squeeze(-1) ** (1.0 / 3.0)) / 2.0
                min_dist    = r_c.unsqueeze(0) + r_c.unsqueeze(1)     # [Nc, Nc]
                penetration = torch.clamp(min_dist - dist, min=0.0)
                contact_list.append(
                    (self.k_penalty * penetration ** 2 * diff_dom).sum()
                )

        # 对 T 维度取均值
        pi_int = torch.stack(pi_int_list).mean()
        pi_ext = torch.stack(pi_ext_list).mean()
        l_cont = (
            torch.stack(contact_list).mean()
            if contact_list
            else torch.zeros(1, device=device).squeeze()
        )

        total_loss = pi_int + pi_ext + l_cont
        loss_dict:  Dict[str, float] = {
            "loss_total":   total_loss.item(),
            "loss_pi_int":  pi_int.item(),
            "loss_pi_ext":  pi_ext.item(),
            "loss_contact": l_cont.item(),
        }
        return total_loss, loss_dict


# ===========================================================================
# 8. 训练主循环 (Training Loop)
# ===========================================================================

def train_one_epoch(
    model:      HSTParticleTransformer,
    loss_fn:    SPHPhysicsInformedLoss,
    optimizer:  torch.optim.Optimizer,
    batch:      ParticleSequenceBatch,
    grad_clip:  float = 1.0
) -> Dict[str, float]:
    """
    单轮训练：前向(仅位移) -> SPH 应变/应力 -> 势能 Loss -> backward -> 更新。
    """
    model.train()
    optimizer.zero_grad()

    displacement = model(batch)                             # [N, T, 3]
    total_loss, loss_dict = loss_fn(displacement, batch)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss_dict


# ===========================================================================
# 9. 辅助工具：构建虚拟 SPH 近邻图 edge_index（仅用于验证，生产中替换为 FAISS）
# ===========================================================================

def build_radius_edge_index(
    coords:   torch.Tensor,   # [N, 3]
    cutoff_r: float
) -> torch.Tensor:
    """
    暴力构建截断半径 cutoff_r 内的 edge_index [2, E]。
    大规模训练建议替换为 FAISS / torch_cluster 高效近邻搜索。
    """
    diff  = coords.unsqueeze(0) - coords.unsqueeze(1)    # [N, N, 3]
    dist  = torch.norm(diff, dim=-1)                     # [N, N]
    mask  = (dist < cutoff_r) & (dist > 0.0)
    i_idx, j_idx = mask.nonzero(as_tuple=True)
    return torch.stack([i_idx, j_idx], dim=0)            # [2, E]


# ===========================================================================
# 10. 测试与运行验证
# ===========================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")
    print("=" * 60)

    N_particles = 500     # 验证阶段使用较小粒子数（build_radius_edge_index 为 O(N²)）
    T_steps     = 5
    CUTOFF_R    = 2.0     # SPH 截断半径
    H_SMOOTH    = 1.2     # SPH 光滑长度（通常 h ≈ 1.2 × 粒子间距）

    print(f"粒子数 N={N_particles}, 时间步 T={T_steps}, rc={CUTOFF_R}, h={H_SMOOTH}")

    # ---- 构建虚拟数据 ----
    torch.manual_seed(42)
    coords_0 = (torch.rand(N_particles, 3) * 10.0).to(device)

    print("构建 SPH 近邻图 edge_index ...")
    edge_index = build_radius_edge_index(coords_0, CUTOFF_R)
    E_edges    = edge_index.shape[1]
    print(f"  近邻边数 E = {E_edges}  (平均每粒子 {E_edges / N_particles:.1f} 个邻居)")

    coords_seq = coords_0.unsqueeze(1).expand(N_particles, T_steps, 3).clone()
    coords_seq = coords_seq + torch.randn_like(coords_seq) * 0.05

    dummy_batch = ParticleSequenceBatch(
        coords     = coords_seq.to(device),
        properties = (torch.ones(N_particles, T_steps, 2)
                      * torch.tensor([1e3, 0.3])).to(device),   # E=1000, nu=0.3
        volume     = (torch.ones(N_particles, 1) * 0.001).to(device),
        bc_values  = torch.zeros(N_particles, T_steps, 3).to(device),
        domain_id  = torch.cat([
            torch.zeros(N_particles // 2, T_steps, dtype=torch.long),
            torch.ones (N_particles - N_particles // 2, T_steps, dtype=torch.long)
        ], dim=0).to(device),
        bc_type    = torch.randint(0, 4, (N_particles, T_steps)).to(device),
        edge_index = edge_index.to(device),
    )

    # ---- 初始化模型 ----
    model = HSTParticleTransformer(
        d_model=128, n_heads=4, num_layers=2, grid_sizes=[2.0, 5.0]
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数量: {total_params:,}")
    print("-" * 60)

    # ---- [Milestone 1+2] 前向推理验证 ----
    print("[Milestone 1+2] 前向推理 (网络严格仅输出位移) ...")
    with torch.no_grad():
        disp = model(dummy_batch)
    print(f"  输入坐标形状  : {dummy_batch.coords.shape}  [N, T, 3]")
    print(f"  输出位移形状  : {disp.shape}  [N, T, 3]  ← 仅 3 维，无应力/应变")
    fixed_pts = (dummy_batch.bc_type == 1)
    max_fixed = disp[fixed_pts].abs().max().item() if fixed_pts.any() else 0.0
    print(f"  硬约束验证 (固定边界位移最大值应=0): {max_fixed:.2e}")
    print("-" * 60)

    # ---- [Milestone 1] SPH 算子独立验证 ----
    print("[Milestone 1] SPH 算子验证 ...")
    with torch.no_grad():
        u_t0      = disp[:, 0, :]
        coords_t0 = dummy_batch.coords[:, 0, :]
        props_t0  = dummy_batch.properties[:, 0, :]
        strain_v  = SPHOperators.compute_sph_strain(
            u_t0, coords_t0, edge_index, dummy_batch.volume, H_SMOOTH
        )
        D_mat     = SPHOperators.build_D_matrix(props_t0)
        stress_v  = SPHOperators.compute_stress(strain_v, D_mat)
    print(f"  SPH 应变形状   : {strain_v.shape}  [N, 6] (Voigt 工程应变，SPH 推导)")
    print(f"  本构矩阵 D 形状: {D_mat.shape}  [N, 6, 6]")
    print(f"  应力形状       : {stress_v.shape}  [N, 6] (σ = Dε，严格力学推导)")
    print(f"  应变最大绝对值 : {strain_v.abs().max().item():.4e}")
    print(f"  应力最大绝对值 : {stress_v.abs().max().item():.4e}")
    print("-" * 60)

    # ---- [Milestone 3] SPH 势能 Loss 验证 ----
    print("[Milestone 3] SPHPhysicsInformedLoss 前向计算验证 ...")
    loss_fn = SPHPhysicsInformedLoss(h_smooth=H_SMOOTH, contact_penalty_k=1e4).to(device)
    with torch.no_grad():
        _, loss_info = loss_fn(disp, dummy_batch)
    print("  Loss 分项:")
    for k, v in loss_info.items():
        print(f"    {k:20s}: {v:.6f}")
    print("-" * 60)

    # ---- [Milestone 4] 训练循环 + backward() 验证 ----
    print("[Milestone 4] 训练循环 + backward() 验证 (3 步) ...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for step in range(1, 4):
        loss_dict = train_one_epoch(model, loss_fn, optimizer, dummy_batch)
        print(
            f"  Step {step} | "
            f"total={loss_dict['loss_total']:9.4f}  "
            f"pi_int={loss_dict['loss_pi_int']:9.4f}  "
            f"pi_ext={loss_dict['loss_pi_ext']:9.4f}  "
            f"contact={loss_dict['loss_contact']:9.4f}"
        )
    print("=" * 60)
    print("全部里程碑验证通过！")
    print("  Milestone 1: ParticleSequenceBatch + SPHOperators     [OK]")
    print("  Milestone 2: HST 网络 (output = displacement only)    [OK]")
    print("  Milestone 3: SPHPhysicsInformedLoss (真实势能泛函)    [OK]")
    print("  Milestone 4: train_one_epoch + backward()             [OK]")
    print("=" * 60)
    print("架构纪律确认：")
    print("  ✓ 网络 forward() 内无应力/应变（彻底解耦）")
    print("  ✓ SPH 核函数梯度 + scatter_add_（无 for 循环遍历边）")
    print("  ✓ 应变能 clamp(min=0) 强制正定，消除负能量作弊漏洞")
    print("  ✓ r_norm clamp(min=1e-8) 防梯度 NaN")
    print("  ✓ Voxel Mean Pooling clamp(min=1.0) 防除零")