import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

# ==========================================
# 1 & 2. 粒子数据结构与 Batch 结构体
# ==========================================

@dataclass
class ParticleSequenceBatch:
    """
    模型输入的 Tensor Batch 结构体。
    基于自回归时序推演设计，包含空间维度 N 和 时间序列维度 T。
    """
    # --- 1. 几何与坐标 ---
    # [N, T, 3] 当前构型的空间坐标 (x, y, z)
    coords: torch.Tensor       
    
    # --- 2. 物理与材料属性 ---
    # [N, T, 2] 材料属性，例如 弹性模量E, 泊松比nu
    properties: torch.Tensor   
    # [N, 1] 粒子的物理体积 dV (用于后续能量积分)
    volume: torch.Tensor       
    
    # --- 3. 边界条件数值 ---
    # [N, T, 3] 外荷载 (Fx, Fy, Fz) 或强制位移数值
    bc_values: torch.Tensor    
    
    # --- 4. 离散几何与边界标记 ---
    # [N, T] 区域编号 (例如区分不同的物体或材料域)
    domain_id: torch.Tensor    
    # [N, T] 边界类型 (0:内部, 1:固定边界, 2:力边界, 3:接触边界)
    bc_type: torch.Tensor      

# ==========================================
# 辅助模块：傅里叶高频坐标映射
# ==========================================
class FourierFeatureMapping(nn.Module):
    """将低维坐标映射为高频特征，消除神经网络的频谱偏置(Spectral Bias)"""
    def __init__(self, input_dim=3, mapping_size=32, scale=10.0):
        super().__init__()
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B) # 冻结参数，随模型转移至 GPU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: [..., 3]
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [..., mapping_size * 2]

# ==========================================
# 3. 粒子混合嵌入模型 (Hybrid Embedding)
# ==========================================
class HybridPhysicsEmbedding(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        # 坐标的高频映射 (3维 -> 64维特征)
        self.fourier_map = FourierFeatureMapping(input_dim=3, mapping_size=32)
        
        # 离散标记查表 (num_embeddings: bc_type 0~4，domain_id 0~9)
        self.embed_bc = nn.Embedding(num_embeddings=5, embedding_dim=16)
        self.embed_domain = nn.Embedding(num_embeddings=10, embedding_dim=16)
        
        # 连续物理量投影 (bc_values: 3 + properties: 2 = 5)
        self.proj_phys = nn.Linear(5, 32)
        
        # 融合网络：64(坐标) + 16(边界) + 16(区域) + 32(物理量) = 128维
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, batch: ParticleSequenceBatch) -> torch.Tensor:
        feat_coords = self.fourier_map(batch.coords)          # [N, T, 64]
        feat_bc = self.embed_bc(batch.bc_type)                # [N, T, 16]
        feat_domain = self.embed_domain(batch.domain_id)      # [N, T, 16]
        
        phys_vars = torch.cat([batch.bc_values, batch.properties], dim=-1)
        feat_phys = self.proj_phys(phys_vars)                 # [N, T, 32]
        
        raw_feat = torch.cat([feat_coords, feat_bc, feat_domain, feat_phys], dim=-1)
        h_emb = self.fusion_mlp(raw_feat)                     # [N, T, d_model]
        return h_emb

# ==========================================
# 4. 核心算子：基于网格的层级聚类注意力 
#    (Hierarchical Voxel Attention)
# ==========================================
class HierarchicalVoxelAttention(nn.Module):
    def __init__(self, d_model: int, grid_sizes: List[float], n_heads: int = 4):
        """
        grid_sizes: 定义多重网格的尺度，例如 [1.0, 5.0] 代表细网格和粗网格。
        """
        super().__init__()
        self.d_model = d_model
        self.grid_sizes = grid_sizes
        
        # 为每个宏观网格层级，配备一个 Transformer 用于算超级粒子之间的全局注意力
        self.macro_attentions = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True
            ) for _ in grid_sizes
        ])
        
        # 多尺度特征融合 (原始特征 + N个层级的广播特征)
        self.multi_scale_fusion = nn.Linear(d_model * (len(grid_sizes) + 1), d_model)

    def forward(self, h: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        h: 当前时间步的粒子特征 [N, d_model]
        coords: 当前时间步的物理坐标 [N, 3]
        """
        N = h.shape[0]
        device = h.device
        multi_scale_features = [h] # 保存原始尺度
        
        for i, grid_size in enumerate(self.grid_sizes):
            # ----------------------------------------------------
            # Step 1: 粗糙体素化划分 (Voxelization)
            # ----------------------------------------------------
            # 将物理坐标划归到离散的 3D 整数网格上
            voxel_indices = torch.floor(coords / grid_size).long() # [N, 3]
            
            # 寻找当前空间存在的独立网格 (宏观粒子)，并获取原始粒子的反向映射
            # unique_voxels: 独一无二的网格坐标 [M, 3], M 是宏观粒子数
            # inverse_mapping: 长度为 N 的一维张量，指明每个底层粒子属于哪个宏观粒子 (0 到 M-1)
            unique_voxels, inverse_mapping = torch.unique(
                voxel_indices, dim=0, return_inverse=True
            )
            M_macro = unique_voxels.shape[0]
            
            # ----------------------------------------------------
            # Step 2: Scatter 聚合综合宏观特征 (Mean Pooling)
            # ----------------------------------------------------
            # 准备接受宏观特征的张量 [M, d_model]
            macro_h = torch.zeros(M_macro, self.d_model, device=device)
            
            # 将 inverse_mapping 扩展到与特征 h 同维度，用于 scatter
            idx_expanded = inverse_mapping.unsqueeze(1).expand_as(h) # [N, d_model]
            
            # 把同一个格子内的真实粒子特征累加到对应的宏观粒子上
            macro_h.scatter_add_(dim=0, index=idx_expanded, src=h)
            
            # 计算每个格子里包含的真实粒子数量，用于求平均
            counts = torch.bincount(inverse_mapping, minlength=M_macro).unsqueeze(1).float()
            # Mean Pooling: 避免除零
            macro_h = macro_h / torch.clamp(counts, min=1.0) 
            
            # ----------------------------------------------------
            # Step 3: 宏观超级粒子的全局交互 (Macro Global Attention)
            # ----------------------------------------------------
            # 增加 Batch 维度以符合 Transformer 要求: [1, M, d_model]
            macro_h = macro_h.unsqueeze(0)
            macro_h = self.macro_attentions[i](macro_h)
            macro_h = macro_h.squeeze(0) # [M, d_model]
            
            # ----------------------------------------------------
            # Step 4: 宏观边界广播与特征下放 (Broadcasting)
            # ----------------------------------------------------
            # 根据反向映射，直接将全局融合后的特征广播回底层的 N 个真实粒子
            broadcasted_h = macro_h[inverse_mapping] # [N, d_model]
            
            multi_scale_features.append(broadcasted_h)
            
        # ----------------------------------------------------
        # Step 5: 融合所有尺度的物理信息
        # ----------------------------------------------------
        # 拼接形状: [N, d_model * (1 + len(grid_sizes))]
        concat_h = torch.cat(multi_scale_features, dim=-1)
        final_h = self.multi_scale_fusion(concat_h) # [N, d_model]
        
        return final_h

# ==========================================
# 5. 完整时空架构组合 (HST-PT)
# ==========================================
class HSTParticleTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, num_layers=3, grid_sizes=[1.0, 5.0]):
        super().__init__()
        self.embedding = HybridPhysicsEmbedding(d_model)
        
        # 堆叠时空处理块
        self.num_layers = num_layers
        self.temporal_attentions = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.spatial_attentions = nn.ModuleList([
            HierarchicalVoxelAttention(d_model, grid_sizes, n_heads)
            for _ in range(num_layers)
        ])
        
        # 输出解码器：降维到物理状态
        # 假设输出 15 维 = 3(位移) + 6(应力) + 6(应变)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 15)
        )

    def forward(self, batch: ParticleSequenceBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. 输入嵌入: [N, T, d_model]
        h = self.embedding(batch)
        N, T, d_model = h.shape
        
        for i in range(self.num_layers):
            # --- 时序注意力 (Temporal) ---
            # 粒子之间隔离，只看自己的过去和未来。复杂度 O(N * T^2)
            h = self.temporal_attentions[i](h) 
            
            # --- 空间层级注意力 (Spatial) ---
            # 由于每一个时间步的物理坐标可能不同（大变形），我们按时间步遍历执行空间融合
            # 复杂度 O(T * N)，具有绝佳的拓展性
            h_spatial_updated = torch.zeros_like(h)
            for t in range(T):
                # 提取当前时刻的所有粒子特征和坐标
                h_t = h[:, t, :]               # [N, d_model]
                coords_t = batch.coords[:, t, :] # [N, 3]
                
                # 执行基于网格的层级聚类和广播
                h_spatial_updated[:, t, :] = self.spatial_attentions[i](h_t, coords_t)
            
            h = h_spatial_updated
            
        # 2. 逐点解码物理场: [N, T, 15]
        out_state = self.decoder(h)                          # shape: [N, T, 15]
        
        # 3. 截断切片获取具体的物理量
        displacement = out_state[..., 0:3]   # shape: [N, T, 3]
        stress       = out_state[..., 3:9]   # shape: [N, T, 6]
        strain       = out_state[..., 9:15]  # shape: [N, T, 6]
        
        # 模块 C 硬约束：bc_type == 1（固定位移边界）的粒子位移强制归零
        fixed_mask = (batch.bc_type == 1).unsqueeze(-1)      # shape: [N, T, 1]
        displacement = displacement * (~fixed_mask).float()  # shape: [N, T, 3]
        
        return displacement, stress, strain

# ==========================================
# 6. 物理驱动损失函数 (Physics-Informed Loss)
#    模块 D: 深度能量法 (Deep Energy Method)
# ==========================================
class PhysicsInformedLoss(nn.Module):
    """
    无监督物理约束损失，基于最小势能原理 (Principle of Minimum Potential Energy)。
    总势能：Π_total = Π_int + Π_ext + L_contact
    """
    def __init__(self, contact_penalty_k: float = 1e4):
        super().__init__()
        # Voigt 符号下双缩并权重向量: σ:ε = Σ w_k * σ_k * ε_k
        # 分量顺序: [σ_11, σ_22, σ_33, σ_12, σ_23, σ_31]
        self.register_buffer('voigt_weights', torch.tensor([1., 1., 1., 2., 2., 2.]))
        self.k_penalty = contact_penalty_k  # 接触罚刚度系数

    def forward(
        self,
        displacement: torch.Tensor,      # [N, T, 3]
        stress:       torch.Tensor,      # [N, T, 6]
        strain:       torch.Tensor,      # [N, T, 6]
        batch:        ParticleSequenceBatch
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        N, T, _ = displacement.shape
        device   = displacement.device

        # ============================================================
        # 损失 1: 应变能密度积分 (Internal Strain Energy)
        # Π_int = 0.5 * Σ_i (σ_i : ε_i) * dV_i，在 T 维度取均值
        # ============================================================
        # Voigt 双缩并: (σ : ε) 逐分量相乘加权求和       shape: [N, T]
        double_contraction = (stress * strain * self.voigt_weights).sum(dim=-1)
        # volume: [N, 1] 广播至 [N, T]，对空间 N 求和后对时间 T 取均值
        pi_int = 0.5 * (double_contraction * batch.volume).sum(dim=0).mean()  # scalar

        # ============================================================
        # 损失 2: 外力功 (External Work)
        # Π_ext = -Σ_i F_i · u_i * [bc_type == 2]（仅力边界激活）
        # ============================================================
        force_mask = (batch.bc_type == 2).unsqueeze(-1).float()              # shape: [N, T, 1]
        ext_work   = (batch.bc_values * displacement * force_mask).sum(dim=-1)  # shape: [N, T]
        pi_ext     = -(ext_work * batch.volume).sum(dim=0).mean()            # scalar

        # ============================================================
        # 损失 3: 接触穿透罚函数 (Contact Penetration Penalty)
        # 针对 bc_type == 3 且 domain_id 不同的粒子对，计算穿透量 g
        # L_contact = k * Σ max(0, -g)^2，在时间步上取均值
        # ============================================================
        contact_loss_per_step: List[torch.Tensor] = []

        for t in range(T):
            contact_mask = (batch.bc_type[:, t] == 3)             # shape: [Nc] bool
            Nc = contact_mask.sum().item()
            if Nc < 2:
                continue

            # 当前时刻接触粒子的变形后坐标: x + u           shape: [Nc, 3]
            deformed_coords = (
                batch.coords[:, t, :][contact_mask]
                + displacement[:, t, :][contact_mask]
            )
            contact_domains = batch.domain_id[:, t][contact_mask]  # shape: [Nc]

            # 粒子间距离矩阵                                    shape: [Nc, Nc]
            diff = deformed_coords.unsqueeze(0) - deformed_coords.unsqueeze(1)  # [Nc, Nc, 3]
            dist = torch.norm(diff, dim=-1)                                     # shape: [Nc, Nc]

            # 只对不同 domain_id 的粒子对施加惩罚             shape: [Nc, Nc]
            diff_domain = (contact_domains.unsqueeze(0) != contact_domains.unsqueeze(1)).float()

            # 粒子等效半径（由体积估算）: r ≈ V^(1/3) / 2     shape: [Nc]
            radius   = (batch.volume[contact_mask].squeeze(-1) ** (1.0 / 3.0)) / 2.0
            min_dist = radius.unsqueeze(0) + radius.unsqueeze(1)                # shape: [Nc, Nc]

            # 穿透量 > 0 即发生穿透                           shape: [Nc, Nc]
            penetration = torch.clamp(min_dist - dist, min=0.0)

            step_loss = (self.k_penalty * penetration ** 2 * diff_domain).sum()
            contact_loss_per_step.append(step_loss)

        loss_contact = (
            torch.stack(contact_loss_per_step).mean()
            if contact_loss_per_step
            else torch.zeros(1, device=device).squeeze()
        )

        # ============================================================
        # 汇总总势能
        # ============================================================
        total_loss = pi_int + pi_ext + loss_contact
        loss_dict: Dict[str, float] = {
            "loss_total":   total_loss.item(),
            "loss_pi_int":  pi_int.item(),
            "loss_pi_ext":  pi_ext.item(),
            "loss_contact": loss_contact.item(),
        }
        return total_loss, loss_dict


# ==========================================
# 7. 自回归训练主循环 (Training Loop)
# ==========================================
def train_one_epoch(
    model:      HSTParticleTransformer,
    loss_fn:    PhysicsInformedLoss,
    optimizer:  torch.optim.Optimizer,
    batch:      ParticleSequenceBatch,
    grad_clip:  float = 1.0
) -> Dict[str, float]:
    """
    单轮训练步骤：前向 -> 物理损失 -> 反向传播 -> 梯度裁剪 -> 参数更新。
    返回包含各项损失值的字典（已 detach，不保留计算图）。
    """
    model.train()
    optimizer.zero_grad()

    # 前向推演：输出三大物理场
    displacement, stress, strain = model(batch)    # [N,T,3], [N,T,6], [N,T,6]

    # 计算无监督物理约束损失
    total_loss, loss_dict = loss_fn(displacement, stress, strain, batch)

    # 反向传播
    total_loss.backward()

    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # 参数更新
    optimizer.step()

    return loss_dict


# ==========================================
# 8. 测试与运行验证
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")
    
    # 模拟 10,000 个粒子，推演 5 个时间步
    N_particles = 10000 
    T_steps = 5
    
    # 构建虚拟数据集 (归一化到一个 10x10x10 的立方体空间)
    dummy_batch = ParticleSequenceBatch(
        coords = (torch.rand(N_particles, T_steps, 3) * 10.0).to(device),
        properties = torch.ones(N_particles, T_steps, 2).to(device),
        volume = (torch.ones(N_particles, 1) * 0.001).to(device),
        bc_values = torch.zeros(N_particles, T_steps, 3).to(device),
        domain_id = torch.zeros(N_particles, T_steps, dtype=torch.long).to(device),
        bc_type = torch.randint(0, 4, (N_particles, T_steps)).to(device)
    )
    
    # 初始化模型
    # grid_sizes=[2.0, 5.0] 意味着在 10x10x10 的空间里：
    # 第一层级将切出 5x5x5=125 个宏观粒子，负责中尺度边界传导
    # 第二层级将切出 2x2x2=8 个超宏观粒子，负责全域左边界到右边界的瞬间耦合
    model = HSTParticleTransformer(
        d_model=128, 
        n_heads=4, 
        num_layers=2, 
        grid_sizes=[2.0, 5.0]
    ).to(device)
    
    print("模型构建完成，开始前向推理测试...")

    # ---- 推理验证 (无梯度) ----
    with torch.no_grad():
        disp, stress, strain = model(dummy_batch)

    print(f"输入张量坐标形状: {dummy_batch.coords.shape}  => [粒子数N, 时间步T, 维度D]")
    print("-" * 50)
    print(f"成功输出 预测位移 张量: {disp.shape}   (bc_type==1 处已硬约束归零)")
    print(f"成功输出 预测应力 张量: {stress.shape}")
    print(f"成功输出 预测应变 张量: {strain.shape}")
    print("-" * 50)

    # ---- 硬约束验证 ----
    fixed_pts = (dummy_batch.bc_type == 1)
    max_disp_at_fixed = disp[fixed_pts].abs().max().item()
    print(f"硬约束验证 (固定边界位移最大值应≈0): {max_disp_at_fixed:.6f}")
    print("-" * 50)

    # ---- 物理损失函数验证 ----
    print("初始化物理驱动损失函数 (Deep Energy Method)...")
    loss_fn = PhysicsInformedLoss(contact_penalty_k=1e4).to(device)

    with torch.no_grad():
        _, loss_info = loss_fn(disp, stress, strain, dummy_batch)

    print("损失函数前向计算通过:")
    for k, v in loss_info.items():
        print(f"  {k:20s}: {v:.6f}")
    print("-" * 50)

    # ---- 训练循环验证 (仅跑 3 步以验证 backward 可用性) ----
    print("开始训练循环验证 (3 步)...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for step in range(1, 4):
        loss_dict = train_one_epoch(model, loss_fn, optimizer, dummy_batch)
        print(
            f"  Step {step} | "
            f"total={loss_dict['loss_total']:.4f}  "
            f"pi_int={loss_dict['loss_pi_int']:.4f}  "
            f"pi_ext={loss_dict['loss_pi_ext']:.4f}  "
            f"contact={loss_dict['loss_contact']:.4f}"
        )
    print("-" * 50)
    print("全部里程碑验证通过！")
    print("  Milestone 1: Data & Embedding Foundation  [OK]")
    print("  Milestone 2: Hierarchical Voxel Attention [OK]")
    print("  Milestone 3: Decoder + Energy Loss        [OK]")
    print("  Milestone 4: Training Loop + backward()   [OK]")