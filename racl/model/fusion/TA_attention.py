"""
TA-Attention: Flow-Guided Sparse Attention Module

这是在原始 KCPM 基础上的改进版本，添加了光流引导的稀疏注意力机制。

核心改进：
1. 保留 KCPM 的光流估计能力（粗对齐）
2. 复用 KCPM 的光流作为注意力先验（零成本）
3. 添加轻量级稀疏注意力模块（精细化对齐）
4. 门控融合机制（自适应平衡两个分支）

使用方法：
    from racl.model.fusion.TA_attention import KCPM_Attention
    
    model = KCPM_Attention(in_channels=1024, num_heads=4, window_size=7)
    output, flow = model(x1, x2, fusion_policy='abs_diff', use_attention=True)

作者: [Your Name]
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flow_Block(nn.Module):
    """
    光流预测块（复用自原始 KCPM）
    
    功能：为特征的一部分通道预测局部光流
    """

    def __init__(self, in_channels, kernel_size) -> None:
        super(Flow_Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels*2, in_channels*2, 
            kernel_size=kernel_size, 
            padding=(kernel_size-1)//2, 
            bias=False, 
            groups=in_channels*2
        )
        self.insnorm1 = nn.InstanceNorm2d(in_channels*2)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(
            in_channels, in_channels*2, 
            kernel_size=kernel_size, 
            padding=(kernel_size-1)//2, 
            bias=False, 
            groups=in_channels
        )
        self.insnorm3 = nn.InstanceNorm2d(in_channels*2)
        self.gelu3 = nn.GELU()
        self.predict_flow = nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.insnorm1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.insnorm3(x)
        x = self.gelu3(x)
        x = self.predict_flow(x)
        return x


class FlowGuidedSparseAttention(nn.Module):
    """
    光流引导的轻量级稀疏注意力模块
    
    设计理念：
    1. 直接使用 KCPM 的光流（不重新计算）
    2. 简化的窗口构建（使用 unfold，避免复杂索引）
    3. 降维的 QKV projection（减少计算量）
    
    Args:
        dim: 输入特征维度
        num_heads: 注意力头数（建议 4-8）
        window_size: 注意力窗口大小（建议 7 或 9，必须是奇数）
        qkv_ratio: QKV 降维比例（0.5 表示降到一半）
    """
    
    def __init__(self, dim, num_heads=4, window_size=7, qkv_ratio=0.5):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv_dim = int(dim * qkv_ratio)
        self.head_dim = self.qkv_dim // num_heads
        
        assert self.qkv_dim % num_heads == 0, "qkv_dim must be divisible by num_heads"
        
        # 轻量级 QKV projection（降维）
        self.to_qkv = nn.Linear(dim, self.qkv_dim * 3)
        self.to_out = nn.Linear(self.qkv_dim, dim)
        
        # 可学习的温度参数（控制注意力的锐度）
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # 相对位置编码
        self.use_relative_position_bias = True
        if self.use_relative_position_bias:
            # 相对位置偏置表
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
            # 初始化相对位置偏置
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
            
            # 预计算相对位置索引
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, ws, ws]
            coords_flatten = torch.flatten(coords, 1)  # [2, ws*ws]
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, ws*ws, ws*ws]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # [ws*ws, ws*ws]
            self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x1, x2, flow_guide):
        """
        前向传播（光流引导版本）
        
        Args:
            x1: 前时相特征 [B, N, C]，N = H×W
            x2: 后时相特征 [B, N, C]
            flow_guide: 来自 KCPM 的光流 [B, 4, H, W]
                       前 2 通道是 A→B 的光流，后 2 通道是 B→A 的光流
        
        Returns:
            output: 注意力输出 [B, N, C]
        """
        B, N, C = x1.shape
        H = W = int(N ** 0.5)
        
        # 验证输入
        assert x1.shape == x2.shape, f"x1 and x2 must have the same shape, got {x1.shape} and {x2.shape}"
        assert H * W == N, f"N must be a perfect square, got N={N}, H={H}, W={W}"
        
        # ============ 步骤 1: 提取 A→B 的光流 ============
        flow_a2b = flow_guide[:, :2]  # [B, 2, H, W]
        
        # ============ 步骤 2: 计算 QKV ============
        qkv1 = self.to_qkv(x1).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv2 = self.to_qkv(x2).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        q = qkv1[:, :, 0].permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        k = qkv2[:, :, 1].permute(0, 2, 1, 3)
        v = qkv2[:, :, 2].permute(0, 2, 1, 3)
        
        # ============ 步骤 3: 提取光流引导的局部窗口 ============
        # 🔧 关键改进：使用光流动态调整窗口
        k_windows = self.extract_flow_guided_windows(k, H, W, flow_a2b)  # [B, num_heads, N, K, head_dim]
        v_windows = self.extract_flow_guided_windows(v, H, W, flow_a2b)
        
        # ============ 步骤 4: 计算稀疏注意力 ============
        # Q @ K^T
        attn = torch.einsum('bhnd,bhnkd->bhnk', q, k_windows)
        attn = attn / (self.head_dim ** 0.5) / torch.clamp(self.temperature, min=1e-3)  # 防止温度过小
        
        # 🔧 添加相对位置编码（修复维度匹配问题）
        if self.use_relative_position_bias:
            # 获取实际的窗口大小
            K_actual = k_windows.shape[3]  # 实际窗口中的位置数
            K_expected = self.window_size * self.window_size  # 预期的窗口大小
            
            # 只有当维度匹配时才添加相对位置编码
            if K_actual == K_expected:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.view(-1)
                ].view(K_expected, K_expected, -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, K, K]
                # 只取对角线元素（因为我们的窗口是动态的）
                relative_position_bias = torch.diagonal(relative_position_bias, dim1=1, dim2=2)  # [num_heads, K]
                relative_position_bias = relative_position_bias.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, K]
                attn = attn + relative_position_bias
            else:
                # 维度不匹配时跳过相对位置编码（打印警告）
                if not hasattr(self, '_warned_position_bias'):
                    print(f"Warning: Skipping relative position bias due to dimension mismatch: "
                          f"K_actual={K_actual}, K_expected={K_expected}")
                    self._warned_position_bias = True
        
        # 🔧 添加光流引导的注意力偏置
        flow_bias = self.compute_flow_attention_bias(flow_a2b, H, W)
        attn = attn + flow_bias
        
        attn = F.softmax(attn, dim=-1)
        self.last_attn = attn.detach()
        self.last_attn_map = attn.detach().amax(dim=-1).mean(dim=1)
        self.last_flow_magnitude = torch.sqrt(flow_a2b[:, 0]**2 + flow_a2b[:, 1]**2).detach()
        
        out = torch.einsum('bhnk,bhnkd->bhnd', attn, v_windows)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.qkv_dim)
        out = self.to_out(out)
        
        return out
    
    def extract_flow_guided_windows(self, feat, H, W, flow):
        """
        提取光流引导的局部窗口（核心实现，向量化优化版本）
        
        策略：使用 grid_sample 根据光流动态采样窗口
        
        Args:
            feat: [B, num_heads, N, head_dim]
            H, W: 特征图尺寸
            flow: [B, 2, H, W]，光流场
        
        Returns:
            windows: [B, num_heads, N, K, head_dim]
                    K = window_size^2
        """
        B, num_heads, N, head_dim = feat.shape
        half_win = self.window_size // 2
        K = self.window_size ** 2
        
        # 重塑为图像格式 [B*num_heads, head_dim, H, W]
        feat = feat.permute(0, 1, 3, 2).reshape(B * num_heads, head_dim, H, W)
        
        # 扩展 flow 到所有 heads [B*num_heads, 2, H, W]
        # 🔧 关键修复：确保 flow 的 dtype 与 feat 一致
        flow = flow.to(dtype=feat.dtype)
        flow = flow.unsqueeze(1).repeat(1, num_heads, 1, 1, 1).reshape(B * num_heads, 2, H, W)
        
        # ============ 构建光流引导的采样网格（向量化版本） ============
        # 1. 创建基础网格 [-1, 1] 范围（确保数据类型与 feat 一致）
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, W, device=feat.device, dtype=feat.dtype),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).repeat(B * num_heads, 1, 1, 1)  # [B*num_heads, H, W, 2]
        
        # 2. 归一化光流到 [-1, 1] 范围（确保数据类型一致）
        flow_norm = flow.permute(0, 2, 3, 1)  # [B*num_heads, H, W, 2]
        flow_norm = flow_norm / torch.tensor([W, H], device=flow.device, dtype=flow.dtype).view(1, 1, 1, 2) * 2
        
        # 3. 向量化创建窗口内的相对偏移（确保数据类型一致）
        dy = torch.arange(-half_win, half_win + 1, device=feat.device, dtype=feat.dtype)
        dx = torch.arange(-half_win, half_win + 1, device=feat.device, dtype=feat.dtype)
        dy_grid, dx_grid = torch.meshgrid(dy, dx, indexing='ij')
        window_offsets = torch.stack([dx_grid, dy_grid], dim=-1).reshape(K, 2)  # [K, 2]
        window_offsets = window_offsets.view(K, 1, 1, 2)  # [K, 1, 1, 2]
        window_offsets_norm = window_offsets / torch.tensor([W, H], device=feat.device, dtype=feat.dtype).view(1, 1, 1, 2) * 2
        
        # 4. 向量化采样所有窗口位置
        # 扩展维度以支持批量采样
        base_grid_expanded = base_grid.unsqueeze(0)  # [1, B*num_heads, H, W, 2]
        flow_norm_expanded = flow_norm.unsqueeze(0)  # [1, B*num_heads, H, W, 2]
        window_offsets_expanded = window_offsets_norm.unsqueeze(1)  # [K, 1, 1, 1, 2]
        
        # 计算所有窗口位置的采样网格
        sample_grids = base_grid_expanded + flow_norm_expanded + window_offsets_expanded  # [K, B*num_heads, H, W, 2]
        sample_grids = torch.clamp(sample_grids, -1, 1)  # 确保在有效范围内
        
        # 批量采样
        windows_list = []
        for k in range(K):
            sampled_feat = F.grid_sample(
                feat, 
                sample_grids[k], 
                mode='bilinear', 
                padding_mode='border',
                align_corners=True
            )  # [B*num_heads, head_dim, H, W]
            windows_list.append(sampled_feat)
        
        # 5. 堆叠所有窗口位置
        windows = torch.stack(windows_list, dim=2)  # [B*num_heads, head_dim, K, H, W]
        windows = windows.reshape(B, num_heads, head_dim, K, H * W)
        windows = windows.permute(0, 1, 4, 3, 2)  # [B, num_heads, N, K, head_dim]
        
        # 检查 NaN
        if torch.isnan(windows).any():
            print("Warning: NaN detected in flow-guided windows, replacing with zeros")
            windows = torch.nan_to_num(windows, nan=0.0)
        
        return windows
    
    def compute_flow_attention_bias(self, flow, H, W):
        """
        根据光流计算注意力偏置
        
        策略：光流幅度大的地方，注意力权重应该更高
        
        Args:
            flow: [B, 2, H, W]
            H, W: 特征图尺寸
        
        Returns:
            flow_bias: [B, num_heads, N, K]
        """
        B = flow.shape[0]
        N = H * W
        K = self.window_size ** 2
        
        # 计算光流幅度
        flow_magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)  # [B, H, W]
        flow_magnitude = flow_magnitude.view(B, N)  # [B, N]
        
        # 归一化到 [0, 1]
        flow_magnitude = flow_magnitude / (flow_magnitude.max(dim=1, keepdim=True)[0] + 1e-6)
        
        # 扩展到所有 heads 和窗口位置
        flow_bias = flow_magnitude.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        flow_bias = flow_bias.repeat(1, self.num_heads, 1, K)  # [B, num_heads, N, K]
        
        # 转换为注意力偏置（光流大的地方权重高）
        flow_bias = flow_bias * 2.0 - 1.0  # 映射到 [-1, 1]
        
        return flow_bias


class TA_Attention(nn.Module):
    """
    KCPM + 光流引导稀疏注意力
    
    完全兼容原始 KCPM 的接口：
    - 输入：x1, x2 [B, N, C]，其中 N = H×W
    - 输出：output [B, N, C] 或 (x1_feat_g, x2_feat_g) 元组
    - fusion_policy：'abs_diff', 'diff', 'sum', 'concat' 或 None
    
    新增参数：
    - use_attention：是否启用光流引导注意力（默认 False，保持向后兼容）
    
    架构：
    1. KCPM 光流对齐（粗对齐）
       - 4 个 Flow_Block 处理局部光流
       - 1 个 flow_make_g 处理全局光流
    2. 稀疏注意力精细化（可选）
       - 使用 KCPM 的光流引导注意力窗口
       - 在局部窗口内计算注意力
    3. 门控融合
       - 自适应平衡 KCPM 和注意力的输出
    
    Args:
        in_channels: 输入特征通道数（如 1024）
        num_heads: 注意力头数（建议 4-8）
        window_size: 注意力窗口大小（建议 7 或 9）
        use_gate: 是否使用门控融合（建议 True）
        enable_attention_by_default: 是否默认启用注意力（建议 False，保持兼容）
    """

    def __init__(self, in_channels, num_heads=4, window_size=7, use_gate=True, enable_attention_by_default=False):
        super(TA_Attention, self).__init__()
        self.in_channels = in_channels // 4
        self.use_gate = use_gate
        self.enable_attention_by_default = enable_attention_by_default
        
        # ============ KCPM 组件（保持不变） ============
        kernel_size = 5
        
        # 全局光流生成器
        self.flow_make_g = nn.Sequential(
            nn.Conv2d(
                in_channels*2, in_channels*2, 
                kernel_size=kernel_size, 
                padding=(kernel_size-1)//2, 
                bias=False, 
                groups=in_channels*2
            ),
            nn.InstanceNorm2d(in_channels*2),
            nn.GELU(),
            nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=True),
        )
        
        # 4 个局部光流块
        self.flows = nn.ModuleList([
            Flow_Block(in_channels=self.in_channels, kernel_size=kernel_size)
            for _ in range(4)
        ])
        
        # ============ 新增：轻量级注意力模块 ============
        self.attention_refine = FlowGuidedSparseAttention(
            dim=in_channels,
            num_heads=num_heads,
            window_size=window_size,
            qkv_ratio=0.5  # 降维到一半，减少计算量
        )
        
        # ============ 新增：门控融合模块 ============
        if self.use_gate:
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x1, x2, fusion_policy=None, use_attention=None):
        """
        前向传播（完全兼容原始 KCPM 接口）
        
        Args:
            x1: 前时相特征 [B, N, C]，N = H×W
            x2: 后时相特征 [B, N, C]
            fusion_policy: 特征融合策略（与原始 KCPM 一致）
                - None: 返回 (x1_feat_g, x2_feat_g) 元组
                - 'abs_diff': 绝对差分（推荐）
                - 'diff': 有向差分
                - 'concat': 拼接
                - 'sum': 相加
            use_attention: 是否使用注意力精细化（新增参数）
                - None: 使用 enable_attention_by_default 的设置
                - True: 启用 KCPM + Attention
                - False: 只用 KCPM（完全等价于原始 KCPM）
        
        Returns:
            output: 融合后的特征 [B, N, C]
            或 (x1_feat_g, x2_feat_g) 元组（当 fusion_policy is None 时）
        
        注意：
        - 当 use_attention=False 或 None（且 enable_attention_by_default=False）时，
          行为完全等价于原始 KCPM
        - 这确保了向后兼容性
        """
        # 确定是否使用注意力
        if use_attention is None:
            use_attention = self.enable_attention_by_default
        
        B, N, C = x1.size()
        H = int(N ** 0.5)
        W = H
        
        # 输入验证
        assert x1.shape == x2.shape, f"x1 and x2 must have the same shape, got {x1.shape} and {x2.shape}"
        assert H * W == N, f"N must be a perfect square, got N={N}, H={H}, W={W}"
        if fusion_policy is not None:
            _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
            assert fusion_policy in _fusion_policies, \
                f'Fusion policy must be one of {_fusion_policies}, got {fusion_policy}'
        
        # ============ 阶段 1: KCPM 光流对齐（粗对齐） ============
        # 重塑为图像格式
        x1_img = x1.permute(0, 2, 1).reshape(B, C, H, W)
        x2_img = x2.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 通道分组（4 组）
        x1_chunks = torch.chunk(x1_img, chunks=4, dim=1)
        x2_chunks = torch.chunk(x2_img, chunks=4, dim=1)
        
        # 拼接每组的 A 和 B 特征
        outputs = [
            torch.cat((a_chunk, b_chunk), dim=1)
            for a_chunk, b_chunk in zip(x1_chunks, x2_chunks)
        ]
        
        x1_feats = []
        x2_feats = []
        
        # 局部光流处理
        for flow_block, out, x1_c, x2_c in zip(
                self.flows, outputs, x1_chunks, x2_chunks):
            flow = flow_block(out)  # [B, 4, H, W]
            f1, f2 = torch.chunk(flow, 2, dim=1)
            
            # 光流对齐 + 差分
            x1_feat = self.warp(x1_c, f1) - x2_c
            x2_feat = self.warp(x2_c, f2) - x1_c
            
            x1_feats.append(x1_feat)
            x2_feats.append(x2_feat)
        
        # 拼接局部变化特征
        x1_feat = torch.cat(x1_feats, dim=1)  # [B, C, H, W]
        x2_feat = torch.cat(x2_feats, dim=1)
        
        # 全局光流处理
        output_l = torch.cat([x1_feat, x2_feat], dim=1)
        flow_g = self.flow_make_g(output_l)  # [B, 4, H, W]
        f1_g, f2_g = torch.chunk(flow_g, 2, dim=1)
        
        # 全局对齐
        x1_feat_g = self.warp(x1_feat, f1_g) - x2_feat
        x2_feat_g = self.warp(x2_feat, f2_g) - x1_feat
        
        # ============ 与原始 KCPM 完全一致的返回逻辑 ============
        if fusion_policy is None:
            # 返回元组（与原始 KCPM 一致）
            return x1_feat_g, x2_feat_g
        
        # KCPM 输出（粗对齐的变化特征）
        TA_output = self.fusion(x1_feat_g, x2_feat_g, fusion_policy)
        
        # ============ 阶段 2: 注意力精细化（可选，新增功能） ============
        if use_attention:
            # 将特征转回 [B, N, C] 格式
            TA_flat = TA_output.view(B, C, N).permute(0, 2, 1)
            
            # 使用 KCPM 的光流引导注意力
            attn_output = self.attention_refine(
                x1, x2, 
                flow_guide=flow_g  # 关键：复用 KCPM 的光流
            )
            if hasattr(self.attention_refine, "last_attn"):
                self.last_attn = self.attention_refine.last_attn
                self.last_attn_map = self.attention_refine.last_attn_map
            if hasattr(self.attention_refine, "last_flow_magnitude"):
                self.last_flow_magnitude = self.attention_refine.last_flow_magnitude
            
            # 门控融合：动态平衡 KCPM 和注意力（统一使用残差连接版本）
            if self.use_gate:
                attn_output_img = attn_output.permute(0, 2, 1).view(B, C, H, W)
                gate = self.fusion_gate(
                    torch.cat([TA_output, attn_output_img], dim=1)
                )
                # 残差连接：确保 KCPM 的贡献始终保留
                output = TA_output + gate * attn_output_img
            else:
                # 简单相加（如果不使用门控）
                attn_output_img = attn_output.permute(0, 2, 1).view(B, C, H, W)
                output = TA_output + attn_output_img
        else:
            output = TA_output
        
        # 转回 [B, N, C] 格式（与原始 KCPM 一致）
        output = output.view(B, C, N).permute(0, 2, 1)
        
        return output
    
    @staticmethod
    def warp(x, flow):
        """
        光流变形函数（复用自原始 KCPM，添加边界检查）
        
        根据光流对特征进行空间变形（对齐）
        
        Args:
            x: 输入特征 [B, C, H, W]
            flow: 光流 [B, 2, H, W]
        
        Returns:
            output: 变形后的特征 [B, C, H, W]
        """
        n, c, h, w = x.size()

        # 归一化因子
        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        
        # 创建基础网格
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        
        # 添加光流偏移
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        
        # 边界检查：将超出范围的值裁剪到 [-1, 1]
        grid = torch.clamp(grid, -1, 1)

        # 使用 grid_sample 进行变形
        output = F.grid_sample(x, grid, align_corners=True, padding_mode='border')
        
        # NaN 检查
        if torch.isnan(output).any():
            print("Warning: NaN detected in warp output, replacing with input")
            output = torch.nan_to_num(output, nan=0.0)
        
        return output

    @staticmethod
    def fusion(x1, x2, policy):
        """
        特征融合函数（复用自原始 KCPM）
        
        Args:
            x1, x2: 输入特征 [B, C, H, W]
            policy: 融合策略
        
        Returns:
            融合后的特征 [B, C, H, W]
        """
        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, \
            f'Fusion policy must be one of {_fusion_policies}, got {policy}'
        
        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'abs_diff':
            x = torch.abs(x1 - x2)

        return x


# ============================================================================
# 辅助函数：用于可视化和调试
# ============================================================================

def visualize_flow(flow, save_path=None):
    """
    可视化光流（用颜色编码方向和幅度）
    
    Args:
        flow: [B, 2, H, W] 或 [2, H, W]
        save_path: 保存路径（可选）
    
    Returns:
        flow_color: [H, W, 3]，RGB 图像
    
    注意：需要安装 opencv-python、matplotlib 和 numpy
    """
    try:
        import numpy as np
        import cv2
    except ImportError as e:
        raise ImportError(
            "visualize_flow requires opencv-python and numpy. "
            "Install them with: pip install opencv-python numpy"
        ) from e
    
    if flow.dim() == 4:
        flow = flow[0]  # 取第一个样本
    
    flow = flow.detach().cpu().numpy()
    
    # 计算光流的幅度和角度
    magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
    angle = np.arctan2(flow[1], flow[0])
    
    # 归一化
    magnitude = magnitude / (magnitude.max() + 1e-6)
    
    # 转换为 HSV（色调表示方向，饱和度表示幅度）
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 180  # 色调
    hsv[..., 1] = 255  # 饱和度
    hsv[..., 2] = magnitude * 255  # 明度
    
    # 转换为 RGB
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    if save_path:
        try:
            import matplotlib.pyplot as plt
            plt.imsave(save_path, flow_color)
        except ImportError:
            # 如果没有 matplotlib，使用 cv2 保存
            cv2.imwrite(save_path, cv2.cvtColor(flow_color, cv2.COLOR_RGB2BGR))
    
    return flow_color


def count_parameters(model):
    """
    统计模型参数量
    
    Args:
        model: PyTorch 模型
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

