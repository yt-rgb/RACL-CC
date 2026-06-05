"""
TA-Attention Stacked: Multi-Layer Flow-Guided Sparse Attention Module

这是在 KCPM-Attention 基础上的堆叠式增强版本，支持多层注意力堆叠。

核心改进：
1. 支持 1-4 层可配置的注意力堆叠
2. 渐进式窗口策略（大窗口 → 中窗口 → 小窗口）
3. 层间残差连接和 LayerNorm
4. 可选的跨层连接（DenseNet-style）
5. 可选的梯度检查点（节省内存）

使用方法：
    from racl.model.fusion.TA_attention_stacked import TA_Attention_Stacked
    
    model = KCPM_Attention_Stacked(
        in_channels=1024,
        num_attention_layers=3,
    )
    output = model(x1, x2, fusion_policy='abs_diff', use_attention=True)

作者: Kiro AI Assistant
日期: 2024-01-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, List, Dict

# 导入现有的组件
from .TA_attention import FlowGuidedSparseAttention, Flow_Block


class StackedFlowGuidedAttention(nn.Module):
    """
    堆叠式光流引导注意力模块
    
    支持多层注意力堆叠，每层可独立配置：
    - 窗口大小（window_size）
    - 注意力头数（num_heads）
    - QKV 降维比例（qkv_ratio）
    
    设计理念：
    1. 渐进式窗口策略：大窗口（粗粒度）→ 小窗口（细粒度）
    2. 层间残差连接：确保梯度流畅
    3. LayerNorm：稳定训练
    4. 可选跨层连接：增强特征交互
    5. 可选梯度检查点：节省内存
    
    Args:
        dim: 输入特征维度
        num_layers: 注意力层数（1-4）
        layer_configs: 每层的配置列表，每个配置包含：
            - num_heads: 注意力头数
            - window_size: 窗口大小
            - qkv_ratio: QKV 降维比例
        use_cross_layer_connection: 是否使用跨层连接（DenseNet-style）
        use_gradient_checkpointing: 是否使用梯度检查点
    
    Example:
        >>> model = StackedFlowGuidedAttention(
        ...     dim=1024,
        ...     num_layers=3,
        ... )
        >>> x1 = torch.randn(2, 256, 1024)
        >>> x2 = torch.randn(2, 256, 1024)
        >>> flow = torch.randn(2, 4, 16, 16)
        >>> output = model(x1, x2, flow)
        >>> print(output.shape)  # torch.Size([2, 256, 1024])
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int = 3,
        layer_configs: Optional[List[Dict]] = None,
        use_cross_layer_connection: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        assert 1 <= num_layers <= 4, f"num_layers must be in [1, 4], got {num_layers}"
        
        self.dim = dim
        self.num_layers = num_layers
        self.use_cross_layer_connection = use_cross_layer_connection
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # ============ 默认配置：渐进式窗口策略 ============
        if layer_configs is None:
            # 确保 qkv_dim 能被 num_heads 整除
            # 对于 dim=1024, qkv_ratio=0.5 -> qkv_dim=512
            # 512 能被 4, 8, 16 整除，但不能被 6 整除
            layer_configs = [
                {'num_heads': 4, 'window_size': 9, 'qkv_ratio': 0.5},  # Layer 1: 粗粒度全局上下文
                {'num_heads': 8, 'window_size': 7, 'qkv_ratio': 0.5},  # Layer 2: 中粒度语义理解
                {'num_heads': 8, 'window_size': 5, 'qkv_ratio': 0.5},  # Layer 3: 细粒度局部细节
                {'num_heads': 8, 'window_size': 5, 'qkv_ratio': 0.5},  # Layer 4: 超细粒度（可选）
            ]
        
        # 验证配置数量
        if len(layer_configs) < num_layers:
            raise ValueError(
                f"layer_configs must have at least {num_layers} configs, "
                f"got {len(layer_configs)}"
            )
        
        # ============ 创建注意力层 ============
        self.attention_layers = nn.ModuleList([
            FlowGuidedSparseAttention(
                dim=dim,
                num_heads=config['num_heads'],
                window_size=config['window_size'],
                qkv_ratio=config.get('qkv_ratio', 0.5)
            )
            for config in layer_configs[:num_layers]
        ])
        
        # ============ 创建 LayerNorm 层 ============
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        
        # ============ 可选：跨层连接的投影层 ============
        if use_cross_layer_connection:
            # 从第 2 层开始，每层需要一个投影层
            # 投影层将拼接的特征（dim * (i+1)）投影回原始维度（dim）
            self.cross_layer_projections = nn.ModuleList([
                nn.Linear(dim * (i + 1), dim) 
                for i in range(1, num_layers)
            ])
        else:
            self.cross_layer_projections = None
        
        # 存储配置信息（用于调试和可视化）
        self.layer_configs = layer_configs[:num_layers]
    
    def forward(self, x1, x2, flow_guide):
        """
        前向传播
        
        Args:
            x1: 前时相特征 [B, N, C]，N = H×W
            x2: 后时相特征 [B, N, C]
            flow_guide: 光流 [B, 4, H, W]，前 2 通道是 A→B，后 2 通道是 B→A
        
        Returns:
            output: 堆叠注意力输出 [B, N, C]
        
        流程：
            1. 初始化：output = x1
            2. 对于每一层：
               a. 计算注意力：attn_out = attention_layer(output, x2, flow_guide)
               b. 残差连接：output = output + attn_out
               c. LayerNorm：output = layer_norm(output)
               d. 可选跨层连接：output = project(concat(all_previous_outputs))
            3. 返回最终输出
        """
        B, N, C = x1.shape
        
        # 验证输入
        assert x1.shape == x2.shape, f"x1 and x2 must have the same shape"
        assert C == self.dim, f"Input dim {C} != model dim {self.dim}"
        
        # 初始化
        output = x1  # 从 x1 开始（前时相特征）
        layer_outputs = []  # 存储每层输出（用于跨层连接）
        
        # ============ 逐层处理 ============
        for i, (attn_layer, norm_layer) in enumerate(
            zip(self.attention_layers, self.layer_norms)
        ):
            # -------- 步骤 1：计算注意力 --------
            if self.use_gradient_checkpointing and self.training:
                # 使用梯度检查点（节省内存，但增加计算时间）
                attn_out = checkpoint(
                    self._attention_forward,
                    attn_layer,
                    output,
                    x2,
                    flow_guide,
                    use_reentrant=False  # PyTorch 2.0+ 推荐设置
                )
            else:
                # 正常前向传播
                attn_out = attn_layer(output, x2, flow_guide)
            
            # -------- 步骤 2：残差连接 --------
            output = output + attn_out
            
            # -------- 步骤 3：LayerNorm --------
            output = norm_layer(output)
            
            # -------- 步骤 4：可选跨层连接 --------
            if self.use_cross_layer_connection:
                # 存储当前层输出
                layer_outputs.append(output)
                
                # 从第 2 层开始，使用跨层连接
                if i > 0:
                    # 拼接所有之前的层输出
                    concat_features = torch.cat(layer_outputs, dim=-1)  # [B, N, dim*(i+1)]
                    
                    # 投影回原始维度
                    output = self.cross_layer_projections[i - 1](concat_features)  # [B, N, dim]
        
        return output
    
    @staticmethod
    def _attention_forward(attn_layer, output, x2, flow_guide):
        """
        静态方法：用于梯度检查点
        
        这是一个包装函数，用于 torch.utils.checkpoint.checkpoint
        """
        return attn_layer(output, x2, flow_guide)
    
    def get_num_parameters(self):
        """
        统计参数量
        
        Returns:
            total_params: 总参数量
            layer_params: 每层参数量列表
        """
        layer_params = []
        for i, layer in enumerate(self.attention_layers):
            params = sum(p.numel() for p in layer.parameters())
            layer_params.append(params)
        
        total_params = sum(layer_params)
        
        # 添加 LayerNorm 参数
        norm_params = sum(p.numel() for p in self.layer_norms.parameters())
        total_params += norm_params
        
        # 添加跨层投影参数（如果有）
        if self.cross_layer_projections is not None:
            proj_params = sum(p.numel() for p in self.cross_layer_projections.parameters())
            total_params += proj_params
        
        return total_params, layer_params


class TA_Attention_Stacked(nn.Module):
    """
    KCPM + 堆叠式光流引导注意力
    
    完全兼容 KCPM_Attention 的接口，新增堆叠配置：
    
    架构：
    1. KCPM 光流对齐（粗对齐）
       - 4 个 Flow_Block 处理局部光流
       - 1 个 flow_make_g 处理全局光流
    2. 堆叠式注意力精细化（可选）
       - 多层光流引导注意力
       - 渐进式窗口策略
       - 层间残差连接和 LayerNorm
    3. 门控融合
       - 自适应平衡 KCPM 和注意力的输出
    
    Args:
        in_channels: 输入特征通道数（如 1024）
        num_attention_layers: 注意力层数（1-4）
        attention_layer_configs: 每层配置列表（可选）
        use_gate: 是否使用门控融合
        use_cross_layer_connection: 是否使用跨层连接
        use_gradient_checkpointing: 是否使用梯度检查点
        enable_attention_by_default: 是否默认启用注意力
    
    Example:
        >>> model = KCPM_Attention_Stacked(
        ...     in_channels=1024,
        ...     num_attention_layers=3,
        ... )
        >>> x1 = torch.randn(2, 256, 1024)
        >>> x2 = torch.randn(2, 256, 1024)
        >>> output = model(x1, x2, fusion_policy='abs_diff', use_attention=True)
        >>> print(output.shape)  # torch.Size([2, 256, 1024])
    """
    
    def __init__(
        self,
        in_channels: int,
        num_attention_layers: int = 3,
        attention_layer_configs: Optional[List[Dict]] = None,
        use_gate: bool = True,
        use_cross_layer_connection: bool = False,
        use_gradient_checkpointing: bool = False,
        enable_attention_by_default: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels // 4
        self.use_gate = use_gate
        self.enable_attention_by_default = enable_attention_by_default
        
        # ============ KCPM 组件（与原始一致） ============
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
        
        # ============ 新增：堆叠式注意力模块 ============
        self.attention_refine = StackedFlowGuidedAttention(
            dim=in_channels,
            num_layers=num_attention_layers,
            layer_configs=attention_layer_configs,
            use_cross_layer_connection=use_cross_layer_connection,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        # ============ 门控融合模块 ============
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
            fusion_policy: 特征融合策略
                - None: 返回 (x1_feat_g, x2_feat_g) 元组
                - 'abs_diff': 绝对差分（推荐）
                - 'diff': 有向差分
                - 'concat': 拼接
                - 'sum': 相加
            use_attention: 是否使用注意力精细化
                - None: 使用 enable_attention_by_default 的设置
                - True: 启用堆叠式注意力
                - False: 只用 KCPM（完全等价于原始 KCPM）
        
        Returns:
            output: 融合后的特征 [B, N, C]
            或 (x1_feat_g, x2_feat_g) 元组（当 fusion_policy is None 时）
        """
        # 确定是否使用注意力
        if use_attention is None:
            use_attention = self.enable_attention_by_default
        
        B, N, C = x1.size()
        H = int(N ** 0.5)
        W = H
        
        # 输入验证
        assert x1.shape == x2.shape, f"x1 and x2 must have the same shape"
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
        
        # ============ 阶段 2: 堆叠式注意力精细化（可选） ============
        if use_attention:
            # 将特征转回 [B, N, C] 格式
            TA_flat = TA_output.view(B, C, N).permute(0, 2, 1)
            
            # 使用堆叠式注意力（复用 KCPM 的光流）
            attn_output = self.attention_refine(
                x1, x2, 
                flow_guide=flow_g  # 关键：复用 KCPM 的光流
            )
            
            # 门控融合：动态平衡 KCPM 和注意力
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
        光流变形函数（复用自原始 KCPM）
        
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
    
    def get_num_parameters(self):
        """
        统计参数量
        
        Returns:
            dict: 包含各部分参数量的字典
        """
        # KCPM 参数
        TA_params = sum(p.numel() for p in self.flows.parameters())
        TA_params += sum(p.numel() for p in self.flow_make_g.parameters())
        
        # 注意力参数
        attn_params, layer_params = self.attention_refine.get_num_parameters()
        
        # 门控参数
        gate_params = 0
        if self.use_gate:
            gate_params = sum(p.numel() for p in self.fusion_gate.parameters())
        
        # 总参数
        total_params = TA_params + attn_params + gate_params
        
        return {
            'total': total_params,
            'TA': TA_params,
            'attention': attn_params,
            'attention_layers': layer_params,
            'gate': gate_params,
        }


# ============================================================================
# 辅助函数
# ============================================================================

def count_parameters_by_layer(model):
    """
    统计堆叠式注意力模型的参数量（按层）
    
    Args:
        model: KCPM_Attention_Stacked 实例
    
    Returns:
        None（打印结果）
    """
    if not isinstance(model, KCPM_Attention_Stacked):
        raise TypeError("model must be an instance of KCPM_Attention_Stacked")
    
    params_dict = model.get_num_parameters()
    
    print("=" * 60)
    print("参数量统计")
    print("=" * 60)
    print(f"KCPM 组件:        {params_dict['TA']:>12,} parameters")
    print(f"注意力模块:      {params_dict['attention']:>12,} parameters")
    
    for i, layer_params in enumerate(params_dict['attention_layers']):
        print(f"  - Layer {i+1}:    {layer_params:>12,} parameters")
    
    print(f"门控融合:        {params_dict['gate']:>12,} parameters")
    print("-" * 60)
    print(f"总计:            {params_dict['total']:>12,} parameters")
    print("=" * 60)


def create_default_stacked_model(in_channels=1024, num_layers=3, **kwargs):
    """
    创建默认配置的堆叠式注意力模型
    
    Args:
        in_channels: 输入通道数
        num_layers: 注意力层数
        **kwargs: 其他参数
    
    Returns:
        model: KCPM_Attention_Stacked 实例
    """
    model = KCPM_Attention_Stacked(
        in_channels=in_channels,
        num_attention_layers=num_layers,
        use_gate=True,
        use_cross_layer_connection=False,
        use_gradient_checkpointing=False,
        enable_attention_by_default=True,
        **kwargs
    )
    
    return model
