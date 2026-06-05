from .TA import TA
from .TA_attention import TA_Attention
from .TA_attention_stacked import TA_Attention_Stacked

def build_change_detector(config, **kwargs):
    """
    构建变化检测器模块
    
    支持的类型：
    - 'TA': 原始 KCPM（粗对齐）
    - 'TA_attention': KCPM + 单层光流引导注意力（粗对齐 + 精细化）
    - 'TA_attention_stacked': KCPM + 堆叠式光流引导注意力（多层精细化）
    
    配置参数：
    - mm_change_detector_type: 检测器类型
    - mm_hidden_size: 特征通道数
    
    对于 'TA_attention':
    - use_attention: 是否启用注意力（默认 False）
    - attention_num_heads: 注意力头数（默认 4）
    - attention_window_size: 注意力窗口大小（默认 7）
    - use_gate: 是否使用门控融合（默认 True）
    
    对于 'TA_attention_stacked':
    - num_attention_layers: 注意力层数（默认 3）
    - attention_layer_configs: 每层配置列表（可选）
    - use_attention: 是否启用注意力（默认 True）
    - use_gate: 是否使用门控融合（默认 True）
    - use_cross_layer_connection: 是否使用跨层连接（默认 False）
    - use_gradient_checkpointing: 是否使用梯度检查点（默认 False）
    """
    mm_change_hidden_size = config.mm_hidden_size
    detector_type = config.mm_change_detector_type
    
    if detector_type == "TA":
        # 原始 KCPM
        return TA(in_channels=mm_change_hidden_size)
    
    elif detector_type == "TA_attention":
        # KCPM + 单层注意力精细化
        # 从配置中获取参数，提供默认值
        num_heads = getattr(config, 'attention_num_heads', 4)
        window_size = getattr(config, 'attention_window_size', 7)
        use_gate = getattr(config, 'use_gate', True)
        enable_attention_by_default = getattr(config, 'use_attention', False)
        
        return TA_Attention(
            in_channels=mm_change_hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            use_gate=use_gate,
            enable_attention_by_default=enable_attention_by_default
        )
    
    elif detector_type == "TA_attention_stacked":
        # KCPM + 堆叠式注意力精细化（新增）
        # 从配置中获取参数，提供默认值
        num_attention_layers = getattr(config, 'num_attention_layers', 3)
        attention_layer_configs = getattr(config, 'attention_layer_configs', None)
        use_gate = getattr(config, 'use_gate', True)
        use_cross_layer_connection = getattr(config, 'use_cross_layer_connection', False)
        use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        enable_attention_by_default = getattr(config, 'use_attention', True)
        
        return TA_Attention_Stacked(
            in_channels=mm_change_hidden_size,
            num_attention_layers=num_attention_layers,
            attention_layer_configs=attention_layer_configs,
            use_gate=use_gate,
            use_cross_layer_connection=use_cross_layer_connection,
            use_gradient_checkpointing=use_gradient_checkpointing,
            enable_attention_by_default=enable_attention_by_default
        )
    
    else:
        raise ValueError(
            f"Unsupported mm_change_detector_type: {detector_type}. "
            f"Supported types: ['TA', 'TA_attention', 'TA_attention_stacked']"
        )