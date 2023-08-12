from .macro_arch import StageConfig, MacroArch
from .utils import channel_list


class OnnxMacroArch(MacroArch):
    
    def __init__(self) -> None:
        stage_configs = [
            #           name           width                            depth      stride
            StageConfig('first_conv',   channel_list(16, 32, step=16),   [1],             2),
            StageConfig('stage0',       channel_list(16, 32, step=16),   [1, 2],          1),
            StageConfig('stage1',       channel_list(32, 48, step=16),   [2, 3],          2),
            StageConfig('stage2',       channel_list(48, 64, step=16),   [2, 3, 4],       2),
            StageConfig('stage3',       channel_list(80, 128, step=16),  [2, 3, 4, 5, 6], 2),
            StageConfig('stage4',       channel_list(128, 144, step=16), [2, 3, 4, 5, 6], 1),
            StageConfig('stage5',       channel_list(192, 256, step=16), [2, 3, 4, 5, 6], 2),
            StageConfig('stage6',       channel_list(320, 432, step=16), [1, 2],          1),
            StageConfig('final_expand', channel_list(1280, 1280),        [1],             1),
            StageConfig('feature_mix',  channel_list(1920, 1920),        [1],             1),
            StageConfig('logits',       channel_list(1000, 1000),        [1],             1),
        ]
        block_kd_hw_list = channel_list(160, 224, 32)
        supernet_hw_list = channel_list(160, 224, 16)
        super().__init__(stage_configs, block_kd_hw_list=block_kd_hw_list, supernet_hw_list=supernet_hw_list, cin=3)
        
    @staticmethod
    def need_search(stage: StageConfig) -> bool:
        return stage.name not in ['final_expand', 'feature_mix', 'logits']


class OnnxMacroArchW(MacroArch):
    
    def __init__(self) -> None:
        stage_configs = [
            #           name           width                            depth      stride   width_window_size
            StageConfig('first_conv',   channel_list(16, 32, step=16),   [1],             2),
            StageConfig('stage0',       channel_list(16, 32, step=16),   [1, 2],          1),
            StageConfig('stage1',       channel_list(32, 64, step=16),   [2, 3, 4],       2, 2),
            StageConfig('stage2',       channel_list(32, 96, step=16),   [2, 3, 4],       2, 2),
            StageConfig('stage3',       channel_list(64, 144, step=16),  [2, 3, 4, 5, 6], 2, 3),
            StageConfig('stage4',       channel_list(112, 192, step=16), [2, 3, 4, 5, 6], 1, 3),
            StageConfig('stage5',       channel_list(192, 304, step=16), [2, 3, 4, 5, 6], 2, 5),
            StageConfig('stage6',       channel_list(304, 448, step=16), [1, 2],          1, 7),
            StageConfig('final_expand', channel_list(1280, 1280),        [1],             1),
            StageConfig('feature_mix',  channel_list(1920, 1920),        [1],             1),
            StageConfig('logits',       channel_list(1000, 1000),        [1],             1),
        ]
        block_kd_hw_list = channel_list(160, 224, 32)
        supernet_hw_list = channel_list(160, 224, 16)
        super().__init__(stage_configs, block_kd_hw_list=block_kd_hw_list, supernet_hw_list=supernet_hw_list, cin=3)
        
    @staticmethod
    def need_search(stage: StageConfig) -> bool:
        return stage.name not in ['final_expand', 'feature_mix', 'logits']