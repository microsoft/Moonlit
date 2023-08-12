from .macro_arch import StageConfig, MacroArch
from .utils import channel_list


class MobileMacroArchOriginal(MacroArch):

    def __init__(self) -> None:
        stage_configs = [
            #           name           width                            depth      stride
            StageConfig('first_conv',   channel_list(32, 40),            [1],       2),
            StageConfig('stage0',       channel_list(16, 24),            [1],       1),
            StageConfig('stage1',       channel_list(24, 32),            [2, 3, 4], 2),
            StageConfig('stage2',       channel_list(40, 56),            [2, 3, 4], 2),
            StageConfig('stage3',       channel_list(80, 104),           [2, 3, 4], 2),
            StageConfig('stage4',       channel_list(96, 128),           [2, 3, 4], 1),
            StageConfig('stage5',       channel_list(192, 256, step=16), [2, 3, 4], 2),
            StageConfig('stage6',       channel_list(320, 416, step=16), [2, 3, 4], 1),
            StageConfig('final_expand', channel_list(1280, 1280),        [1],       1),
            StageConfig('feature_mix',  channel_list(1920, 1920),        [1],       1),
            StageConfig('logits',       channel_list(1000, 1000),        [1],       1),
        ]
        block_kd_hw_list = channel_list(160, 224, 32)
        supernet_hw_list = channel_list(160, 224, 16)
        super().__init__(stage_configs, block_kd_hw_list=block_kd_hw_list, supernet_hw_list=supernet_hw_list, cin=3)
        
    @staticmethod
    def need_search(stage: StageConfig) -> bool:
        return stage.name not in ['final_expand', 'feature_mix', 'logits']


class MobileMacroArchV1(MacroArch):

    def __init__(self) -> None:
        stage_configs = [
            #           name           width                            depth      stride
            StageConfig('first_conv',   channel_list(32, 40),            [1],             2),
            StageConfig('stage0',       channel_list(16, 24),            [1],             1),
            StageConfig('stage1',       channel_list(24, 32),            [2, 3, 4],       2),
            StageConfig('stage2',       channel_list(40, 56),            [2, 3, 4],       2),
            StageConfig('stage3',       channel_list(80, 104),           [2, 3, 4],       2),
            StageConfig('stage4',       channel_list(96, 128),           [2, 3, 4, 5, 6], 1),
            StageConfig('stage5',       channel_list(192, 256, step=16), [2, 3, 4, 5, 6], 2),
            StageConfig('stage6',       channel_list(320, 416, step=16), [1, 2],          1),
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


class MobileMacroArchW(MacroArch):

    def __init__(self) -> None:
        stage_configs = [
            #           name           width                                           depth           stride    width window size
            StageConfig('first_conv',   channel_list(16, 32),                          [1],                   2),
            StageConfig('stage0',       channel_list(16, 32),                          [1, 2],                1),
            StageConfig('stage1',       channel_list(24, 40),                          [2, 3, 4],             2, 2),
            StageConfig('stage2',       channel_list(24, 72),                          [2, 3, 4],             2, 2),
            StageConfig('stage3',       channel_list(56, 112),                         [2, 3, 4, 5, 6],       2, 3),
            StageConfig('stage4',       channel_list(112, 176),                        [2, 3, 4, 5, 6, 7, 8], 1, 3),
            StageConfig('stage5',       [192, 200, 208, 216, 224, 240, 256, 272, 288], [2, 3, 4, 5, 6, 7, 8], 2, 5),
            StageConfig('stage6',       channel_list(216, 360, step=16),               [1, 2],                1, 7),
            StageConfig('final_expand', channel_list(1280, 1280),                      [1],                   1),
            StageConfig('feature_mix',  channel_list(1920, 1920),                      [1],                   1),
            StageConfig('logits',       channel_list(1000, 1000),                      [1],                   1),
        ]
        block_kd_hw_list = channel_list(160, 224, 32)
        supernet_hw_list = channel_list(160, 224, 16)
        super().__init__(stage_configs, block_kd_hw_list=block_kd_hw_list, supernet_hw_list=supernet_hw_list, cin=3)
        
    @staticmethod
    def need_search(stage: StageConfig) -> bool:
        return stage.name not in ['final_expand', 'feature_mix', 'logits']
