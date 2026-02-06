# Custom Action 实现模块
# 在这里导出所有 action 类，供注册中心使用

from .general import MyCustomAction
from .seed_pick import InitSeedSelection
from .dungeon import (
    DungeonFullAuto,  # 地牢刷关唯一入口
    # 以下为内部辅助类，可选导出
    DungeonNavigator,
    DungeonStageSelector,
    DungeonSwipeRight,
    DungeonSwipeLeft,
    DungeonAutoProgress,
    DungeonCheckCompleted,
    DungeonTryQuickBattle,
    DungeonNormalBattle,
    DungeonBattleFlow,
    DungeonSetQuickBattleFlag,
    DungeonSelectCardEffect,
    DungeonCompleteStage,
)
from .battle_mode import BattleModeManager
from .pipeline_override import PipelineConfigOverride, BattleModeConfigOverride
from .screenshot_collector import (
    ScreenshotCollector,
    BatchScreenshotCollector,
    ConditionalScreenshotCollector,
)
from .farm_event import (
    FarmEventHandler,      # 农场事件处理器（通用入口）
    FarmWaterwheelRepair,  # 水车修理专用
    FarmWindmillRepair,    # 风车修理专用
    FarmWormCatching,      # 捉虫专用
    FarmWateringAll,       # 全农场浇水专用
)
from .test_run_reco import TestRunRecoHandler  # 测试 NeuralNetworkDetect
from .fishing import AutoFishing  # 自动钓鱼

__all__ = [
    "MyCustomAction",
    "InitSeedSelection",
    # 地牢刷关主入口
    "DungeonFullAuto",
    # 战斗模式管理
    "BattleModeManager",
    # Pipeline 配置覆盖
    "PipelineConfigOverride",
    "BattleModeConfigOverride",
    # 辅助action（可选，调试用）
    "DungeonNavigator",
    "DungeonStageSelector",
    "DungeonSwipeRight",
    "DungeonSwipeLeft",
    "DungeonAutoProgress",
    "DungeonCheckCompleted",
    "DungeonTryQuickBattle",
    "DungeonNormalBattle",
    "DungeonBattleFlow",
    "DungeonSetQuickBattleFlag",
    "DungeonSelectCardEffect",
    "DungeonCompleteStage",
    # 截图采集器（YOLOv8 数据集收集）
    "ScreenshotCollector",
    "BatchScreenshotCollector",
    "ConditionalScreenshotCollector",
    # 农场事件处理
    "FarmEventHandler",
    "FarmWaterwheelRepair",
    "FarmWindmillRepair",
    "FarmWormCatching",
    "FarmWateringAll",
    # 测试工具
    "TestRunRecoHandler",
    # 自动钓鱼
    "AutoFishing",
]
