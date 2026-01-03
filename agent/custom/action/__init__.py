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
]
