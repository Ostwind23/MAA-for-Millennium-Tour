"""
Agent 注册中心
统一导入并注册所有 Custom Action 和 Custom Recognition

使用方法：
    在 main.py 中 import agent_register 即可自动完成所有注册
"""

from maa.agent.agent_server import AgentServer

# ===== 导入所有 Custom Action =====
from custom.action import (
    MyCustomAction,
    InitSeedSelection,
    # 地牢刷关
    DungeonFullAuto,
)

# ===== 导入所有 Custom Recognition =====
from custom.reco import (
    MyRecognition,
)


# ===== 注册 Custom Actions =====
# 格式: AgentServer.register_custom_action("注册名称", 类实例)

AgentServer.register_custom_action("my_action_111", MyCustomAction())
AgentServer.register_custom_action("InitSeedSelection", InitSeedSelection())

# 地牢刷关 - 唯一入口，自动循环到 6-1 停止
# Pipeline 调用示例: {"custom_action": "DungeonFullAuto", "custom_action_param": "{\"max_stages\": 60}"}
AgentServer.register_custom_action("DungeonFullAuto", DungeonFullAuto())


# ===== 注册 Custom Recognitions =====
# 格式: AgentServer.register_custom_recognition("注册名称", 类实例)

AgentServer.register_custom_recognition("my_reco_222", MyRecognition())


# ===== 注册信息汇总（便于调试）=====
REGISTERED_ACTIONS = [
    "my_action_111",
    "InitSeedSelection",
    "DungeonFullAuto",  # 地牢刷关唯一入口
]

REGISTERED_RECOGNITIONS = [
    "my_reco_222",
]


def print_registered_info():
    """打印已注册的 Action 和 Recognition 信息"""
    print("=" * 50)
    print("已注册的 Custom Actions:")
    for name in REGISTERED_ACTIONS:
        print(f"  - {name}")
    print()
    print("已注册的 Custom Recognitions:")
    for name in REGISTERED_RECOGNITIONS:
        print(f"  - {name}")
    print("=" * 50)

