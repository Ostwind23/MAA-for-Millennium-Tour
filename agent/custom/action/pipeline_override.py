"""
Pipeline 配置覆盖 Custom Action
用于在任务链执行时动态覆盖其他节点的参数

主要解决问题：
MaaFramework 的 interface.json 中 pipeline_override 是在初始化时全局完成的，
导致不同任务设置的不同参数会被最后一个覆盖。

此 Action 可在任务链执行时读取当前节点的 attach 字段，
并实时覆盖指定目标节点的参数。
"""

import json
from maa.custom_action import CustomAction
from maa.context import Context


class PipelineConfigOverride(CustomAction):
    """
    Pipeline 配置覆盖器
    
    功能：
    读取当前节点的 attach 字段中的配置，实时覆盖目标节点的参数。
    
    Pipeline 调用示例:
    {
        "recognition": "DirectHit",
        "action": "Custom",
        "custom_action": "PipelineConfigOverride",
        "max_hit": 1,
        "attach": {
            "target_node": "BattleMode_Check",
            "override_params": {
                "custom_action_param": {"target_mode": "manual_skip"}
            }
        }
    }
    
    attach 字段说明:
    - target_node: 目标节点名称，要覆盖其参数的节点
    - override_params: 要覆盖的参数字典，会直接合并到目标节点的定义中
    
    注意事项:
    - 建议配合 max_hit: 1 使用，确保每个任务链只执行一次配置
    - 此 Action 应放在需要使用被覆盖节点之前执行
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """执行配置覆盖逻辑"""
        
        # 从 custom_action_param 获取配置信息
        # 注意：attach 字段的内容需要通过 custom_action_param 传递
        raw_param = argv.custom_action_param
        
        # 解析参数
        if isinstance(raw_param, dict):
            config = raw_param
        elif isinstance(raw_param, str) and raw_param.strip():
            try:
                config = json.loads(raw_param)
            except json.JSONDecodeError as e:
                print(f"[PipelineConfigOverride] JSON 解析失败: {e}")
                return CustomAction.RunResult(success=False)
        else:
            print("[PipelineConfigOverride] 未提供配置参数")
            return CustomAction.RunResult(success=False)
        
        # 获取目标节点和覆盖参数
        target_node = config.get("target_node")
        override_params = config.get("override_params")
        
        if not target_node:
            print("[PipelineConfigOverride] 缺少 target_node 参数")
            return CustomAction.RunResult(success=False)
        
        if not override_params:
            print("[PipelineConfigOverride] 缺少 override_params 参数")
            return CustomAction.RunResult(success=False)
        
        # 构建覆盖字典
        pipeline_override = {target_node: override_params}
        
        print(f"[PipelineConfigOverride] 覆盖节点 '{target_node}' 的参数: {override_params}")
        
        # 执行覆盖
        context.override_pipeline(pipeline_override)
        
        print(f"[PipelineConfigOverride] 配置覆盖完成")
        
        return CustomAction.RunResult(success=True)


class BattleModeConfigOverride(CustomAction):
    """
    战斗模式配置覆盖器 - PipelineConfigOverride 的简化版
    
    专门用于覆盖 BattleMode_Check 节点的 custom_action_param 参数。
    
    Pipeline 调用示例:
    {
        "recognition": "DirectHit",
        "action": "Custom", 
        "custom_action": "BattleModeConfigOverride",
        "custom_action_param": {"target_mode": "auto"},
        "max_hit": 1
    }
    
    custom_action_param 字段说明:
    - target_mode: 目标战斗模式
      - "auto": 自动战斗模式
      - "manual_skip": 手动+跳过模式
    
    注意事项:
    - 建议配合 max_hit: 1 使用，确保每个任务链只执行一次配置
    - 此节点应放在任务链中 BattleMode_Check 之前执行
    """
    
    # 目标节点名称
    TARGET_NODE = "BattleMode_Check"

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """执行战斗模式配置覆盖"""
        
        raw_param = argv.custom_action_param
        
        # 解析参数
        if isinstance(raw_param, dict):
            mode_config = raw_param
        elif isinstance(raw_param, str) and raw_param.strip():
            try:
                mode_config = json.loads(raw_param)
            except json.JSONDecodeError as e:
                print(f"[BattleModeConfigOverride] JSON 解析失败: {e}")
                return CustomAction.RunResult(success=False)
        else:
            print("[BattleModeConfigOverride] 未提供配置参数，使用默认模式 'auto'")
            mode_config = {"target_mode": "auto"}
        
        target_mode = mode_config.get("target_mode", "auto")
        
        # 构建覆盖字典 - 覆盖 BattleMode_Check 的 custom_action_param
        pipeline_override = {
            self.TARGET_NODE: {
                "custom_action_param": {"target_mode": target_mode}
            }
        }
        
        print(f"[BattleModeConfigOverride] 设置战斗模式为: {target_mode}")
        
        # 执行覆盖
        context.override_pipeline(pipeline_override)
        
        print(f"[BattleModeConfigOverride] 战斗模式配置完成")
        
        return CustomAction.RunResult(success=True)
