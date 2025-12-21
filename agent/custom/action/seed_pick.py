"""
种子购买选择 Custom Action
"""

from maa.custom_action import CustomAction
from maa.context import Context
import json


# 种子名称到模板路径的映射
SEED_TEMPLATES = {
    "龙鳞果": "seed/龙鳞果.png",
    "青海莓": "seed/青海莓.png",
    "千蝶菇": "seed/千蝶菇.png",
    "黄金葡萄": "seed/黄金葡萄.png",
    "连叶卷心菜": "seed/连叶卷心菜.png",
    "拇指玉米": "seed/拇指玉米.png",
    "月牙马铃薯": "seed/月牙马铃薯.png",
    "火焰大番茄": "seed/火焰大番茄.png",
    "黑松露": "seed/黑松露.png",
    "龟甲菇": "seed/龟甲菇.png",
    "天谕橙": "seed/天谕橙.png",
    "绿翠绒": "seed/绿翠绒.png",
}


class InitSeedSelection(CustomAction):
    """
    读取 Seed_Mark 节点的 attach 标志，根据用户选择动态覆写 Findseed 节点的 template 列表。
    
    参数格式:
    {
        "flag_node": "Seed_Mark",      // 存储标志的节点名
        "target_node": "Findseed"       // 要覆写的目标节点名
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        # 1. 解析传入的参数
        param = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        flag_node = param.get("flag_node", "Seed_Mark")
        target_node = param.get("target_node", "Findseed")

        print(f"[InitSeedSelection] 读取标志节点: {flag_node}")

        # 2. 读取标志节点的当前定义
        try:
            node_json = context.get_node_data(flag_node)
            node_data = json.loads(node_json)
            attach = node_data.get("attach", {})
            print(f"[InitSeedSelection] attach 内容: {attach}")
        except Exception as e:
            print(f"[InitSeedSelection] 读取标志节点失败: {e}")
            return CustomAction.RunResult(success=False)

        # 3. 根据 attach 中的标志构建模板列表
        templates = [
            tpl for seed, tpl in SEED_TEMPLATES.items()
            if attach.get(seed, False)
        ]

        selected_seeds = [k for k, v in attach.items() if v]
        print(f"[InitSeedSelection] 用户选择的种子: {selected_seeds}")
        print(f"[InitSeedSelection] 生成的模板列表: {templates}")

        # 4. 覆写目标节点
        if templates:
            context.override_pipeline({
                target_node: {
                    "template": templates
                }
            })
            print(f"[InitSeedSelection] 已覆写 {target_node} 的 template，共 {len(templates)} 个")
        else:
            # 没有选择任何种子，禁用该节点
            context.override_pipeline({
                target_node: {
                    "enabled": False
                }
            })
            print(f"[InitSeedSelection] 未选择任何种子，已禁用 {target_node}")

        return CustomAction.RunResult(success=True)
