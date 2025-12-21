"""
通用 Custom Action 实现
"""

from maa.custom_action import CustomAction
from maa.context import Context


class MyCustomAction(CustomAction):
    """
    示例自定义动作
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        print("my_action_111 is running!")

        return CustomAction.RunResult(success=True)
