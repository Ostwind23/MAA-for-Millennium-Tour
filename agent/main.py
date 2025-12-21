"""
MAA Agent Server 入口点
启动 AgentServer 并加载所有自定义 Action 和 Recognition
"""

import sys

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

# 导入注册中心，自动完成所有 Custom Action/Recognition 的注册
import agent_register


def main():
    Toolkit.init_option("./")

    if len(sys.argv) < 2:
        print("Usage: python main.py <socket_id>")
        print("socket_id is provided by AgentIdentifier.")
        sys.exit(1)

    socket_id = sys.argv[-1]

    # 打印已注册的信息（可选，便于调试）
    agent_register.print_registered_info()

    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
