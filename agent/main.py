"""
MAA Agent Server 入口点
启动 AgentServer 并加载所有自定义 Action 和 Recognition
"""

import sys
from pathlib import Path

# 当前脚本所在目录（打包后为 install/agent）
_script_dir = Path(__file__).parent.resolve()

# 确保 agent 目录本身在 sys.path 中，便于导入同目录下的 agent_register 等模块
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

# 导入注册中心，自动完成所有 Custom Action/Recognition 的注册
import agent_register


def main():
    Toolkit.init_option("./")
    if len(sys.argv) >1:
        socket_id = sys.argv[-1]
    else:
        socket_id = "MAA_AGENT_SOCKET"
    # if len(sys.argv) < 2:
    #     print("Usage: python main.py <socket_id>")
    #     print("socket_id is provided by AgentIdentifier.")
    #     print(f"Starting Agent Server with socket ID: {socket_id}")
    #     sys.exit(1)

    # socket_id = sys.argv[-1]

    # 打印已注册的信息（可选，便于调试）
    agent_register.print_registered_info()
    print(f"Starting Agent Server with socket ID: {socket_id}")
    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
