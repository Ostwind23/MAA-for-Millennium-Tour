import sys
import shutil
from pathlib import Path

try:
    import jsonc
except ModuleNotFoundError as e:
    raise ImportError(
        "Missing dependency 'json-with-comments' (imported as 'jsonc').\n"
        f"Install it with:\n  {sys.executable} -m pip install json-with-comments\n"
        "Or add it to your project's requirements."
    ) from e

ci_dir = Path(__file__).resolve().parent
tools_dir = ci_dir.parent
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

from configure import configure_ocr_model  # type: ignore
from utils import working_dir  # type: ignore


install_path = working_dir / "install"


def get_dotnet_platform_tag(os_name: str, arch: str) -> str:
    if os_name == "win" and arch == "x86_64":
        return "win-x64"
    if os_name == "win" and arch == "aarch64":
        return "win-arm64"
    if os_name == "macos" and arch == "x86_64":
        return "osx-x64"
    if os_name == "macos" and arch == "aarch64":
        return "osx-arm64"
    if os_name == "linux" and arch == "x86_64":
        return "linux-x64"
    if os_name == "linux" and arch == "aarch64":
        return "linux-arm64"
    print(f"Unsupported OS or architecture: {os_name}-{arch}")
    sys.exit(1)


def install_maafw(os_name: str, arch: str):
    if not (working_dir / "deps" / "bin").exists():
        print('Please download the MaaFramework to "deps" first.')
        print('请先下载 MaaFramework 到 "deps"。')
        sys.exit(1)

    shutil.copytree(
        working_dir / "deps" / "bin",
        install_path
        / "runtimes"
        / get_dotnet_platform_tag(os_name, arch)
        / "native",
        ignore=shutil.ignore_patterns(
            "*MaaDbgControlUnit*",
            "*MaaThriftControlUnit*",
            "*MaaRpc*",
            "*MaaHttp*",
            "plugins",
            "*.node",
            "*MaaPiCli*",
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        working_dir / "deps" / "share" / "MaaAgentBinary",
        install_path / "libs" / "MaaAgentBinary",
        dirs_exist_ok=True,
    )


def install_resource(version: str):
    configure_ocr_model()

    shutil.copytree(
        working_dir / "assets" / "resource",
        install_path / "resource",
        dirs_exist_ok=True,
    )
    shutil.copy2(
        working_dir / "assets" / "interface.json",
        install_path,
    )

    with open(install_path / "interface.json", "r", encoding="utf-8") as f:
        interface = jsonc.load(f)

    interface["version"] = version

    with open(install_path / "interface.json", "w", encoding="utf-8") as f:
        jsonc.dump(interface, f, ensure_ascii=False, indent=4)


def install_chores():
    shutil.copy2(
        working_dir / "README.md",
        install_path,
    )
    shutil.copy2(
        working_dir / "LICENSE",
        install_path,
    )


def install_agent(os_name: str):
    shutil.copytree(
        working_dir / "agent",
        install_path / "agent",
        dirs_exist_ok=True,
    )

    interface_path = install_path / "interface.json"
    if not interface_path.exists():
        print("Warning: interface.json not found in install directory.")
        return

    with open(interface_path, "r", encoding="utf-8") as f:
        interface = jsonc.load(f)

    agent_config = interface.get("agent", {})
    if os_name == "win":
        agent_config["child_exec"] = "python/python.exe"
    elif os_name == "macos":
        agent_config["child_exec"] = "python/bin/python3"
    elif os_name == "linux":
        agent_config["child_exec"] = "python3"
    else:
        print(f"Unsupported OS: {os_name}")
        sys.exit(1)

    agent_config["child_args"] = ["agent/main.py"]
    interface["agent"] = agent_config

    with open(interface_path, "w", encoding="utf-8") as f:
        jsonc.dump(interface, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python install.py <version> <os> <arch>")
        print("Example: python install.py v1.0.0 win x86_64")
        sys.exit(1)

    version = sys.argv[1]
    os_name = sys.argv[2]
    arch = sys.argv[3]

    install_maafw(os_name, arch)
    install_resource(version)
    install_chores()
    install_agent(os_name)

    print(f"Install to {install_path} successfully.")
