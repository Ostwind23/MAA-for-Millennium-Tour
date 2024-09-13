from pathlib import Path

import shutil
import sys
import json
import os
os.system(f"pip install {'pywin32'}")

from win32com.client import Dispatch

from configure import configure_ocr_model


working_dir = Path(__file__).parent
install_path = working_dir / Path("install")
version = len(sys.argv) > 1 and sys.argv[1] or "v0.1.2"
os.system('chcp 65001')


def install_deps():
    if not (working_dir / "deps" / "bin").exists():
        print("Please download the MaaFramework to \"deps\" first.")
        print("请先下载 MaaFramework 到 \"deps\"。")
        sys.exit(1)

    shutil.copytree(
        working_dir / "deps" / "bin",
        install_path,
        ignore=shutil.ignore_patterns(
            "*MaaDbgControlUnit*",
            "*MaaThriftControlUnit*",
            "*MaaRpc*",
            "*MaaHttp*",
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        working_dir / "deps" / "share" / "MaaAgentBinary",
        install_path / "MaaAgentBinary",
        dirs_exist_ok=True,
    )


def install_resource():

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
    shutil.copytree(
        working_dir / "assets" / "python",
        install_path / "python",
        dirs_exist_ok=True,
    )

    with open(install_path / "interface.json", "r", encoding="utf-8") as f:
        interface = json.load(f)

    interface["version"] = version

    with open(install_path / "interface.json", "w", encoding="utf-8") as f:
        json.dump(interface, f, ensure_ascii=False, indent=4)


def install_chores():
    shutil.copy2(
        working_dir / "README.md",
        install_path,
    )
    shutil.copy2(
        working_dir / "LICENSE",
        install_path,
    )


def create_shortcut():
    target = str(install_path / "python" / "Autofishing.exe")
    shortcut_path = str(install_path / "Autofishing.lnk")
    icon_path = str(install_path / "python" / "icon.ico")
    shell = Dispatch('WScript.Shell')

    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.Targetpath = target
    shortcut.IconLocation = icon_path
    shortcut.WorkingDirectory = str(install_path / "python")  # 设置起始位置
    shortcut.save()


if __name__ == "__main__":
    install_deps()
    install_resource()
    install_chores()
    create_shortcut()

    print(f"Install to {install_path} successfully.")