#!/usr/bin/env python3
"""
下载 Python 依赖到指定目录（用于 CI 打包）
"""

import sys
import subprocess
import argparse
from pathlib import Path


def get_platform_tag(os_name: str, arch: str) -> str:
    target = (os_name, arch)
    match target:
        case ("win", "x86_64"):
            return "win_amd64"
        case ("win", "aarch64"):
            return "win_arm64"
        case ("macos", "x86_64"):
            return "macosx_13_0_x86_64"
        case ("macos", "aarch64"):
            return "macosx_13_0_arm64"
        case ("linux", "x86_64"):
            return "manylinux2014_x86_64"
        case ("linux", "aarch64"):
            return "manylinux2014_aarch64"
        case _:
            print(f"不支持的操作系统或架构: {os_name}-{arch}")
            sys.exit(1)


def download_cross_platform(requirements_file: Path, deps_dir: Path, platform_tag: str):
    deps_dir.mkdir(parents=True, exist_ok=True)

    if not requirements_file.exists():
        print(f"错误: requirements.txt 文件不存在: {requirements_file}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_file),
        "--platform",
        platform_tag,
        "--only-binary",
        ":all:",
        "--no-deps",
        "--target",
        str(deps_dir),
    ]

    print(f"执行下载命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"依赖下载失败: {result.returncode}")
        sys.exit(result.returncode)

    print(f"依赖已经下载到目录: {deps_dir}")


def download_native(requirements_file: Path, deps_dir: Path):
    deps_dir.mkdir(parents=True, exist_ok=True)

    if not requirements_file.exists():
        print(f"错误: requirements.txt 文件不存在: {requirements_file}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_file),
        "--only-binary",
        ":all:",
        "--target",
        str(deps_dir),
    ]

    print(f"执行下载命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print("检测到失败，尝试不限制 binary 重新下载...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_file),
            "--target",
            str(deps_dir),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    if result.returncode != 0:
        print(f"依赖下载失败: {result.returncode}")
        sys.exit(result.returncode)

    print(f"依赖已经下载到目录: {deps_dir}")


def main():
    parser = argparse.ArgumentParser(description="下载 Python 依赖到目标目录")
    parser.add_argument("--deps-dir", default="deps", help="依赖下载目录")
    parser.add_argument("--requirements", default=None, help="依赖文件路径")
    parser.add_argument("--os", default=None, help="目标系统 (win/macos/linux)")
    parser.add_argument("--arch", default=None, help="目标架构 (x86_64/aarch64)")

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    requirements_file = (
        Path(args.requirements)
        if args.requirements
        else project_root / "agent" / "requirements.txt"
    )

    if args.os and args.arch:
        platform_tag = get_platform_tag(args.os, args.arch)
        download_cross_platform(requirements_file, Path(args.deps_dir), platform_tag)
    else:
        download_native(requirements_file, Path(args.deps_dir))


if __name__ == "__main__":
    main()
