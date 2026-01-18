#!/usr/bin/env python3
"""
下载 Python 依赖包到指定目录

用于 CI 构建时预下载依赖，打包后用户无需联网安装。

用法：
    python download_deps.py --deps-dir install/deps [--requirements agent/requirements.txt]
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def get_platform_tag():
    """获取当前平台的 pip wheel 标签"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        if machine in ("x86_64", "amd64"):
            return "win_amd64"
        elif machine in ("aarch64", "arm64"):
            return "win_arm64"
    elif system == "darwin":
        if machine in ("x86_64", "amd64"):
            return "macosx_10_9_x86_64"
        elif machine in ("aarch64", "arm64"):
            return "macosx_11_0_arm64"
    elif system == "linux":
        if machine in ("x86_64", "amd64"):
            return "manylinux2014_x86_64"
        elif machine in ("aarch64", "arm64"):
            return "manylinux2014_aarch64"
    
    return None


def download_deps(requirements_file: Path, deps_dir: Path, python_exe: str = None):
    """下载依赖包到指定目录"""
    
    if not requirements_file.exists():
        print(f"[错误] 依赖文件不存在: {requirements_file}")
        sys.exit(1)
    
    # 创建目录
    deps_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定 Python 可执行文件
    if python_exe is None:
        python_exe = sys.executable
    
    print(f"[信息] Python: {python_exe}")
    print(f"[信息] 依赖文件: {requirements_file}")
    print(f"[信息] 输出目录: {deps_dir}")
    
    # 读取依赖列表
    with open(requirements_file, "r", encoding="utf-8") as f:
        deps = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
    
    print(f"[信息] 需要下载的依赖: {deps}")
    
    # 使用 pip download 下载依赖
    # -d: 下载目录
    # --only-binary :all: 只下载预编译的 wheel（避免编译）
    # --platform: 目标平台
    # --python-version: 目标 Python 版本
    
    cmd = [
        python_exe, "-m", "pip", "download",
        "-d", str(deps_dir),
        "--only-binary", ":all:",
        "-r", str(requirements_file),
    ]
    
    print(f"[执行] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"[警告] pip download 返回非零状态: {result.returncode}")
        print("[信息] 尝试不带 --only-binary 重新下载...")
        
        # 重试，不限制 binary
        cmd = [
            python_exe, "-m", "pip", "download",
            "-d", str(deps_dir),
            "-r", str(requirements_file),
        ]
        result = subprocess.run(cmd, capture_output=False)
    
    # 统计下载的文件
    downloaded = list(deps_dir.glob("*.whl")) + list(deps_dir.glob("*.tar.gz"))
    print(f"\n[完成] 已下载 {len(downloaded)} 个包到 {deps_dir}")
    for f in downloaded:
        print(f"  - {f.name}")
    
    # 安装依赖到 deps 目录（解压 wheel）
    print("\n[信息] 正在安装依赖到目标目录...")
    install_cmd = [
        python_exe, "-m", "pip", "install",
        "--target", str(deps_dir),
        "--no-deps",  # 不安装依赖的依赖（已经下载了）
        "-r", str(requirements_file),
    ]
    
    print(f"[执行] {' '.join(install_cmd)}")
    subprocess.run(install_cmd, capture_output=False)
    
    # 清理 wheel 文件（已经安装）
    for whl in deps_dir.glob("*.whl"):
        whl.unlink()
    for tar in deps_dir.glob("*.tar.gz"):
        tar.unlink()
    
    print(f"\n[完成] 依赖已安装到 {deps_dir}")


def main():
    import argparse
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent
    
    parser = argparse.ArgumentParser(description="下载 Python 依赖包")
    parser.add_argument(
        "--deps-dir", 
        default="install/deps", 
        help="依赖包输出目录"
    )
    parser.add_argument(
        "--requirements", 
        default=str(project_root / "agent" / "requirements.txt"),
        help="依赖文件路径"
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python 可执行文件路径"
    )
    args = parser.parse_args()
    
    download_deps(
        requirements_file=Path(args.requirements).resolve(),
        deps_dir=Path(args.deps_dir).resolve(),
        python_exe=args.python,
    )


if __name__ == "__main__":
    main()
