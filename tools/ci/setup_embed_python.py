#!/usr/bin/env python3
"""
下载并设置嵌入式 Python 环境

支持平台：
- Windows: 使用官方 embed 版本
- macOS: 使用 python-build-standalone
- Linux: 使用系统 Python（不需要嵌入式）

用法：
    python setup_embed_python.py [--version 3.11.9] [--output-dir install/python]
"""

import os
import sys
import platform
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

# 默认 Python 版本
DEFAULT_PYTHON_VERSION = "3.11.9"

# Python 嵌入式下载地址
PYTHON_EMBED_URLS = {
    "windows": {
        "x86_64": "https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip",
        "aarch64": "https://www.python.org/ftp/python/{version}/python-{version}-embed-arm64.zip",
    },
    "darwin": {
        # 使用 python-build-standalone
        "x86_64": "https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-{version}+20240415-x86_64-apple-darwin-install_only.tar.gz",
        "aarch64": "https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-{version}+20240415-aarch64-apple-darwin-install_only.tar.gz",
    },
}

# pip 下载地址
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def get_platform_info():
    """获取当前平台信息"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # 标准化架构名称
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        arch = machine
    
    return system, arch


def download_file(url: str, dest: Path, desc: str = ""):
    """下载文件"""
    print(f"[下载] {desc or url}")
    print(f"  -> {dest}")
    
    # 创建父目录
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # 下载
    urllib.request.urlretrieve(url, dest)
    print(f"  完成: {dest.stat().st_size / 1024 / 1024:.1f} MB")


def setup_windows_embed(version: str, output_dir: Path, arch: str):
    """设置 Windows 嵌入式 Python"""
    url_template = PYTHON_EMBED_URLS["windows"].get(arch)
    if not url_template:
        print(f"[错误] 不支持的 Windows 架构: {arch}")
        sys.exit(1)
    
    url = url_template.format(version=version)
    zip_path = output_dir.parent / f"python-{version}-embed.zip"
    
    # 下载
    download_file(url, zip_path, f"Python {version} embed for Windows {arch}")
    
    # 解压
    print(f"[解压] {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    
    # 删除 zip
    zip_path.unlink()
    
    # 修改 python*._pth 文件以启用 site-packages
    for pth_file in output_dir.glob("python*._pth"):
        print(f"[配置] 修改 {pth_file.name} 以启用 site-packages")
        content = pth_file.read_text()
        # 取消注释 import site
        content = content.replace("#import site", "import site")
        # 添加 deps 目录到路径
        if "../deps" not in content:
            content += "\n../deps\n"
        pth_file.write_text(content)
    
    # 安装 pip
    get_pip_path = output_dir / "get-pip.py"
    download_file(GET_PIP_URL, get_pip_path, "get-pip.py")
    
    python_exe = output_dir / "python.exe"
    print(f"[安装] pip")
    os.system(f'"{python_exe}" "{get_pip_path}" --no-warn-script-location')
    get_pip_path.unlink()
    
    print(f"[完成] Windows 嵌入式 Python 已设置到 {output_dir}")


def setup_macos_embed(version: str, output_dir: Path, arch: str):
    """设置 macOS 嵌入式 Python"""
    url_template = PYTHON_EMBED_URLS["darwin"].get(arch)
    if not url_template:
        print(f"[错误] 不支持的 macOS 架构: {arch}")
        sys.exit(1)
    
    url = url_template.format(version=version)
    tar_path = output_dir.parent / f"python-{version}-macos.tar.gz"
    
    # 下载
    download_file(url, tar_path, f"Python {version} standalone for macOS {arch}")
    
    # 解压
    print(f"[解压] {tar_path}")
    temp_dir = output_dir.parent / "python_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(temp_dir)
    
    # 移动到目标目录
    extracted_dir = temp_dir / "python"
    if extracted_dir.exists():
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(extracted_dir), str(output_dir))
    
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)
    tar_path.unlink()
    
    print(f"[完成] macOS 嵌入式 Python 已设置到 {output_dir}")


def setup_linux_embed(version: str, output_dir: Path, arch: str):
    """Linux 不需要嵌入式 Python，使用系统 Python"""
    print("[信息] Linux 平台使用系统 Python，跳过嵌入式 Python 设置")
    print("[信息] 依赖包将下载到 deps/ 目录，运行时通过 PYTHONPATH 加载")
    
    # 创建一个标记文件
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "USE_SYSTEM_PYTHON"
    marker.write_text("Linux 平台使用系统 Python\n请确保已安装 Python 3.10+")
    
    print(f"[完成] 已创建标记文件 {marker}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="设置嵌入式 Python 环境")
    parser.add_argument("--version", default=DEFAULT_PYTHON_VERSION, help="Python 版本")
    parser.add_argument("--output-dir", default="install/python", help="输出目录")
    parser.add_argument("--arch", default=None, help="目标架构 (x86_64 或 aarch64)")
    args = parser.parse_args()
    
    system, detected_arch = get_platform_info()
    arch = args.arch or detected_arch
    output_dir = Path(args.output_dir).resolve()
    
    print(f"[信息] 平台: {system}, 架构: {arch}")
    print(f"[信息] Python 版本: {args.version}")
    print(f"[信息] 输出目录: {output_dir}")
    
    if system == "windows":
        setup_windows_embed(args.version, output_dir, arch)
    elif system == "darwin":
        setup_macos_embed(args.version, output_dir, arch)
    elif system == "linux":
        setup_linux_embed(args.version, output_dir, arch)
    else:
        print(f"[错误] 不支持的平台: {system}")
        sys.exit(1)


if __name__ == "__main__":
    main()
