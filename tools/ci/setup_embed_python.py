#!/usr/bin/env python3
"""
下载并设置嵌入式 Python 环境

支持平台：
- Windows: 使用官方 embed 版本
- macOS: 使用 python-build-standalone
- Linux: 使用系统 Python（不需要嵌入式）

用法：
    python setup_embed_python.py <操作系统> <架构> [--version 3.11.9] [--output-dir install/python]
"""

import os
import sys
import shutil
import subprocess
import urllib.request
import urllib.error
import zipfile
import tarfile
import stat

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
except AttributeError:
    pass


PYTHON_VERSION_TARGET = os.getenv("PYTHON_EMBED_VERSION", "3.12.4")
PYTHON_BUILD_STANDALONE_RELEASE_TAG = os.getenv(
    "PYTHON_BUILD_STANDALONE_RELEASE_TAG", "20240507"
)


def download_file(url, dest_path):
    print(f"正在下载: {url}")
    print(f"到: {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    print("下载完成。")


def extract_zip(zip_path, dest_dir):
    print(f"正在解压 ZIP: {zip_path} 到 {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    print("ZIP 解压完成。")


def extract_tar(tar_path, dest_dir):
    print(f"正在解压 TAR: {tar_path} 到 {dest_dir}")
    with tarfile.open(tar_path, "r:*") as tar_ref:
        tar_ref.extractall(path=dest_dir)
    print("TAR 解压完成。")


def get_python_executable_path(base_dir, os_type):
    if os_type == "win":
        return os.path.join(base_dir, "python.exe")
    if os_type == "macos":
        py3_path = os.path.join(base_dir, "bin", "python3")
        py_path = os.path.join(base_dir, "bin", "python")
        if os.path.exists(py3_path):
            return py3_path
        if os.path.exists(py_path):
            return py_path
    return None


def ensure_pip(python_executable, python_install_dir):
    if not python_executable or not os.path.exists(python_executable):
        print("错误: Python 可执行文件未找到，无法安装 pip。")
        return False

    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    get_pip_script_path = os.path.join(python_install_dir, "get-pip.py")

    print(f"正在下载 get-pip.py 从 {get_pip_url}")
    download_file(get_pip_url, get_pip_script_path)

    print("正在使用 get-pip.py 安装 pip...")
    try:
        subprocess.run([python_executable, get_pip_script_path], check=True)
        print("pip 安装成功。")
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"pip 安装失败: {e}")
        return False
    finally:
        if os.path.exists(get_pip_script_path):
            os.remove(get_pip_script_path)


def setup_windows_embed(version, output_dir, arch):
    arch_mapping = {
        "AMD64": "amd64",
        "x86_64": "amd64",
        "ARM64": "arm64",
        "aarch64": "arm64",
    }
    win_arch_suffix = arch_mapping.get(arch, arch.lower())

    if win_arch_suffix not in ["amd64", "arm64"]:
        print(f"错误: 不支持的 Windows 架构: {arch} -> {win_arch_suffix}")
        return

    download_url = (
        f"https://www.python.org/ftp/python/{version}/"
        f"python-{version}-embed-{win_arch_suffix}.zip"
    )
    zip_filename = f"python-{version}-embed-{win_arch_suffix}.zip"
    zip_filepath = os.path.join(output_dir, zip_filename)

    download_file(download_url, zip_filepath)
    extract_zip(zip_filepath, output_dir)
    os.remove(zip_filepath)

    version_nodots = version.replace(".", "")[:3]
    pth_filename_pattern = f"python{version_nodots}._pth"
    pth_file_path = os.path.join(output_dir, pth_filename_pattern)
    if not os.path.exists(pth_file_path):
        found_pth_files = [
            f
            for f in os.listdir(output_dir)
            if f.startswith("python") and f.endswith("._pth")
        ]
        if found_pth_files:
            pth_file_path = os.path.join(output_dir, found_pth_files[0])
        else:
            print(f"错误: 未在 {output_dir} 中找到 ._pth 文件。")
            return

    print(f"正在修改 ._pth 文件: {pth_file_path}")
    with open(pth_file_path, "r+", encoding="utf-8") as f:
        content = f.read()
        content = content.replace("#import site", "import site")
        content = content.replace("# import site", "import site")

        required_paths = [".", "Lib", "Lib\\site-packages", "DLLs"]
        for p_path in required_paths:
            if p_path not in content.splitlines():
                content += f"\n{p_path}"
        f.seek(0)
        f.write(content)
        f.truncate()
    print("._pth 文件修改完成。")

    python_executable_final_path = get_python_executable_path(output_dir, "win")
    if ensure_pip(python_executable_final_path, output_dir):
        print("嵌入式 Python 环境安装和 pip 配置完成。")
    else:
        print("嵌入式 Python 环境安装完成，但 pip 配置失败。")


def setup_macos_embed(version, output_dir, arch):
    arch_mapping = {"x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}
    pbs_arch = arch_mapping.get(arch, arch)

    if pbs_arch not in ["x86_64", "aarch64"]:
        print(f"错误: 不支持的 macOS 架构: {arch} -> {pbs_arch}")
        return

    candidate_tags = [PYTHON_BUILD_STANDALONE_RELEASE_TAG, "20240415"]
    candidate_tags = list(dict.fromkeys(candidate_tags))
    tar_filepath = None

    for tag in candidate_tags:
        pbs_filename = (
            f"cpython-{version}+{tag}-{pbs_arch}-apple-darwin-install_only.tar.gz"
        )
        download_url = (
            "https://github.com/indygreg/python-build-standalone/releases/download/"
            f"{tag}/{pbs_filename}"
        )
        tar_filepath = os.path.join(output_dir, pbs_filename)
        try:
            download_file(download_url, tar_filepath)
            break
        except urllib.error.HTTPError as err:
            if err.code == 404:
                print(f"未找到构建包（tag={tag}），尝试下一个标签。")
                continue
            raise
    else:
        print("错误: 无法找到匹配的 python-build-standalone 构建包。")
        return
    temp_extract_dir = os.path.join(output_dir, "_temp_extract")
    os.makedirs(temp_extract_dir, exist_ok=True)
    extract_tar(tar_filepath, temp_extract_dir)

    extracted_python_root = os.path.join(temp_extract_dir, "python")
    if os.path.isdir(extracted_python_root):
        for item_name in os.listdir(extracted_python_root):
            src = os.path.join(extracted_python_root, item_name)
            dst = os.path.join(output_dir, item_name)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(temp_extract_dir)
    else:
        print(f"错误: 解压后未找到 'python' 子目录于 {temp_extract_dir}")
        shutil.rmtree(temp_extract_dir)
        return

    if os.path.exists(tar_filepath):
        os.remove(tar_filepath)

    bin_dir = os.path.join(output_dir, "bin")
    if os.path.isdir(bin_dir):
        for item_name in os.listdir(bin_dir):
            item_path = os.path.join(bin_dir, item_name)
            if os.path.isfile(item_path) and not os.access(item_path, os.X_OK):
                current_mode = os.stat(item_path).st_mode
                os.chmod(
                    item_path,
                    current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
                )

    python_executable_final_path = get_python_executable_path(output_dir, "macos")
    if ensure_pip(python_executable_final_path, output_dir):
        print("嵌入式 Python 环境安装和 pip 配置完成。")
    else:
        print("嵌入式 Python 环境安装完成，但 pip 配置失败。")


def setup_linux_embed(output_dir):
    print("Linux 平台使用系统 Python，跳过嵌入式 Python 设置")
    os.makedirs(output_dir, exist_ok=True)
    marker = os.path.join(output_dir, "USE_SYSTEM_PYTHON")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("Linux 平台使用系统 Python\n请确保已安装 Python 3.10+\n")
    print(f"已创建标记文件 {marker}")


def main():
    if len(sys.argv) < 3:
        print("用法: python setup_embed_python.py <操作系统> <架构>")
        print("示例: python setup_embed_python.py win x86_64")
        sys.exit(1)

    os_name = sys.argv[1]
    arch = sys.argv[2]

    version = PYTHON_VERSION_TARGET
    output_dir = os.path.join("install", "python")
    if "--version" in sys.argv:
        version = sys.argv[sys.argv.index("--version") + 1]
    if "--output-dir" in sys.argv:
        output_dir = sys.argv[sys.argv.index("--output-dir") + 1]

    print(f"操作系统: {os_name}, 架构: {arch}")
    print(f"目标 Python 版本: {version}")
    print(f"目标安装目录: {output_dir}")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"目标目录 {output_dir} 已存在，清理后重新安装。")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    if os_name == "win":
        setup_windows_embed(version, output_dir, arch)
    elif os_name == "macos":
        setup_macos_embed(version, output_dir, arch)
    elif os_name == "linux":
        setup_linux_embed(output_dir)
    else:
        print(f"错误: 不支持的操作系统: {os_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
