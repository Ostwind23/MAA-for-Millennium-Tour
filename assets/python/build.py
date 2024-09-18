import PyInstaller.__main__
import os
import site
import shutil

# 递归搜索指定文件
def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# 获取当前工作目录
current_dir = os.getcwd()

# 搜索当前目录下的 __main__.py 文件
main_script_path = find_file('__main__.py', current_dir)

# 如果当前目录下未找到，则切换到 assets/python 目录继续搜索
if main_script_path is None:
    assets_python_path = os.path.join(current_dir, 'assets', 'python')
    main_script_path = find_file('__main__.py', assets_python_path)
    if main_script_path is None:
        raise FileNotFoundError("Can't find __main__.py through searching.")

# 获取 site-packages 目录列表
site_packages_paths = site.getsitepackages()

# 查找包含 maa/bin 的路径
maa_bin_path = None
for path in site_packages_paths:
    potential_path = os.path.join(path, 'maa', 'bin')
    if os.path.exists(potential_path):
        maa_bin_path = potential_path
        break

if maa_bin_path is None:
    raise FileNotFoundError("未找到包含 maa/bin 的路径")

# 构建 --add-data 参数
add_data_param = f'{maa_bin_path}{os.pathsep}maa/bin'

# 查找包含 MaaAgentBinary 的路径
maa_bin_path2 = None
for path in site_packages_paths:
    potential_path = os.path.join(path, 'MaaAgentBinary')
    if os.path.exists(potential_path):
        maa_bin_path2 = potential_path
        break

if maa_bin_path2 is None:
    raise FileNotFoundError("未找到包含 MaaAgentBinary 的路径")

# 构建 --add-data 参数
add_data_param2 = f'{maa_bin_path2}{os.pathsep}MaaAgentBinary'

print("Current working directory:", os.getcwd())
print(f"Found __main__.py at: {main_script_path}")
print(f"Found maa/bin at: {maa_bin_path}")
print(f"Found MaaAgentBinary at: {maa_bin_path2}")

# 运行 PyInstaller
PyInstaller.__main__.run([
    main_script_path,
    '--onefile',
    '--name=Autofishing',
    f'--add-data={maa_bin_path}{os.pathsep}maa/bin',
    f'--add-data={maa_bin_path2}{os.pathsep}MaaAgentBinary',
    '--clean',
])

current_dir = os.getcwd()
# 遍历current_dir目录，寻找Autofishing.exe文件
for root, dirs, files in os.walk(current_dir):
    if 'Autofishing.exe' in files:
        src = os.path.join(root, 'Autofishing.exe')
        break
    else:
        print("src Autofishing.exe not found")

gui_flag = 0
for root, dirs, _ in os.walk(current_dir, topdown=True):
    if 'gui' in dirs:
        gui_flag = 1
        break

if src and gui_flag == 1:
    dst = os.path.join(current_dir, 'gui', 'Autofishing.exe')
    print(f"Found GUI.")
elif src:
    dst = os.path.join(current_dir, 'Autofishing.exe')
    print(f"Didn't find GUI.")
else:
    print("Autofishing.exe not found")

# 检查目标文件是否存在并删除
if os.path.exists(dst):
    os.remove(dst)

shutil.move(src, dst)
print(f"[Autofishing.exe] moved from {src} to {dst}")
