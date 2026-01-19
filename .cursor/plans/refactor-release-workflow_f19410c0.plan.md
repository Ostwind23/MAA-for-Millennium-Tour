---
name: refactor-release-workflow
overview: 将你的打包发版流程重构为示范项目风格：矩阵构建 + CI 脚本拆分，同时固定 MaaFramework 与 MFAAvalonia 版本，并移除 MirrorChyan 相关步骤。
todos: []
---

# 打包发版流程重构计划

## 目标

- 采用示范项目式的 matrix 构建与 CI 脚本组织方式
- 固定版本：MaaFramework `v5.3.3`，MFAAvalonia `v2.5.6`
- 不包含任何 MirrorChyan 相关步骤
- 扩展支持 `win/linux/macos` 的 `x86_64` 与 `arm64`

## 涉及文件

- [.github/workflows/install.yml](H:/MAA-for-Millennium-Tour/MAA_for_Millennium_Tour/.github/workflows/install.yml)
- [tools/install.py](H:/MAA-for-Millennium-Tour/MAA_for_Millennium_Tour/tools/install.py)
- [tools/ci/install.py](H:/MAA-for-Millennium-Tour/MAA_for_Millennium_Tour/tools/ci/install.py)（新增）
- [tools/ci/utils.py](H:/MAA-for-Millennium-Tour/MAA_for_Millennium_Tour/tools/ci/utils.py)（新增）
- [tools/ci/download_deps.py](H:/MAA-for-Millennium-Tour/MAA_for_Millennium_Tour/tools/ci/download_deps.py)（新增或调整）
- [tools/ci/setup_embed_python.py](H:/MAA-for-Millennium-Tour/MAA_for_Millennium_Tour/tools/ci/setup_embed_python.py)（新增或调整）
- [agent/requirements.txt](H:/MAA-for-Millennium-Tour/MAA_for_Millennium_Tour/agent/requirements.txt)（需要迁移或复用）

## 计划步骤

1. 迁移/新增 CI 脚本到 `tools/ci/`

- 将现有 `tools/install.py` 中的安装逻辑拆分成 CI 专用版本（参考示范项目的 `tools/ci/install.py`）
- 新增 `tools/ci/utils.py`，集中管理 `working_dir` 等基础路径
- 新增/迁移依赖下载脚本 `tools/ci/download_deps.py`，支持跨平台 wheel 下载
- 新增/迁移 `tools/ci/setup_embed_python.py`，用于 Windows/macOS 的完整 Python

1. 改造 `install.yml` 为 matrix 构建

- 用 `install` job + matrix 覆盖 `win/linux/macos` 的 `x86_64` 与 `arm64`
- 固定版本：`MAA_FRAMEWORK_VERSION=v5.3.3`，`MFAA_VERSION=v2.5.6`
- 将当前分散在 job 内的步骤统一为：下载 MFAA / 下载 MAA / 设置完整 Python / 运行 CI 安装脚本 / 下载依赖 / 上传 artifact
- 保留 `meta`、`changelog`、`release` 三段逻辑，但注释掉 MirrorChyan 相关步骤

1. 对齐 agent 运行时配置

- 在 CI 安装脚本中统一写入 `interface.json` 的 `agent.child_exec` 与 `child_args`
- 保持 Windows/macOS 使用完整 Python，Linux 使用系统/虚拟环境 Python（与示范项目一致）

1. 校验与整理

- 检查 CI 生成的 `install/` 目录结构是否与现有使用方式兼容
- 确认 `agent/requirements.txt` 与 CI 下载依赖的路径一致
- 如需要，补充 README/文档中对新流程的说明

## 验证建议

- 本地不执行（GitHub Actions 环境构建）
- 触发一次 `workflow_dispatch` 验证 matrix 构建是否全平台通过
- 检查产物内 `interface.json` 中的 `agent` 配置和版本号是否正确

## 待办事项

- [ ] 拆分并新增 CI 脚本（`tools/ci/*`）
- [ ] 迁移工作流到 matrix，并固定版本号
- [ ] 移除 MirrorChyan 相关步骤
- [ ] 校验产物结构与依赖路径
- [ ] 更新必要的文档说明（如需要）
