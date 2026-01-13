"""
截图采集器 - 用于收集 YOLOv8 训练数据集

功能：
- 在游戏运行时截取屏幕图像
- 按照 YOLOv8 数据集标准命名方式保存
- 支持自定义保存路径、前缀、间隔等参数

使用方法：
Pipeline 调用示例:
{
    "custom_action": "ScreenshotCollector",
    "custom_action_param": {
        "prefix": "battle",           // 可选，文件名前缀，默认 "img"
        "save_dir": "training/images", // 可选，保存目录，默认 "training/images"
        "format": "jpg",              // 可选，图片格式 jpg/png，默认 "jpg"
        "quality": 95                 // 可选，JPEG 质量 1-100，默认 95
    }
}
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from maa.custom_action import CustomAction
from maa.context import Context

# 获取项目根目录（agent 文件夹的父目录的父目录）
AGENT_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = AGENT_DIR.parent


class ScreenshotCollector(CustomAction):
    """
    截图采集器
    用于收集 YOLOv8 训练数据的原始图像
    """

    # 全局计数器，用于生成唯一文件名
    _counter = 0
    # 会话 ID，每次程序启动时更新
    _session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """
        执行截图采集

        参数说明 (通过 custom_action_param 传入):
        - prefix: 文件名前缀，默认 "img"
        - save_dir: 保存目录（相对于项目根目录），默认 "training/images"
        - format: 图片格式，支持 "jpg" 或 "png"，默认 "jpg"
        - quality: JPEG 质量，1-100，默认 95
        - include_timestamp: 是否在文件名中包含时间戳，默认 True
        """
        # 解析参数
        params = self._parse_params(argv.custom_action_param)
        prefix = params.get("prefix", "img")
        save_dir = params.get("save_dir", "training/images")
        img_format = params.get("format", "jpg").lower()
        quality = params.get("quality", 95)
        include_timestamp = params.get("include_timestamp", True)

        # 验证格式
        if img_format not in ("jpg", "png"):
            print(f"[ScreenshotCollector] 不支持的图片格式: {img_format}，使用默认 jpg")
            img_format = "jpg"

        # 构建保存路径
        save_path = PROJECT_ROOT / save_dir
        save_path.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        filename = self._generate_filename(prefix, img_format, include_timestamp)
        filepath = save_path / filename

        # 获取截图（使用 MaaFramework 的截图方式，会自动统一压缩到 1280x720）
        try:
            context.tasker.controller.post_screencap().wait()
            image = context.tasker.controller.cached_image
            
            if image is None:
                print("[ScreenshotCollector] 截图失败：未获取到图像")
                return CustomAction.RunResult(success=False)

            # 保存图像
            self._save_image(image, filepath, img_format, quality)
            
            print(f"[ScreenshotCollector] 截图已保存: {filepath}")
            ScreenshotCollector._counter += 1

            return CustomAction.RunResult(success=True)

        except Exception as e:
            print(f"[ScreenshotCollector] 截图异常: {e}")
            return CustomAction.RunResult(success=False)

    def _parse_params(self, param_str: str) -> dict:
        """解析自定义动作参数"""
        if not param_str:
            return {}
        try:
            return json.loads(param_str) if isinstance(param_str, str) else param_str
        except json.JSONDecodeError:
            print(f"[ScreenshotCollector] 参数解析失败: {param_str}")
            return {}

    def _generate_filename(self, prefix: str, ext: str, include_timestamp: bool) -> str:
        """
        生成 YOLOv8 数据集标准命名格式的文件名
        
        命名格式: {prefix}_{session}_{counter:06d}.{ext}
        或带时间戳: {prefix}_{timestamp}_{counter:06d}.{ext}
        
        例如: battle_20260110_143052_000001.jpg
        """
        counter_str = f"{ScreenshotCollector._counter:06d}"
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{prefix}_{timestamp}_{counter_str}.{ext}"
        else:
            return f"{prefix}_{self._session_id}_{counter_str}.{ext}"

    def _save_image(self, image, filepath: Path, img_format: str, quality: int):
        """
        保存图像到文件
        
        注意：MaaFramework 返回的 image 是 numpy.ndarray (BGR 格式)
        """
        import numpy as np
        
        # 检查是否为有效的 numpy 数组
        if not isinstance(image, np.ndarray):
            raise ValueError(f"无效的图像类型: {type(image)}")
        
        # 使用 OpenCV 保存图像
        import cv2
        
        if img_format == "jpg":
            # JPEG 质量参数
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        else:
            # PNG 压缩级别 (0-9，默认 3)
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        
        success = cv2.imwrite(str(filepath), image, params)
        if not success:
            raise IOError(f"保存图像失败: {filepath}")


class BatchScreenshotCollector(CustomAction):
    """
    批量截图采集器
    在指定时间间隔内连续采集多张截图
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """
        执行批量截图采集

        参数说明 (通过 custom_action_param 传入):
        - count: 截图数量，默认 5
        - interval: 截图间隔（毫秒），默认 500
        - prefix: 文件名前缀，默认 "batch"
        - save_dir: 保存目录，默认 "training/images"
        - format: 图片格式，默认 "jpg"
        - quality: JPEG 质量，默认 95
        """
        # 解析参数
        params = self._parse_params(argv.custom_action_param)
        count = params.get("count", 5)
        interval_ms = params.get("interval", 500)
        
        # 创建单次截图采集器的参数
        single_params = {
            "prefix": params.get("prefix", "batch"),
            "save_dir": params.get("save_dir", "training/images"),
            "format": params.get("format", "jpg"),
            "quality": params.get("quality", 95),
            "include_timestamp": True,
        }
        
        collector = ScreenshotCollector()
        success_count = 0
        
        print(f"[BatchScreenshotCollector] 开始批量截图: 数量={count}, 间隔={interval_ms}ms")
        
        for i in range(count):
            # 创建模拟的 argv
            mock_argv = type(argv)(
                task_detail=argv.task_detail,
                node_name=argv.node_name,
                custom_action_name=argv.custom_action_name,
                custom_action_param=json.dumps(single_params),
                reco_detail=argv.reco_detail,
                box=argv.box,
            )
            
            result = collector.run(context, mock_argv)
            if result.success:
                success_count += 1
            
            # 等待间隔（除了最后一次）
            if i < count - 1:
                time.sleep(interval_ms / 1000.0)
        
        print(f"[BatchScreenshotCollector] 批量截图完成: {success_count}/{count}")
        
        return CustomAction.RunResult(success=success_count > 0)

    def _parse_params(self, param_str: str) -> dict:
        """解析自定义动作参数"""
        if not param_str:
            return {}
        try:
            return json.loads(param_str) if isinstance(param_str, str) else param_str
        except json.JSONDecodeError:
            print(f"[BatchScreenshotCollector] 参数解析失败: {param_str}")
            return {}


class ConditionalScreenshotCollector(CustomAction):
    """
    条件截图采集器
    只在识别成功时保存截图，方便收集特定场景的数据
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """
        执行条件截图采集
        
        该动作会检查当前节点的识别结果，只有在识别成功时才保存截图
        适用于收集特定 UI 元素出现时的数据

        参数说明 (通过 custom_action_param 传入):
        - prefix: 文件名前缀，默认使用当前节点名称
        - save_dir: 保存目录，默认 "training/images"
        - format: 图片格式，默认 "jpg"
        - quality: JPEG 质量，默认 95
        - save_on_fail: 是否在识别失败时也保存，默认 False
        """
        # 解析参数
        params = self._parse_params(argv.custom_action_param)
        save_on_fail = params.get("save_on_fail", False)
        
        # 检查识别结果
        reco_detail = argv.reco_detail
        has_recognition = reco_detail is not None and reco_detail.hit
        
        if not has_recognition and not save_on_fail:
            print(f"[ConditionalScreenshotCollector] 跳过截图: 识别未命中")
            return CustomAction.RunResult(success=True)
        
        # 使用节点名称作为默认前缀
        default_prefix = argv.node_name.replace(" ", "_").replace("/", "_")
        
        # 创建单次截图采集器的参数
        single_params = {
            "prefix": params.get("prefix", default_prefix),
            "save_dir": params.get("save_dir", "training/images"),
            "format": params.get("format", "jpg"),
            "quality": params.get("quality", 95),
            "include_timestamp": True,
        }
        
        collector = ScreenshotCollector()
        mock_argv = type(argv)(
            task_detail=argv.task_detail,
            node_name=argv.node_name,
            custom_action_name=argv.custom_action_name,
            custom_action_param=json.dumps(single_params),
            reco_detail=argv.reco_detail,
            box=argv.box,
        )
        
        return collector.run(context, mock_argv)

    def _parse_params(self, param_str: str) -> dict:
        """解析自定义动作参数"""
        if not param_str:
            return {}
        try:
            return json.loads(param_str) if isinstance(param_str, str) else param_str
        except json.JSONDecodeError:
            print(f"[ConditionalScreenshotCollector] 参数解析失败: {param_str}")
            return {}
