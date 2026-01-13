"""
测试 MaaFramework NeuralNetworkDetect 识别功能

专门用于测试 run_recognition 调用 YOLO 模型的功能
包含详细的日志输出和可视化结果保存
"""

import os
import time
import json
import traceback
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2

from maa.context import Context
from maa.custom_action import CustomAction

# ==================== 配置常量 ====================

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets", "resource")
_MODEL_DIR = os.path.join(_ASSETS_DIR, "model")
_DEBUG_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "assets", "debug", "test_run_reco")

# YOLO 模型配置
FARMING_MODEL_PATH = "farming.onnx"
YOLO_LABELS = ["bugs", "girl"]
YOLO_THRESHOLD = 0.3
YOLO_NMS_THRESHOLD = 0.45

# 屏幕尺寸
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# 调试选项
SAVE_DEBUG_IMAGES = True
VERBOSE_LOGGING = True

# ==================== 辅助函数 ====================

def _parse_param(param: str) -> dict:
    """解析 JSON 参数字符串"""
    try:
        if isinstance(param, dict):
            return param
        return json.loads(param) if param else {}
    except json.JSONDecodeError as e:
        print(f"[TestRunReco] 参数解析失败: {e}")
        return {}


def _reco_hit(reco_result) -> bool:
    """检查识别是否命中"""
    if reco_result is None:
        return False
    return getattr(reco_result, 'hit', False) or getattr(reco_result, 'success', False)


def _reco_box(reco_result) -> list:
    """获取识别结果的边界框"""
    if not _reco_hit(reco_result):
        return None
    if hasattr(reco_result, 'best_result') and reco_result.best_result:
        if hasattr(reco_result.best_result, 'box'):
            return reco_result.best_result.box
    if hasattr(reco_result, 'box'):
        return reco_result.box
    return None


def _box_center(box: list) -> tuple:
    """计算边界框的中心点"""
    if box is None or len(box) != 4:
        return None
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


def _draw_detection(image: np.ndarray, box: list, label: str, confidence: float, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    参数:
        image: BGR 图像
        box: [x, y, w, h]
        label: 类别标签
        confidence: 置信度
        color: BGR 颜色
        
    返回:
        绘制后的图像
    """
    if box is None or len(box) != 4:
        return image
    
    x, y, w, h = box
    
    # 绘制边界框
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # 绘制标签和置信度
    text = f"{label}: {confidence:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 计算文本背景框
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 绘制文本背景
    cv2.rectangle(image, (x, y - text_h - baseline - 5), (x + text_w, y), color, -1)
    
    # 绘制文本
    cv2.putText(image, text, (x, y - baseline - 5), font, font_scale, (255, 255, 255), thickness)
    
    # 绘制中心点
    center = _box_center(box)
    if center:
        cv2.circle(image, center, 5, color, -1)
    
    return image


# ==================== Custom Action ====================

class TestRunRecoHandler(CustomAction):
    """
    测试 MaaFramework run_recognition 功能
    
    参数:
        test_mode: 测试模式
            - "single_class" - 单类别检测（默认检测 girl）
            - "all_classes" - 检测所有类别
            - "no_labels" - 不设置 labels 参数（让 MaaFramework 自动读取）
        target_class: 目标类别索引（仅在 single_class 模式下使用），默认 1 (girl)
        save_result: 是否保存可视化结果，默认 True
    
    Pipeline 调用示例:
    {
        "custom_action": "TestRunRecoHandler",
        "custom_action_param": {
            "test_mode": "all_classes",
            "save_result": true
        }
    }
    """
    
    def __init__(self):
        super().__init__()
        
        # 确保调试输出目录存在
        if SAVE_DEBUG_IMAGES:
            os.makedirs(_DEBUG_OUTPUT_DIR, exist_ok=True)
            print(f"[TestRunReco] 调试输出目录: {_DEBUG_OUTPUT_DIR}")
    
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """执行测试"""
        
        print("\n" + "=" * 80)
        print("[TestRunReco] 开始测试 MaaFramework NeuralNetworkDetect")
        print("=" * 80)
        
        # 解析参数
        params = _parse_param(argv.custom_action_param)
        test_mode = params.get("test_mode", "all_classes")
        target_class = params.get("target_class", 1)  # 默认检测 girl
        save_result = params.get("save_result", True)
        
        print(f"[TestRunReco] 测试模式: {test_mode}")
        print(f"[TestRunReco] 目标类别: {target_class} ({YOLO_LABELS[target_class] if target_class < len(YOLO_LABELS) else 'unknown'})")
        print(f"[TestRunReco] 保存结果: {save_result}")
        print(f"[TestRunReco] 模型路径: {FARMING_MODEL_PATH}")
        print(f"[TestRunReco] 标签列表: {YOLO_LABELS}")
        print(f"[TestRunReco] 检测阈值: {YOLO_THRESHOLD}")
        
        try:
            # 步骤 1: 获取屏幕截图
            print("\n[TestRunReco] 步骤 1: 获取屏幕截图...")
            screenshot = self._capture_screenshot(context)
            
            if screenshot is None:
                print("[TestRunReco] ✗ 截图失败")
                return CustomAction.RunResult(success=False)
            
            print(f"[TestRunReco] ✓ 截图成功，尺寸: {screenshot.shape}")
            
            # 保存原始截图
            if save_result:
                screenshot_path = os.path.join(_DEBUG_OUTPUT_DIR, "01_screenshot.png")
                cv2.imwrite(screenshot_path, screenshot)
                print(f"[TestRunReco] 原始截图已保存: {screenshot_path}")
            
            # 步骤 2: 执行识别
            print("\n[TestRunReco] 步骤 2: 执行识别...")
            
            if test_mode == "no_labels":
                results = self._test_no_labels_mode(context, screenshot)
            elif test_mode == "single_class":
                results = self._test_single_class_mode(context, screenshot, target_class)
            elif test_mode == "all_classes":
                results = self._test_all_classes_mode(context, screenshot)
            else:
                print(f"[TestRunReco] ✗ 未知测试模式: {test_mode}")
                return CustomAction.RunResult(success=False)
            
            # 步骤 3: 可视化结果
            print("\n[TestRunReco] 步骤 3: 可视化结果...")
            if save_result:
                self._visualize_results(screenshot, results, test_mode)
            
            # 步骤 4: 统计结果
            print("\n[TestRunReco] 步骤 4: 统计结果...")
            self._print_statistics(results)
            
            print("\n" + "=" * 80)
            print("[TestRunReco] 测试完成")
            print("=" * 80)
            
            return CustomAction.RunResult(success=True)
            
        except Exception as e:
            print(f"\n[TestRunReco] ✗ 发生异常: {e}")
            print("[TestRunReco] 异常堆栈:")
            traceback.print_exc()
            return CustomAction.RunResult(success=False)
    
    def _parse_model_metadata(self, model_path: str):
        """
        解析 ONNX 模型的 metadata
        
        参数:
            model_path: 模型文件路径（相对于 model 目录）
        """
        print(f"\n[TestRunReco] === 解析模型 Metadata ===")
        print(f"[TestRunReco] 模型路径: {model_path}")
        
        try:
            import onnx
            
            # 构建完整路径
            full_model_path = os.path.join(_MODEL_DIR, model_path)
            print(f"[TestRunReco] 完整路径: {full_model_path}")
            
            if not os.path.exists(full_model_path):
                print(f"[TestRunReco] ✗ 模型文件不存在: {full_model_path}")
                return
            
            # 加载模型
            print(f"[TestRunReco] 正在加载模型...")
            model = onnx.load(full_model_path)
            
            # 读取 metadata
            metadata = model.metadata_props
            print(f"[TestRunReco] ✓ Metadata 数量: {len(metadata)}")
            
            if len(metadata) == 0:
                print(f"[TestRunReco] ! 模型没有 metadata")
            else:
                print(f"[TestRunReco] Metadata 内容:")
                for prop in metadata:
                    print(f"  - {prop.key}: {prop.value}")
            
            # 输出模型基本信息
            print(f"[TestRunReco] 模型基本信息:")
            print(f"  - IR 版本: {model.ir_version}")
            print(f"  - Producer: {model.producer_name} {model.producer_version}")
            print(f"  - 输入数: {len(model.graph.input)}")
            print(f"  - 输出数: {len(model.graph.output)}")
            
        except ImportError:
            print(f"[TestRunReco] ! onnx 库未安装，无法解析 metadata")
            print(f"[TestRunReco] ! 提示: pip install onnx")
        except Exception as e:
            print(f"[TestRunReco] ✗ 解析 metadata 异常: {e}")
            traceback.print_exc()
    
    def _capture_screenshot(self, context: Context) -> np.ndarray:
        """
        获取屏幕截图
        
        返回:
            BGR 格式的图像，失败返回 None
        """
        try:
            controller = context.tasker.controller
            print("[TestRunReco] 正在截图...")
            controller.post_screencap().wait()
            time.sleep(0.1)  # 等待截图完成
            
            image = controller.cached_image
            
            if image is None:
                print("[TestRunReco] ✗ cached_image 为 None")
                return None
            
            print(f"[TestRunReco] ✓ 截图完成: {image.shape}, dtype={image.dtype}")
            return image
            
        except Exception as e:
            print(f"[TestRunReco] ✗ 截图异常: {e}")
            traceback.print_exc()
            return None
    
    def _test_no_labels_mode(self, context: Context, image: np.ndarray) -> List[Dict]:
        """
        测试不设置 labels 参数的模式（让 MaaFramework 自动从模型 metadata 读取）
        
        返回:
            检测结果列表
        """
        print("\n[TestRunReco] === 测试模式: no_labels ===")
        print("[TestRunReco] 不设置 labels 参数，让 MaaFramework 自动读取模型 metadata")
        
        results = []
        
        try:
            # 解析模型 metadata
            self._parse_model_metadata(FARMING_MODEL_PATH)
            
            # 构建识别参数
            task_name = "TestReco_NoLabels"
            pipeline_override = {
                task_name: {
                    "recognition": "NeuralNetworkDetect",
                    "model": FARMING_MODEL_PATH,
                    # 不设置 labels
                    "threshold": YOLO_THRESHOLD,
                    "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
                }
            }
            
            print(f"[TestRunReco] Pipeline 配置:")
            print(json.dumps(pipeline_override, indent=2, ensure_ascii=False))
            
            # 打印 run_recognition 参数
            print(f"\n[TestRunReco] === 调用 run_recognition 参数 ===")
            print(f"[TestRunReco] 参数 1 - task_name (str): {task_name}")
            print(f"[TestRunReco] 参数 2 - image (np.ndarray): shape={image.shape}, dtype={image.dtype}")
            print(f"[TestRunReco] 参数 3 - pipeline_override (dict):")
            print(json.dumps(pipeline_override, indent=2, ensure_ascii=False))
            
            # 执行识别
            print(f"\n[TestRunReco] 调用 context.run_recognition...")
            reco_result = context.run_recognition(task_name, image, pipeline_override)
            
            # 解析结果
            print(f"[TestRunReco] 识别完成，解析结果...")
            results = self._parse_reco_result(reco_result, "auto")
            
        except Exception as e:
            print(f"[TestRunReco] ✗ no_labels 模式异常: {e}")
            traceback.print_exc()
        
        return results
    
    def _test_single_class_mode(self, context: Context, image: np.ndarray, target_class: int) -> List[Dict]:
        """
        测试单类别检测模式
        
        返回:
            检测结果列表
        """
        class_name = YOLO_LABELS[target_class] if target_class < len(YOLO_LABELS) else f"class_{target_class}"
        
        print(f"\n[TestRunReco] === 测试模式: single_class (目标: {class_name}) ===")
        
        results = []
        
        try:
            # 解析模型 metadata
            self._parse_model_metadata(FARMING_MODEL_PATH)
            
            # 构建识别参数
            task_name = f"TestReco_Single_{class_name}"
            pipeline_override = {
                task_name: {
                    "recognition": "NeuralNetworkDetect",
                    "model": FARMING_MODEL_PATH,
                    "labels": YOLO_LABELS,
                    "expected": [target_class],  # 只检测指定类别
                    "threshold": YOLO_THRESHOLD,
                    "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
                }
            }
            
            print(f"[TestRunReco] Pipeline 配置:")
            print(json.dumps(pipeline_override, indent=2, ensure_ascii=False))
            
            # 打印 run_recognition 参数
            print(f"\n[TestRunReco] === 调用 run_recognition 参数 ===")
            print(f"[TestRunReco] 参数 1 - task_name (str): {task_name}")
            print(f"[TestRunReco] 参数 2 - image (np.ndarray): shape={image.shape}, dtype={image.dtype}")
            print(f"[TestRunReco] 参数 3 - pipeline_override (dict):")
            print(json.dumps(pipeline_override, indent=2, ensure_ascii=False))
            
            # 执行识别
            print(f"\n[TestRunReco] 调用 context.run_recognition...")
            reco_result = context.run_recognition(task_name, image, pipeline_override)
            
            # 解析结果
            print(f"[TestRunReco] 识别完成，解析结果...")
            results = self._parse_reco_result(reco_result, class_name)
            
        except Exception as e:
            print(f"[TestRunReco] ✗ single_class 模式异常: {e}")
            traceback.print_exc()
        
        return results
    
    def _test_all_classes_mode(self, context: Context, image: np.ndarray) -> List[Dict]:
        """
        测试检测所有类别模式
        
        返回:
            检测结果列表
        """
        print(f"\n[TestRunReco] === 测试模式: all_classes ===")
        
        # 解析模型 metadata（只需解析一次）
        self._parse_model_metadata(FARMING_MODEL_PATH)
        
        all_results = []
        
        # 对每个类别分别进行检测
        for class_idx, class_name in enumerate(YOLO_LABELS):
            print(f"\n[TestRunReco] --- 检测类别 {class_idx}: {class_name} ---")
            
            try:
                # 构建识别参数
                task_name = f"TestReco_All_{class_name}"
                pipeline_override = {
                    task_name: {
                        "recognition": "NeuralNetworkDetect",
                        "model": FARMING_MODEL_PATH,
                        "labels": YOLO_LABELS,
                        "expected": [class_idx],
                        "threshold": YOLO_THRESHOLD,
                        "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
                    }
                }
                
                print(f"[TestRunReco] Pipeline 配置:")
                print(json.dumps(pipeline_override, indent=2, ensure_ascii=False))
                
                # 打印 run_recognition 参数
                print(f"\n[TestRunReco] === 调用 run_recognition 参数 ===")
                print(f"[TestRunReco] 参数 1 - task_name (str): {task_name}")
                print(f"[TestRunReco] 参数 2 - image (np.ndarray): shape={image.shape}, dtype={image.dtype}")
                print(f"[TestRunReco] 参数 3 - pipeline_override (dict):")
                print(json.dumps(pipeline_override, indent=2, ensure_ascii=False))
                
                # 执行识别
                print(f"\n[TestRunReco] 调用 context.run_recognition...")
                reco_result = context.run_recognition(task_name, image, pipeline_override)
                
                # 解析结果
                print(f"[TestRunReco] 识别完成，解析结果...")
                results = self._parse_reco_result(reco_result, class_name)
                
                # 合并结果
                all_results.extend(results)
                
            except Exception as e:
                print(f"[TestRunReco] ✗ 检测类别 {class_name} 异常: {e}")
                traceback.print_exc()
        
        return all_results
    
    def _parse_reco_result(self, reco_result, expected_label: str) -> List[Dict]:
        """
        解析识别结果，提取所有检测信息
        
        参数:
            reco_result: MaaFramework 识别结果
            expected_label: 期望的类别标签
            
        返回:
            检测结果列表 [{"label": str, "box": [x,y,w,h], "confidence": float, "class_id": int}, ...]
        """
        print(f"\n[TestRunReco] === 解析识别结果 (期望: {expected_label}) ===")
        
        results = []
        
        # 检查结果是否为 None
        if reco_result is None:
            print("[TestRunReco] ✗ reco_result 为 None")
            return results
        
        # 打印结果对象信息
        print(f"[TestRunReco] reco_result 类型: {type(reco_result)}")
        print(f"[TestRunReco] reco_result 属性: {dir(reco_result)}")
        
        # 检查 hit 状态
        hit = _reco_hit(reco_result)
        print(f"[TestRunReco] hit 状态: {hit}")
        
        if not hit:
            print("[TestRunReco] ✗ 识别未命中（hit=False）")
            return results
        
        # 尝试获取 best_result
        if hasattr(reco_result, 'best_result') and reco_result.best_result:
            print(f"[TestRunReco] best_result 存在")
            print(f"[TestRunReco] best_result 类型: {type(reco_result.best_result)}")
            print(f"[TestRunReco] best_result 属性: {dir(reco_result.best_result)}")
            
            best = reco_result.best_result
            
            # 提取信息
            box = getattr(best, 'box', None)
            label = getattr(best, 'label', expected_label)
            cls_index = getattr(best, 'cls_index', -1)
            confidence = getattr(best, 'score', 0.0)
            
            print(f"[TestRunReco] best_result 信息:")
            print(f"  - box: {box}")
            print(f"  - label: {label}")
            print(f"  - cls_index: {cls_index}")
            print(f"  - confidence: {confidence}")
            
            if box is not None:
                results.append({
                    "label": label,
                    "box": list(box),
                    "confidence": float(confidence),
                    "class_id": int(cls_index)
                })
        
        # 尝试获取 all_results
        if hasattr(reco_result, 'all_results') and reco_result.all_results:
            print(f"[TestRunReco] all_results 存在，数量: {len(reco_result.all_results)}")
            
            for i, result in enumerate(reco_result.all_results):
                print(f"\n[TestRunReco] all_results[{i}]:")
                print(f"  类型: {type(result)}")
                print(f"  属性: {dir(result)}")
                
                box = getattr(result, 'box', None)
                label = getattr(result, 'label', expected_label)
                cls_index = getattr(result, 'cls_index', -1)
                confidence = getattr(result, 'score', 0.0)
                
                print(f"  - box: {box}")
                print(f"  - label: {label}")
                print(f"  - cls_index: {cls_index}")
                print(f"  - confidence: {confidence}")
                
                if box is not None:
                    # 检查是否已经在 results 中（避免重复）
                    duplicate = False
                    for existing in results:
                        if existing["box"] == list(box):
                            duplicate = True
                            break
                    
                    if not duplicate:
                        results.append({
                            "label": label,
                            "box": list(box),
                            "confidence": float(confidence),
                            "class_id": int(cls_index)
                        })
        
        print(f"\n[TestRunReco] 最终提取到 {len(results)} 个检测结果")
        
        return results
    
    def _visualize_results(self, screenshot: np.ndarray, results: List[Dict], test_mode: str):
        """
        可视化检测结果并保存图像
        
        参数:
            screenshot: 原始截图
            results: 检测结果列表
            test_mode: 测试模式
        """
        print(f"\n[TestRunReco] 绘制 {len(results)} 个检测结果...")
        
        # 复制图像用于绘制
        vis_image = screenshot.copy()
        
        # 为不同类别分配不同颜色
        colors = {
            "bugs": (0, 0, 255),    # 红色
            "girl": (0, 255, 0),    # 绿色
            "auto": (255, 0, 255),  # 紫色
        }
        
        # 绘制每个检测结果
        for i, det in enumerate(results):
            label = det["label"]
            box = det["box"]
            confidence = det["confidence"]
            
            # 选择颜色
            color = colors.get(label, (255, 255, 0))  # 默认青色
            
            print(f"[TestRunReco] 绘制检测[{i}]: {label} @ {box}, conf={confidence:.3f}")
            
            vis_image = _draw_detection(vis_image, box, label, confidence, color)
        
        # 添加统计信息文本
        info_text = [
            f"Test Mode: {test_mode}",
            f"Detections: {len(results)}",
            f"Model: {FARMING_MODEL_PATH}",
            f"Threshold: {YOLO_THRESHOLD}",
        ]
        
        y_offset = 30
        for line in info_text:
            cv2.putText(vis_image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 0), 1)
            y_offset += 30
        
        # 保存可视化结果
        result_path = os.path.join(_DEBUG_OUTPUT_DIR, f"02_result_{test_mode}.png")
        cv2.imwrite(result_path, vis_image)
        print(f"[TestRunReco] ✓ 可视化结果已保存: {result_path}")
        
        # 同时保存带编号的详细信息
        self._save_detection_details(results, test_mode)
    
    def _save_detection_details(self, results: List[Dict], test_mode: str):
        """
        保存检测结果的详细信息到 JSON 文件
        
        参数:
            results: 检测结果列表
            test_mode: 测试模式
        """
        details = {
            "test_mode": test_mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": FARMING_MODEL_PATH,
            "threshold": YOLO_THRESHOLD,
            "total_detections": len(results),
            "detections": results
        }
        
        details_path = os.path.join(_DEBUG_OUTPUT_DIR, f"03_details_{test_mode}.json")
        
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        
        print(f"[TestRunReco] ✓ 检测详情已保存: {details_path}")
    
    def _print_statistics(self, results: List[Dict]):
        """
        打印检测结果统计信息
        
        参数:
            results: 检测结果列表
        """
        print(f"\n[TestRunReco] === 检测结果统计 ===")
        print(f"[TestRunReco] 总检测数: {len(results)}")
        
        if len(results) == 0:
            print("[TestRunReco] 未检测到任何对象")
            return
        
        # 按类别统计
        class_counts = {}
        class_confidences = {}
        
        for det in results:
            label = det["label"]
            confidence = det["confidence"]
            
            if label not in class_counts:
                class_counts[label] = 0
                class_confidences[label] = []
            
            class_counts[label] += 1
            class_confidences[label].append(confidence)
        
        # 打印每个类别的统计
        for label in sorted(class_counts.keys()):
            count = class_counts[label]
            confs = class_confidences[label]
            avg_conf = sum(confs) / len(confs)
            max_conf = max(confs)
            min_conf = min(confs)
            
            print(f"\n[TestRunReco] 类别 '{label}':")
            print(f"  - 检测数: {count}")
            print(f"  - 平均置信度: {avg_conf:.3f}")
            print(f"  - 最高置信度: {max_conf:.3f}")
            print(f"  - 最低置信度: {min_conf:.3f}")
        
        # 打印所有检测的详细列表
        print(f"\n[TestRunReco] === 所有检测详情 ===")
        for i, det in enumerate(results):
            print(f"[{i}] {det['label']} @ {det['box']} | conf={det['confidence']:.3f} | cls_id={det['class_id']}")


# ==================== 注册 Custom Action ====================

# 注意: 实际注册需要在 agent_register.py 中完成
# 这里只是定义了 TestRunRecoHandler 类
