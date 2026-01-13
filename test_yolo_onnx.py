"""
YOLO ONNX 实时检测可视化工具

用于测试 YOLO 模型在实际游戏画面中的检测效果。
实时显示模拟器画面并标注检测到的目标。

使用方法:
    python test_yolo_onnx.py

功能:
- 实时截图并显示模拟器画面
- 使用 YOLO ONNX 模型进行目标检测
- 在画面上绘制检测框和标签
- 显示检测结果的详细信息（类别、置信度、坐标）
- 不执行任何控制操作，仅用于观察和测试

快捷键:
- ESC 或关闭窗口: 退出程序
- S: 保存当前帧到文件


"""

import os
import sys
import time
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path

# 添加 agent 目录到 Python 路径
_CURRENT_DIR = Path(__file__).parent
_AGENT_DIR = _CURRENT_DIR / "agent"
sys.path.insert(0, str(_AGENT_DIR))

# 导入 MaaFramework
from maa.context import Context
from maa.tasker import Tasker
from maa.resource import Resource
from maa.controller import AdbController
from maa.toolkit import Toolkit

# 导入 ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("❌ 错误: onnxruntime 未安装！")
    print("请安装: pip install onnxruntime-gpu")
    sys.exit(1)

# ==================== 配置 ====================
# 模型配置
MODEL_PATH = _CURRENT_DIR / "assets" / "resource" / "model" / "detect" / "farming.onnx"
YOLO_LABELS = ["bugs", "girl"]  # 模型类别标签
YOLO_THRESHOLD = 0.3  # 检测置信度阈值
YOLO_INPUT_SIZE = 640  # YOLOv8 输入尺寸
YOLO_NMS_THRESHOLD = 0.45  # NMS IOU 阈值

# 显示配置
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WINDOW_TITLE = "YOLO ONNX Real-time Detection Viewer"
FPS_TARGET = 10  # 目标帧率
FRAME_INTERVAL = 1.0 / FPS_TARGET  # 帧间隔

# 颜色配置 (RGB)
COLORS = {
    "bugs": (255, 0, 0),    # 红色
    "girl": (0, 255, 0),    # 绿色
    "info": (255, 255, 255), # 白色
    "fps": (255, 255, 0),   # 黄色
}

# 调试输出目录
DEBUG_OUTPUT_DIR = _CURRENT_DIR / "assets" / "debug" / "yolo_test"
DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== YOLO 推理类 ====================
class YOLODetector:
    """YOLO 检测器（使用 ONNX Runtime）"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session = None
        self._load_model()
    
    def _load_model(self):
        """加载 ONNX 模型"""
        if not self.model_path.exists():
            print(f"❌ 模型文件不存在: {self.model_path}")
            sys.exit(1)
        
        print(f"📦 加载 ONNX 模型: {self.model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            print(f"✅ 使用 Provider: {self.session.get_providers()}")
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            sys.exit(1)
    
    def preprocess(self, image: np.ndarray):
        """预处理图像"""
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 填充
        pad_w = (YOLO_INPUT_SIZE - new_w) // 2
        pad_h = (YOLO_INPUT_SIZE - new_h) // 2
        
        padded = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # BGR -> RGB -> CHW -> 归一化 -> 添加 batch 维度
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1)
        normalized = chw.astype(np.float32) / 255.0
        batched = np.expand_dims(normalized, axis=0)
        
        return batched, scale, pad_w, pad_h
    
    def postprocess(self, output: np.ndarray, scale: float, pad_w: int, pad_h: int):
        """后处理 YOLO 输出"""
        # 处理输出格式
        if len(output.shape) == 3 and output.shape[0] == 1:
            output = output[0]
        
        if output.shape[0] < output.shape[1]:  # (6, 8400)
            output = output.transpose(1, 0)  # -> (8400, 6)
        
        # 提取信息
        boxes = output[:, :4]  # (8400, 4)
        class_scores = output[:, 4:]  # (8400, num_classes)
        
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        
        # 过滤低置信度
        mask = confidences > YOLO_THRESHOLD
        if not mask.any():
            return []
        
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # 转换坐标: 中心点 -> 左上角
        x_centers, y_centers = boxes[:, 0], boxes[:, 1]
        widths, heights = boxes[:, 2], boxes[:, 3]
        
        x1 = x_centers - widths / 2
        y1 = y_centers - heights / 2
        
        # 反向变换: 去除填充和缩放
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        widths = widths / scale
        heights = heights / scale
        
        # NMS
        boxes_for_nms = np.stack([x1, y1, widths, heights], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            confidences.tolist(),
            YOLO_THRESHOLD,
            YOLO_NMS_THRESHOLD
        )
        
        if len(indices) == 0:
            return []
        
        # 构建结果
        results = []
        for i in indices.flatten():
            class_id = int(class_ids[i])
            label = YOLO_LABELS[class_id] if class_id < len(YOLO_LABELS) else f"class_{class_id}"
            
            results.append({
                "label": label,
                "class_id": class_id,
                "confidence": float(confidences[i]),
                "box": [int(x1[i]), int(y1[i]), int(widths[i]), int(heights[i])]
            })
        
        return results
    
    def detect(self, image: np.ndarray):
        """执行检测"""
        # 预处理
        input_tensor, scale, pad_w, pad_h = self.preprocess(image)
        
        # 推理
        input_name = self.session.get_inputs()[0].name
        output_names = [output.name for output in self.session.get_outputs()]
        
        outputs = self.session.run(output_names, {input_name: input_tensor})
        output = outputs[0]
        
        # 后处理
        detections = self.postprocess(output, scale, pad_w, pad_h)
        
        return detections


# ==================== 可视化窗口 ====================
class DetectionViewer:
    """检测结果可视化窗口"""
    
    def __init__(self, detector: YOLODetector, controller):
        self.detector = detector
        self.controller = controller
        
        # 创建窗口
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT + 100}")  # 额外空间显示统计信息
        
        # Canvas
        self.canvas = tk.Canvas(self.root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, bg='black')
        self.canvas.pack()
        
        # 统计信息 Label
        self.info_label = tk.Label(
            self.root, 
            text="", 
            font=("Consolas", 10), 
            bg='black', 
            fg='white',
            justify=tk.LEFT,
            anchor='w'
        )
        self.info_label.pack(fill=tk.BOTH, expand=True)
        
        # 状态
        self.running = True
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.photo_image = None
        self.canvas_image_id = None
        
        # 绑定按键
        self.root.bind('<Escape>', lambda e: self.stop())
        self.root.bind('s', lambda e: self.save_frame())
        self.root.bind('S', lambda e: self.save_frame())
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        
        print("\n📺 窗口已创建")
        print("快捷键:")
        print("  - ESC 或关闭窗口: 退出")
        print("  - S: 保存当前帧\n")
    
    def update_frame(self, image: np.ndarray, detections: list):
        """更新显示帧"""
        # 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 创建叠加层
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 绘制检测框
        for det in detections:
            label = det["label"]
            box = det["box"]
            confidence = det["confidence"]
            
            x, y, w, h = box
            color = COLORS.get(label, (128, 128, 128))
            
            # 半透明填充
            draw.rectangle([x, y, x + w, y + h], fill=color + (60,))
            
            # 边框
            draw.rectangle([x, y, x + w, y + h], outline=color + (255,), width=3)
            
            # 标签
            text = f"{label} {confidence:.2f}"
            draw.text((x, y - 20), text, fill=color + (255,))
            
            # 中心点
            cx, cy = x + w // 2, y + h // 2
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=color + (255,))
        
        # 合成
        pil_image = pil_image.convert('RGBA')
        pil_image = Image.alpha_composite(pil_image, overlay)
        pil_image = pil_image.convert('RGB')
        
        # 更新 Canvas
        self.photo_image = ImageTk.PhotoImage(pil_image)
        if self.canvas_image_id:
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo_image)
        else:
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        
        # 保存当前帧供截图使用
        self.current_display_image = pil_image
    
    def update_info(self, detections: list, inference_time: float):
        """更新统计信息"""
        # 统计各类别数量
        stats = {}
        for det in detections:
            label = det["label"]
            stats[label] = stats.get(label, 0) + 1
        
        # 构建信息文本
        info_lines = [
            f"FPS: {self.fps:.1f} | Frame: {self.frame_count} | Inference: {inference_time*1000:.1f}ms",
            f"Detections: {len(detections)} | " + " | ".join([f"{k}: {v}" for k, v in stats.items()]),
            "",
            "Detection Details:"
        ]
        
        for i, det in enumerate(detections[:10]):  # 最多显示10个
            label = det["label"]
            conf = det["confidence"]
            box = det["box"]
            info_lines.append(f"  [{i+1}] {label:8s} {conf:.3f}  box={box}")
        
        if len(detections) > 10:
            info_lines.append(f"  ... and {len(detections) - 10} more")
        
        self.info_label.config(text="\n".join(info_lines))
    
    def save_frame(self):
        """保存当前帧"""
        if hasattr(self, 'current_display_image'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = DEBUG_OUTPUT_DIR / f"frame_{timestamp}.png"
            self.current_display_image.save(filename)
            print(f"💾 已保存帧: {filename}")
    
    def run(self):
        """主循环"""
        print("🚀 开始检测循环...\n")
        
        def detection_loop():
            if not self.running:
                return
            
            loop_start = time.time()
            
            try:
                # 截图
                self.controller.post_screencap().wait()
                image = self.controller.cached_image
                
                if image is not None:
                    # 检测
                    inference_start = time.time()
                    detections = self.detector.detect(image)
                    inference_time = time.time() - inference_start
                    
                    # 更新显示
                    self.update_frame(image, detections)
                    self.update_info(detections, inference_time)
                    
                    # 更新帧计数和 FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_time >= 1.0:
                        self.fps = self.frame_count / (current_time - self.last_time)
                        self.frame_count = 0
                        self.last_time = current_time
            
            except Exception as e:
                print(f"❌ 检测循环错误: {e}")
                import traceback
                traceback.print_exc()
            
            # 控制帧率
            elapsed = time.time() - loop_start
            delay = max(10, int((FRAME_INTERVAL - elapsed) * 1000))
            self.root.after(delay, detection_loop)
        
        # 启动检测循环
        self.root.after(100, detection_loop)
        
        # 进入 Tkinter 主循环
        self.root.mainloop()
    
    def stop(self):
        """停止"""
        print("\n⏹️  停止...")
        self.running = False
        if self.root:
            self.root.quit()


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("    YOLO ONNX 实时检测可视化工具")
    print("=" * 60)
    
    # 初始化 Toolkit
    Toolkit.init_option(_CURRENT_DIR / "assets" / "config")
    
    # 查找 ADB 设备
    print("\n🔍 扫描 ADB 设备...")
    adb_devices = Toolkit.find_adb_devices()
    
    if not adb_devices:
        print("❌ 未找到 ADB 设备！")
        print("请确保:")
        print("  1. 模拟器已启动")
        print("  2. ADB 已连接")
        sys.exit(1)
    
    print(f"✅ 找到 {len(adb_devices)} 个设备:")
    for i, dev in enumerate(adb_devices):
        print(f"  [{i+1}] {dev.name} ({dev.adb_path})")
    
    # 选择设备
    if len(adb_devices) == 1:
        selected_device = adb_devices[0]
        print(f"\n📱 自动选择设备: {selected_device.name}")
    else:
        while True:
            try:
                choice = int(input(f"\n请选择设备 [1-{len(adb_devices)}]: "))
                if 1 <= choice <= len(adb_devices):
                    selected_device = adb_devices[choice - 1]
                    break
            except (ValueError, KeyboardInterrupt):
                print("\n❌ 已取消")
                sys.exit(0)
    
    # 创建控制器
    print(f"\n🔌 连接到设备: {selected_device.name}")
    controller = AdbController(
        adb_path=selected_device.adb_path,
        address=selected_device.address,
        screencap_methods=selected_device.screencap_methods,
        input_methods=selected_device.input_methods,
        config=selected_device.config
    )
    
    controller.post_connection().wait()
    print("✅ 设备已连接")
    
    # 初始化检测器
    print(f"\n🤖 初始化 YOLO 检测器...")
    print(f"   模型: {MODEL_PATH}")
    print(f"   类别: {YOLO_LABELS}")
    print(f"   阈值: {YOLO_THRESHOLD}")
    
    detector = YOLODetector(MODEL_PATH)
    
    # 创建可视化窗口
    viewer = DetectionViewer(detector, controller)
    
    # 运行
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    finally:
        print("\n✅ 程序结束")


if __name__ == "__main__":
    main()
