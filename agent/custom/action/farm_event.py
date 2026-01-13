"""
农场事件处理 Custom Action
实现农场中水车修理和虫子捕捉的自动化逻辑
使用名为MAA的conda环境

农场场景说明:
- 游戏画面 1280x720，是一片平面农场
- Q版少女角色位于画面中上位置
- 右下角 [97,420,253,255] 区域是虚拟摇杆，用于操纵角色移动
- 使用 YOLOv8 模型 (model/farming.onnx) 实时检测角色(girl)和虫子(bugs)位置

事件处理:
1. 水车修理:
   - 需要角色移动到 [1035,243] 附近
   - 然后模板匹配右下角 [1022,435,206,244] 区域的 farm/修理按钮.png
   - 点击修理按钮完成修理，弹出奖励页面
   - 调用 Sub_Getreward 节点获取奖励

2. 虫子捕捉 (TODO):
   - 使用 YOLOv8 模型检测虫子位置
   - 操纵角色移动到虫子附近
   - 执行捕捉动作
"""

from maa.custom_action import CustomAction
from maa.context import Context
import json
import time
import math
import os
import threading
from typing import Optional, List, Tuple, Callable
import numpy as np
import cv2

# Tkinter 和 PIL 用于调试窗口
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ==================== 调试配置 ====================
# 全局调试开关 - 设置为 True 开启调试功能，False 关闭所有调试功能
# 注意：Tkinter GUI 在 MaaFramework Agent 环境中可能导致崩溃
# 如果遇到"调试适配器意外终止"错误，请设置为 False
DEBUG_ENABLED = True 

# 调试输出目录（相对于项目根目录）
DEBUG_OUTPUT_DIR = "assets/debug/farm_debug"

# 调试文件保存模式 - 设置为 True 时保存截图到文件
DEBUG_FILE_MODE = False

# 可视化颜色配置 (RGB格式，用于 PIL)
DEBUG_COLORS = {
    "girl": (0, 255, 0),      # 绿色 - 角色
    "bugs": (255, 0, 0),      # 红色 - 虫子
    "target": (255, 0, 255),  # 紫色 - 目标位置
    "joystick": (255, 255, 0), # 黄色 - 摇杆
    "info": (255, 255, 255),  # 白色 - 信息文字
}

# 截图间隔配置
DEBUG_SCREENCAP_INTERVAL = 0.05  # 截图间隔（秒），50ms
DEBUG_STOPPING_CHECK_COUNT = 10  # 每多少次截图检查一次 stopping

# ==================== 路径配置 ====================
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))
_IMAGE_DIR = os.path.join(_PROJECT_ROOT, "assets", "resource", "image")
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "assets", "resource", "model")

# 农场相关图片目录
_FARM_IMG_DIR = "farm"

# ==================== 常量配置 ====================
# 屏幕尺寸
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# 虚拟摇杆配置
JOYSTICK_ROI = [97, 420, 253, 255]  # 摇杆区域 [x, y, w, h]
JOYSTICK_CENTER = (
    JOYSTICK_ROI[0] + JOYSTICK_ROI[2] // 2,  # 中心 X
    JOYSTICK_ROI[1] + JOYSTICK_ROI[3] // 2   # 中心 Y
)
JOYSTICK_RADIUS = min(JOYSTICK_ROI[2], JOYSTICK_ROI[3]) // 2 - 20  # 有效操作半径

# 水车修理相关
WATERWHEEL_TARGET_POS = (1035, 243)  # 角色需要移动到的目标位置
REPAIR_BUTTON_ROI = [1022, 435, 206, 244]  # 修理按钮检测区域
REPAIR_BUTTON_TEMPLATE = f"{_FARM_IMG_DIR}/修理按钮.png"

# 浇水相关
WATER_BUTTON_ROI = [1022, 435, 206, 244]  # 右下角浇水按钮区域
WATER_BUTTON_TEMPLATE = f"{_FARM_IMG_DIR}/浇水按钮.png"
WATER_BUTTON_THRESHOLD = 0.7
WATER_CLICK_WAIT = 0.8  # 点击浇水按钮后的等待时间（秒）

# 坑位“湿润”检测（ColorMatch）
# ROI：以坑位中心为左上角，宽 40 高 25
PLOT_WET_ROI_W = 40
PLOT_WET_ROI_H = 25
PLOT_WET_COLOR_RANGES = [
    # upper, lower
    ([45, 60, 83], [25, 40, 63]),
    ([63, 85, 118], [43, 65, 98]),
]
PLOT_WET_COUNT = 25  # 40x25=1000 像素，命中阈值取较小即可

# 土壤湿度 OCR 辅助检测（用于避免对同一坑位反复浇水）
# 以坑位中心为圆心、半径 150px 的外接方作为 ROI（300x300）
MOISTURE_OCR_RADIUS = 150

# 农场坑位中心坐标（16个）
# 用户提供格式为 [x, y, ?, ?]，这里仅使用 x、y
FARM_PLOT_CENTERS: List[Tuple[int, int]] = [
    (355, 346), (488, 344), (752, 344), (888, 346),
    (344, 420), (483, 422), (757, 418), (900, 418),
    (331, 500), (477, 498), (765, 499), (911, 501),
    (311, 589), (470, 586), (770, 588), (920, 583),
]

# 捉虫相关
CATCH_BUTTON_ROI = [1022, 435, 206, 244]  # 捕捉按钮检测区域（与修理按钮相同位置）
CATCH_BUTTON_TEMPLATE = f"{_FARM_IMG_DIR}/捕捉按钮.png"  # 捕捉按钮模板
CATCH_TOLERANCE = 60  # 接近虫子的容差（像素）
CATCH_ANIMATION_WAIT = 3.0  # 捕捉动画等待时间（秒）
NO_BUGS_CHECK_INTERVAL = 5.0  # 无虫子时的检测间隔（秒）
NO_BUGS_CHECK_ROUNDS = 3  # 无虫子时的检测轮数
MAX_BUGS_COUNT = 4  # 场上最多虫子数量

# YOLOv8 模型配置
# 注意：NeuralNetworkDetect 使用的是相对于 model/detect 文件夹的路径
# 如果模型直接放在 model/ 下，需要使用 "../farming.onnx"
# 如果模型放在 model/detect/ 下，直接使用 "farming.onnx"
FARMING_MODEL_PATH = "farming.onnx"  # 修正：相对于 model/detect 文件夹
YOLO_LABELS = ["bugs", "girl"]  # 模型类别标签
YOLO_GIRL_INDEX = 1  # girl 类别索引
YOLO_BUGS_INDEX = 0  # bugs 类别索引
YOLO_THRESHOLD = 0.3  # 检测置信度阈值（降低以提高检测率）

# 移动控制参数
MOVE_TOLERANCE = 26  # 到达目标位置的容差（像素）
MOVE_CHECK_INTERVAL = 0.1  # 移动过程中检测间隔（秒）- 降低以提高帧率
MOVE_TIMEOUT = 15  # 移动超时时间（秒）
SWIPE_DURATION = 500  # 摇杆滑动持续时间（毫秒）

# 默认等待时间
DEFAULT_WAIT = 1.0  # 默认等待时间（秒）
POST_ACTION_WAIT = 0.5  # 动作后等待时间（秒）


def _img_path(filename: str) -> str:
    """构建农场图片的相对路径（相对于 resource/image 目录）"""
    return f"{_FARM_IMG_DIR}/{filename}"


def _parse_param(custom_action_param) -> dict:
    """
    安全解析 custom_action_param 参数
    处理各种可能的输入格式
    """
    if not custom_action_param:
        return {}
    
    if isinstance(custom_action_param, dict):
        return custom_action_param
    
    if isinstance(custom_action_param, str):
        try:
            parsed = json.loads(custom_action_param)
            if isinstance(parsed, str):
                try:
                    return json.loads(parsed)
                except (json.JSONDecodeError, TypeError):
                    return {}
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    return {}


def _reco_hit(reco_result) -> bool:
    """检查 run_recognition 结果是否有效命中"""
    return reco_result is not None and reco_result.hit and reco_result.best_result is not None


def _reco_box(reco_result) -> list:
    """从 run_recognition 结果中获取 box"""
    if _reco_hit(reco_result):
        return reco_result.best_result.box
    return None


def _box_center(box: list) -> tuple:
    """计算 box 的中心坐标"""
    if box is None:
        return None
    return (box[0] + box[2] // 2, box[1] + box[3] // 2)


def _distance(pos1: tuple, pos2: tuple) -> float:
    """计算两点之间的欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def _clamp_roi(roi: List[int]) -> List[int]:
    """
    将 ROI [x,y,w,h] 裁剪到屏幕范围内，避免越界导致识别异常。
    """
    x, y, w, h = roi
    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)
    if w <= 0 or h <= 0:
        return [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT]
    # 裁剪右下边界
    if x + w > SCREEN_WIDTH:
        w = max(1, SCREEN_WIDTH - x)
    if y + h > SCREEN_HEIGHT:
        h = max(1, SCREEN_HEIGHT - y)
    return [x, y, w, h]


def _calculate_joystick_direction(current_pos: tuple, target_pos: tuple) -> tuple:
    """
    计算摇杆滑动的终点位置
    
    根据当前位置和目标位置，计算从摇杆中心滑动到边缘的方向
    
    参数:
        current_pos: 当前角色位置 (x, y)
        target_pos: 目标位置 (x, y)
        
    返回:
        tuple: 摇杆滑动终点坐标 (x, y)
    """
    # 计算方向向量
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    
    # 计算距离
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1:  # 避免除零
        return JOYSTICK_CENTER
    
    # 归一化并乘以摇杆半径
    norm_dx = dx / dist * JOYSTICK_RADIUS
    norm_dy = dy / dist * JOYSTICK_RADIUS
    
    # 计算摇杆终点位置
    end_x = int(JOYSTICK_CENTER[0] + norm_dx)
    end_y = int(JOYSTICK_CENTER[1] + norm_dy)
    
    return (end_x, end_y)


def _safe_swipe(context: Context, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
    """
    安全的滑动操作，使用底层 touch_down/move/up API 实现
    
    相比 post_swipe，此方法：
    1. 每次操作前重新获取 controller 引用
    2. 使用更底层的触控 API，更稳定
    3. 添加了更好的错误处理
    
    参数:
        context: MAA 上下文
        x1, y1: 起点坐标
        x2, y2: 终点坐标
        duration: 滑动持续时间（毫秒）
        
    返回:
        bool: 是否成功
    """
    try:
        # 每次操作都重新获取 controller 引用
        controller = context.tasker.controller
        
        # 确保所有参数都是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        duration = int(duration)
        
        # 步骤1: 按下起点
        touch_down_job = controller.post_touch_down(x1, y1, contact=0, pressure=1)
        if not touch_down_job.wait():
            print(f"[_safe_swipe] touch_down 失败")
            return False
        
        # 步骤2: 移动到终点（可以分多步移动以模拟滑动）
        # 简单的线性插值，分成多个小步
        steps = max(5, duration // 50)  # 至少5步，或每50ms一步
        for i in range(1, steps + 1):
            t = i / steps
            cur_x = int(x1 + (x2 - x1) * t)
            cur_y = int(y1 + (y2 - y1) * t)
            
            touch_move_job = controller.post_touch_move(cur_x, cur_y, contact=0, pressure=1)
            if not touch_move_job.wait():
                print(f"[_safe_swipe] touch_move 失败 (step {i}/{steps})")
                # 尝试抬起手指以清理状态
                try:
                    controller.post_touch_up(contact=0).wait()
                except:
                    pass
                return False
            
            # 短暂等待
            time.sleep(duration / steps / 1000.0)
        
        # 步骤3: 抬起
        touch_up_job = controller.post_touch_up(contact=0)
        if not touch_up_job.wait():
            print(f"[_safe_swipe] touch_up 失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"[_safe_swipe] 滑动异常: {e}")
        import traceback
        traceback.print_exc()
        # 尝试抬起手指以清理状态
        try:
            context.tasker.controller.post_touch_up(contact=0).wait()
        except:
            pass
        return False


# ==================== Tkinter 调试可视化模块 ====================
"""
调试可视化模块 - Tkinter 窗口 + 数据推送架构

重要限制：
MaaFramework 的 Controller 和 Context 对象不是线程安全的！
在独立线程中调用 post_screencap() 或 run_recognition() 会导致 IPC 通信错误。

正确架构：
1. 主线程执行任务逻辑，调用 MaaFramework API（截图、识别）
2. 主线程将截图和识别结果**推送**给 TkDebugViewer
3. TkDebugViewer 在独立线程运行 Tkinter 主循环，只负责**显示**

数据流：
主线程 --push_frame(image)--> TkDebugViewer (显示线程)
主线程 --push_detections(dets)--> TkDebugViewer (显示线程)
"""


class TkDebugViewer:
    """
    Tkinter 调试查看器（数据推送模式）
    
    不主动调用任何 MaaFramework API，只显示主线程推送的数据。
    在独立线程中运行 Tkinter 主循环。
    """
    
    def __init__(self):
        # 线程控制
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # 共享状态（线程安全）
        self._lock = threading.Lock()
        self._current_image: Optional[np.ndarray] = None
        self._detections: List[dict] = []
        self._target_pos: Optional[Tuple[int, int]] = None
        self._info_text: str = ""
        self._frame_count = 0
        self._need_refresh = False
        
        # Tkinter 相关（在 Tkinter 线程中初始化）
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._canvas_image_id = None
        
        # 窗口尺寸
        self._window_width = SCREEN_WIDTH
        self._window_height = SCREEN_HEIGHT
    
    def _check_stopping(self) -> bool:
        """检查是否应该停止"""
        return self._stop_flag.is_set()
    
    # ==================== 数据推送 API（主线程调用） ====================
    
    def push_frame(self, image: np.ndarray):
        """
        推送新的截图帧（由主线程调用）
        
        Args:
            image: BGR 格式的 numpy 图像
        """
        if image is None or image.size == 0:
            return
        
        with self._lock:
            self._current_image = image.copy()
            self._frame_count += 1
            self._need_refresh = True
    
    def push_detections(self, detections: List[dict]):
        """
        推送检测结果（由主线程调用）
        
        Args:
            detections: 检测结果列表 [{"label": str, "box": [x,y,w,h], "confidence": float}, ...]
        """
        with self._lock:
            self._detections = detections.copy() if detections else []
            self._need_refresh = True
    
    def set_target(self, pos: Optional[Tuple[int, int]]):
        """设置目标位置（由主线程调用）"""
        with self._lock:
            self._target_pos = pos
            self._need_refresh = True
    
    def set_info(self, text: str):
        """设置信息文字（由主线程调用）"""
        with self._lock:
            self._info_text = text
            self._need_refresh = True
    
    def get_detections(self) -> List[dict]:
        """获取当前检测结果"""
        with self._lock:
            return self._detections.copy()
    
    # ==================== Tkinter 线程 ====================
    
    def _tkinter_thread(self):
        """Tkinter 主循环线程"""
        print("[TkDebugViewer] Tkinter 线程启动")
        
        try:
            # 创建 Tkinter 窗口
            self._root = tk.Tk()
            self._root.title("Farm Debug Viewer - YOLO Detection [Push Mode]")
            self._root.geometry(f"{self._window_width}x{self._window_height}")
            self._root.protocol("WM_DELETE_WINDOW", self._on_close)
            
            # 创建 Canvas
            self._canvas = tk.Canvas(
                self._root,
                width=self._window_width,
                height=self._window_height,
                bg='black'
            )
            self._canvas.pack(fill=tk.BOTH, expand=True)
            
            # 创建初始占位图像
            self._create_placeholder()
            
            print("[TkDebugViewer] 窗口已创建")
            
            # 启动定时刷新
            self._schedule_refresh()
            
            # 进入 Tkinter 主循环
            self._root.mainloop()
            
        except Exception as e:
            print(f"[TkDebugViewer] Tkinter 线程异常: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._root = None
            self._canvas = None
            print("[TkDebugViewer] Tkinter 线程退出")
    
    def _create_placeholder(self):
        """创建占位图像"""
        # 创建深灰色背景
        placeholder = np.zeros((self._window_height, self._window_width, 3), dtype=np.uint8)
        placeholder[:] = (40, 40, 40)
        
        # 转换为 PIL
        pil_image = Image.fromarray(placeholder)
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制等待提示
        text = "Waiting for frames..."
        draw.text((self._window_width // 2 - 80, self._window_height // 2), text, fill=(128, 128, 128))
        draw.text((10, 10), "Farm Debug [Push Mode]", fill=(100, 100, 100))
        
        # 显示
        self._photo_image = ImageTk.PhotoImage(pil_image)
        self._canvas_image_id = self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo_image)
    
    def _schedule_refresh(self):
        """安排定时刷新"""
        if self._root and not self._check_stopping():
            self._refresh_display()
            # 每 33ms 刷新一次（约 30fps）
            self._root.after(33, self._schedule_refresh)
    
    def _refresh_display(self):
        """刷新显示（在 Tkinter 主线程中调用）"""
        if self._check_stopping() or self._root is None:
            return
        
        # 检查是否有新数据需要刷新
        with self._lock:
            if not self._need_refresh and self._current_image is None:
                return
            
            if self._current_image is None:
                return
            
            image = self._current_image.copy()
            detections = self._detections.copy()
            target = self._target_pos
            info = self._info_text
            frame_count = self._frame_count
            self._need_refresh = False
        
        try:
            # 转换 numpy (BGR) -> PIL (RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 创建绘图对象（支持 RGBA）
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # 绘制检测框
            for det in detections:
                label = det.get("label", "unknown")
                box = det.get("box", [0, 0, 0, 0])
                confidence = det.get("confidence", 0.0)
                
                color = DEBUG_COLORS.get(label, (128, 128, 128))
                x, y, w, h = box
                
                # 绘制半透明填充遮罩
                overlay_color = color + (60,)  # 添加 alpha 通道
                draw.rectangle([x, y, x + w, y + h], fill=overlay_color)
                
                # 绘制矩形边框
                draw.rectangle([x, y, x + w, y + h], outline=color + (255,), width=2)
                
                # 绘制标签
                text = f"{label} {confidence:.2f}" if confidence > 0 else label
                draw.text((x, y - 15), text, fill=color + (255,))
                
                # 绘制中心点
                cx, cy = x + w // 2, y + h // 2
                draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color + (255,))
            
            # 绘制目标位置
            if target:
                tx, ty = target
                color = DEBUG_COLORS["target"]
                # 十字准心
                draw.line([(tx - 20, ty), (tx + 20, ty)], fill=color + (255,), width=2)
                draw.line([(tx, ty - 20), (tx, ty + 20)], fill=color + (255,), width=2)
                # 容差圆
                draw.ellipse([tx - MOVE_TOLERANCE, ty - MOVE_TOLERANCE,
                             tx + MOVE_TOLERANCE, ty + MOVE_TOLERANCE], outline=color + (255,), width=1)
                draw.text((tx + 10, ty - 25), "TARGET", fill=color + (255,))
            
            # 绘制摇杆区域
            jx, jy, jw, jh = JOYSTICK_ROI
            jcolor = DEBUG_COLORS["joystick"]
            draw.rectangle([jx, jy, jx + jw, jy + jh], outline=jcolor + (255,), width=1)
            jcx, jcy = JOYSTICK_CENTER
            draw.ellipse([jcx - 4, jcy - 4, jcx + 4, jcy + 4], fill=jcolor + (255,))
            draw.ellipse([jcx - JOYSTICK_RADIUS, jcy - JOYSTICK_RADIUS,
                         jcx + JOYSTICK_RADIUS, jcy + JOYSTICK_RADIUS], outline=jcolor + (255,), width=1)
            
            # 绘制信息文字
            info_color = DEBUG_COLORS["info"]
            draw.text((10, 10), f"Farm Debug [Push Mode]", fill=info_color + (255,))
            draw.text((10, 30), f"Frame: {frame_count}", fill=info_color + (255,))
            draw.text((10, 50), f"Detections: {len(detections)}", fill=info_color + (255,))
            if info:
                draw.text((10, 70), info, fill=(255, 255, 0, 255))
            
            timestamp = time.strftime("%H:%M:%S")
            draw.text((self._window_width - 80, 10), timestamp, fill=info_color + (255,))
            
            # 合成叠加层
            pil_image = pil_image.convert('RGBA')
            pil_image = Image.alpha_composite(pil_image, overlay)
            pil_image = pil_image.convert('RGB')
            
            # 更新到 Canvas
            self._photo_image = ImageTk.PhotoImage(pil_image)
            if self._canvas_image_id:
                self._canvas.itemconfig(self._canvas_image_id, image=self._photo_image)
            else:
                self._canvas_image_id = self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo_image)
            
        except Exception as e:
            if not self._check_stopping():
                print(f"[TkDebugViewer] 刷新显示异常: {e}")
    
    def _on_close(self):
        """窗口关闭回调"""
        print("[TkDebugViewer] 用户关闭窗口")
        self._stop_flag.set()
        if self._root:
            self._root.quit()
    
    # ==================== 生命周期管理 ====================
    
    def start(self):
        """启动调试查看器（非阻塞，在独立线程中运行）"""
        if self._thread and self._thread.is_alive():
            print("[TkDebugViewer] 已在运行")
            return
        
        print("[TkDebugViewer] 启动调试查看器...")
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._tkinter_thread, name="TkDebugViewer", daemon=True)
        self._thread.start()
        
        # 等待窗口创建
        time.sleep(0.3)
    
    def stop(self):
        """停止调试查看器"""
        print("[TkDebugViewer] 请求停止...")
        self._stop_flag.set()
        
        # 请求关闭 Tkinter 主循环
        if self._root:
            try:
                self._root.quit()
            except:
                pass
        
        # 等待线程结束
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        self._thread = None
        print("[TkDebugViewer] 已停止")
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._thread is not None and self._thread.is_alive()
    
    # 保留旧的 run() 方法以兼容
    def run(self):
        """运行调试查看器（阻塞式，保留以兼容旧代码）"""
        self._tkinter_thread()


# ==================== 全局调试实例管理 ====================

_debug_viewer: Optional[TkDebugViewer] = None


def get_debug_viewer() -> Optional[TkDebugViewer]:
    """获取当前调试查看器实例"""
    return _debug_viewer


def start_debug_viewer() -> Optional[TkDebugViewer]:
    """启动调试查看器（非阻塞）"""
    global _debug_viewer
    if not DEBUG_ENABLED:
        return None
    if _debug_viewer and _debug_viewer.is_running():
        return _debug_viewer
    _debug_viewer = TkDebugViewer()
    _debug_viewer.start()
    return _debug_viewer


def stop_debug_viewer():
    """停止调试查看器"""
    global _debug_viewer
    if _debug_viewer:
        _debug_viewer.stop()
        _debug_viewer = None


# ==================== 便捷 API（供主线程调用） ====================

def push_debug_frame(image: np.ndarray):
    """推送截图帧到调试窗口（主线程调用）"""
    viewer = get_debug_viewer()
    if viewer:
        viewer.push_frame(image)


def push_debug_detections(detections: List[dict]):
    """推送检测结果到调试窗口（主线程调用）"""
    viewer = get_debug_viewer()
    if viewer:
        viewer.push_detections(detections)


def set_debug_target(target_pos: Optional[Tuple[int, int]]):
    """设置调试目标位置"""
    viewer = get_debug_viewer()
    if viewer:
        viewer.set_target(target_pos)


def set_debug_info(info_text: str):
    """设置调试信息文字"""
    viewer = get_debug_viewer()
    if viewer:
        viewer.set_info(info_text)


# ==================== 兼容旧 API ====================

def get_debug_visualizer():
    """兼容旧 API"""
    return get_debug_viewer()


def start_debug_visualizer(context: Context = None):
    """兼容旧 API"""
    return start_debug_viewer()


def stop_debug_visualizer():
    """兼容旧 API"""
    stop_debug_viewer()


def update_debug_frame(frame: np.ndarray = None) -> bool:
    """兼容旧 API - 推送帧"""
    if frame is not None:
        push_debug_frame(frame)
    return True


def update_debug_detections(detections: List[dict]):
    """兼容旧 API - 推送检测结果"""
    push_debug_detections(detections)


def _joystick_nudge_up(context: Context, times: int = 2, duration_ms: int = 300, wait_s: float = 0.15) -> bool:
    """
    当检测不到角色时，用“向上轻推摇杆”尝试把角色从遮挡/边缘拉回视野。

    Args:
        times: 推动次数
        duration_ms: 每次推动时长
        wait_s: 每次推动后的等待

    Returns:
        bool: 操作流程是否正常执行（不代表一定能找回角色）
    """
    x1 = int(JOYSTICK_CENTER[0])
    y1 = int(JOYSTICK_CENTER[1])
    x2 = int(JOYSTICK_CENTER[0])
    y2 = int(JOYSTICK_CENTER[1] - JOYSTICK_RADIUS)

    ok = True
    for i in range(times):
        print(f"[捉虫] 检测不到角色，尝试向上推动摇杆 ({i + 1}/{times})...")
        if not _safe_swipe(context, x1, y1, x2, y2, int(duration_ms)):
            ok = False
        time.sleep(wait_s)
    return ok


def _joystick_nudge_down(context: Context, times: int = 1, duration_ms: int = 300, wait_s: float = 0.2) -> bool:
    """
    轻推摇杆向下走一小段，用于触发/刷新交互 UI（例如土壤湿度提示）。

    注意：只做“补救”用，避免卡在坑位边缘但没触发 UI。
    """
    x1 = int(JOYSTICK_CENTER[0])
    y1 = int(JOYSTICK_CENTER[1])
    x2 = int(JOYSTICK_CENTER[0])
    y2 = int(JOYSTICK_CENTER[1] + JOYSTICK_RADIUS)

    ok = True
    for i in range(times):
        print(f"[浇水] OCR 未命中，尝试向下推动摇杆触发 UI ({i + 1}/{times})...")
        if not _safe_swipe(context, x1, y1, x2, y2, int(duration_ms)):
            ok = False
        time.sleep(wait_s)
    return ok


# ==================== FarmEventHandler 类 ====================

class FarmEventHandler(CustomAction):
    """
    农场事件处理器
    
    通过参数区分不同的事件类型:
    - event_type: "waterwheel" - 水车修理
    - event_type: "worms" - 捉虫
    
    Pipeline 调用示例:
    {
        "custom_action": "FarmEventHandler",
        "custom_action_param": {"event_type": "waterwheel"}
    }
    """
    
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """执行农场事件处理"""
        
        print("=" * 60)
        print("[FarmEventHandler] 农场事件处理器启动")
        
        # 启动调试可视化（数据推送模式）
        if DEBUG_ENABLED:
            print("[FarmEventHandler] 调试模式已启用，启动 Tkinter 调试窗口...")
            start_debug_viewer()  # 非阻塞，在独立线程运行
            time.sleep(0.5)  # 等待窗口创建
            print("[FarmEventHandler] 调试窗口已启动")
        
        # 解析参数
        params = _parse_param(argv.custom_action_param)
        event_type = params.get("event_type", "waterwheel")
        
        print(f"[FarmEventHandler] 事件类型: {event_type}")
        print(f"[FarmEventHandler] 完整参数: {params}")
        
        try:
            if event_type == "waterwheel":
                # 水车修理
                success = self._handle_waterwheel_repair(context)
            elif event_type == "watering":
                # 全农场浇水（16坑位遍历）
                success = self._handle_watering_all_plots(context)
            elif event_type == "worms":
                # 捉虫（待实现）
                success = self._handle_worm_catching(context, params)
            else:
                print(f"[FarmEventHandler] 未知事件类型: {event_type}")
                return CustomAction.RunResult(success=False)
            
            print(f"[FarmEventHandler] 事件处理{'成功' if success else '失败'}")
            return CustomAction.RunResult(success=success)
            
        except Exception as e:
            print(f"[FarmEventHandler] 发生异常: {e}")
            import traceback
            traceback.print_exc()
            return CustomAction.RunResult(success=False)
        
        finally:
            # 停止调试可视化
            if DEBUG_ENABLED:
                print("[FarmEventHandler] 停止调试可视化窗口...")
                stop_debug_viewer()
    
    def _detect_girl_position(self, context: Context, image=None) -> tuple:
        """
        使用 YOLOv8 模型检测角色位置

        参数:
            context: MAA 上下文
            image: 屏幕截图（可选，为 None 时自动截图）

        返回:
            tuple: 角色中心坐标 (x, y)，未检测到返回 None
        """
        if image is None:
            try:
                context.tasker.controller.post_screencap().wait()
                time.sleep(0.05)  # 短暂延迟确保截图完成
                image = context.tasker.controller.cached_image
            except Exception as e:
                print(f"[FarmEventHandler] 截图失败: {e}")
                return None

        # 更新调试画面（传入当前截图）
        if DEBUG_ENABLED and image is not None:
            update_debug_frame(image)
        
        # 使用 MaaFramework run_recognition 进行检测
        print("[FarmEventHandler] 使用 MaaFramework run_recognition 检测角色...")
        
        # 保存调试截图到文件（仅第一次）
        if DEBUG_FILE_MODE and not hasattr(self, '_debug_screenshot_saved'):
            import os
            debug_dir = os.path.join(_PROJECT_ROOT, DEBUG_OUTPUT_DIR)
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "input_screenshot.png")
            cv2.imwrite(debug_path, image)
            print(f"[FarmEventHandler] 调试截图已保存到: {debug_path}")
            self._debug_screenshot_saved = True

        print("[FarmEventHandler] 使用 YOLOv8 检测角色位置...")
        print(f"[FarmEventHandler] 图像尺寸: {image.shape if image is not None else 'None'}")
        print(f"[FarmEventHandler] 模型路径: {FARMING_MODEL_PATH}")
        print(f"[FarmEventHandler] 检测阈值: {YOLO_THRESHOLD}")
        print(f"[FarmEventHandler] 期望类别: girl (index={YOLO_GIRL_INDEX})")

        # 使用 expected 参数指定只检测 girl 类别
        # 注意：不指定 expected 时，best_result 会返回所有类别中置信度最高的结果
        # 这会导致 bugs 置信度略高时返回 bugs 而不是 girl
        reco_result = context.run_recognition(
            "Farm_DetectGirl",
            image,
            {
                "Farm_DetectGirl": {
                    "recognition": "NeuralNetworkDetect",
                    "model": FARMING_MODEL_PATH,
                    "expected": [YOLO_GIRL_INDEX],  # 关键：只期望检测 girl 类别
                    "threshold": YOLO_THRESHOLD,
                    "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
                }
            }
        )

        # 调试：打印识别结果
        print(f"[FarmEventHandler] 识别结果: hit={reco_result.hit if reco_result else 'None'}")
        if reco_result and hasattr(reco_result, 'best_result') and reco_result.best_result:
            print(f"[FarmEventHandler] best_result: label={reco_result.best_result.label}, "
                  f"cls_index={reco_result.best_result.cls_index}, score={reco_result.best_result.score:.3f}")

        if _reco_hit(reco_result):
            box = _reco_box(reco_result)
            center = _box_center(box)
            confidence = reco_result.best_result.score if hasattr(reco_result.best_result, 'score') else 0.0

            print(f"[FarmEventHandler] ✓ 检测到角色 (girl): box={box}, center={center}, confidence={confidence:.3f}")

            # 更新调试可视化检测结果
            if DEBUG_ENABLED:
                update_debug_detections([{
                    "label": "girl",
                    "box": list(box),
                    "confidence": confidence
                }])

            return center

        print("[FarmEventHandler] 未检测到角色")
        # 清空调试检测结果
        if DEBUG_ENABLED:
            update_debug_detections([])
            update_debug_frame()
        return None

    def _detect_girl_box_and_center(self, context: Context, image=None) -> Tuple[Optional[list], Optional[tuple]]:
        """
        检测角色 box 与中心点。

        Returns:
            (box, center)；未检测到则 (None, None)
        """
        if image is None:
            try:
                context.tasker.controller.post_screencap().wait()
                time.sleep(0.03)
                image = context.tasker.controller.cached_image
            except Exception as e:
                print(f"[FarmEventHandler] 截图失败: {e}")
                return None, None

        # 更新调试画面
        if DEBUG_ENABLED and image is not None:
            update_debug_frame(image)

        reco_result = context.run_recognition(
            "Farm_DetectGirl",
            image,
            {
                "Farm_DetectGirl": {
                    "recognition": "NeuralNetworkDetect",
                    "model": FARMING_MODEL_PATH,
                    "expected": [YOLO_GIRL_INDEX],
                    "threshold": YOLO_THRESHOLD,
                    "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
                }
            },
        )

        if _reco_hit(reco_result):
            box = _reco_box(reco_result)
            center = _box_center(box)
            confidence = reco_result.best_result.score if hasattr(reco_result.best_result, 'score') else 0.0

            if DEBUG_ENABLED:
                update_debug_detections([{
                    "label": "girl",
                    "box": list(box),
                    "confidence": confidence
                }])

            return box, center

        if DEBUG_ENABLED:
            update_debug_detections([])
        return None, None
    
    def _detect_all_objects(self, context: Context, image=None) -> dict:
        """
        使用 YOLOv8 模型检测所有对象（角色和虫子）
        
        参数:
            context: MAA 上下文
            image: 屏幕截图（可选）
            
        返回:
            dict: {"girl": [(box, confidence), ...], "bugs": [(box, confidence), ...]}
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            image = context.tasker.controller.cached_image
        
        # 更新调试画面（传入当前截图）
        if DEBUG_ENABLED and image is not None:
            update_debug_frame(image)
        
        results = {"girl": [], "bugs": []}
        debug_detections = []
        
        # 检测所有类别
        for label_idx, label_name in enumerate(YOLO_LABELS):
            reco_result = context.run_recognition(
                f"Farm_Detect_{label_name}",
                image,
                {
                    f"Farm_Detect_{label_name}": {
                        "recognition": "NeuralNetworkDetect",
                        "model": FARMING_MODEL_PATH,
                        "labels": YOLO_LABELS,
                        "expected": [label_idx],
                        "threshold": YOLO_THRESHOLD,
                        "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
                    }
                }
            )
            
            if _reco_hit(reco_result):
                box = _reco_box(reco_result)
                confidence = 0.0
                if hasattr(reco_result.best_result, 'score'):
                    confidence = reco_result.best_result.score
                
                results[label_name].append((box, confidence))
                debug_detections.append({
                    "label": label_name,
                    "box": list(box),
                    "confidence": confidence
                })
        
        # 更新调试可视化并刷新画面
        if DEBUG_ENABLED:
            update_debug_detections(debug_detections)
            update_debug_frame()
        
        return results
    
    def _move_character_to_target(
        self,
        context: Context,
        target_pos: tuple,
        *,
        target_offset: Tuple[int, int] = (0, 0),
        use_feet: bool = False,
    ) -> bool:
        """
        操纵角色移动到目标位置

        使用虚拟摇杆控制角色移动，通过 YOLOv8 实时检测角色位置

        参数:
            context: MAA 上下文
            target_pos: 目标位置 (x, y)

        返回:
            bool: 是否成功到达目标位置
        """
        start_time = time.time()

        target_pos = (int(target_pos[0] + target_offset[0]), int(target_pos[1] + target_offset[1]))
        print(f"[FarmEventHandler] 开始移动角色到目标位置: {target_pos} (use_feet={use_feet})")
        print(f"[FarmEventHandler] 摇杆中心: {JOYSTICK_CENTER}, 半径: {JOYSTICK_RADIUS}")

        # 立即截图并推送到调试窗口，避免窗口一直显示 "Waiting for frames"
        if DEBUG_ENABLED:
            print("[FarmEventHandler] 截图并推送初始帧到调试窗口...")
            try:
                controller = context.tasker.controller
                controller.post_screencap().wait()
                time.sleep(0.03)
                initial_image = controller.cached_image
                if initial_image is not None:
                    push_debug_frame(initial_image)
                    print("[FarmEventHandler] 初始帧已推送")
            except Exception as e:
                print(f"[FarmEventHandler] 初始截图失败: {e}")

        # 设置调试目标位置
        if DEBUG_ENABLED:
            set_debug_target(target_pos)
            set_debug_info("Moving to target...")
        
        # 角色丢失计数器（连续丢失超过一定次数才放弃）
        lost_count = 0
        max_lost_count = 3  # 允许连续丢失3次
        screencap_fail_count = 0
        max_screencap_fail_count = 20  # 连续截图失败上限（避免死循环/刷屏）

        while time.time() - start_time < MOVE_TIMEOUT:
            # 先截图
            try:
                # 每次循环都重新获取 controller 引用，避免潜在的失效指针/句柄问题
                controller = context.tasker.controller
                controller.post_screencap().wait()
                time.sleep(0.03)  # 短暂延迟，确保 cached_image 刷新
                current_image = controller.cached_image
                # 成功后重置连续失败计数
                screencap_fail_count = 0

                # 调试窗口强制推送当前帧（避免由于后续识别分支导致不刷新）
                if DEBUG_ENABLED and current_image is not None:
                    push_debug_frame(current_image)
            except Exception as e:
                screencap_fail_count += 1
                print(f"[FarmEventHandler] 截图失败: {e}，等待重试... ({screencap_fail_count}/{max_screencap_fail_count})")
                if DEBUG_ENABLED:
                    set_debug_info(f"Screencap failed ({screencap_fail_count}/{max_screencap_fail_count})")

                # 连续失败过多：直接放弃本次移动，交由上层逻辑继续/跳过
                if screencap_fail_count >= max_screencap_fail_count:
                    print("[FarmEventHandler] 连续截图失败过多，放弃移动以避免卡死")
                    if DEBUG_ENABLED:
                        set_debug_target(None)
                    return False

                # 退避等待（逐步加长），减少 IPC/截图压力
                backoff = min(0.6, MOVE_CHECK_INTERVAL * (1 + screencap_fail_count / 5))
                time.sleep(backoff)
                continue
            
            # 检测当前角色位置（传入截图避免内部再次截图）
            box, center_pos = self._detect_girl_box_and_center(context, current_image)
            if box is not None and use_feet:
                # feet 点：box 底部中心
                current_pos = (int(box[0] + box[2] // 2), int(box[1] + box[3]))
            else:
                current_pos = center_pos
            
            if current_pos is None:
                lost_count += 1
                print(f"[FarmEventHandler] 无法检测到角色 ({lost_count}/{max_lost_count})，等待重试...")
                if DEBUG_ENABLED:
                    set_debug_info(f"Character not detected ({lost_count}/{max_lost_count})")

                # 复用捉虫逻辑：检测不到角色时，先向上轻推摇杆两次尝试把角色拉回视野
                _joystick_nudge_up(context, times=2, duration_ms=500, wait_s=0.2)

                # 如果连续丢失次数过多，放弃
                if lost_count >= max_lost_count:
                    print(f"[FarmEventHandler] 连续 {lost_count} 次无法检测到角色，放弃移动")
                    return False

                time.sleep(MOVE_CHECK_INTERVAL)
                continue
            
            # 检测到角色，重置丢失计数器
            lost_count = 0
            
            # 计算与目标的距离
            dist = _distance(current_pos, target_pos)
            print(f"[FarmEventHandler] 当前位置: {current_pos}, 距离目标: {dist:.1f}px")
            
            # 更新调试信息
            if DEBUG_ENABLED:
                elapsed = time.time() - start_time
                set_debug_info(f"Distance: {dist:.1f}px | Time: {elapsed:.1f}s")
            
            # 检查是否到达目标
            if dist <= MOVE_TOLERANCE:
                print(f"[FarmEventHandler] 已到达目标位置！")
                if DEBUG_ENABLED:
                    set_debug_info("Target reached!")
                    set_debug_target(None)  # 清除目标标记
                return True
            
            # 计算摇杆滑动方向
            joystick_end = _calculate_joystick_direction(current_pos, target_pos)
            print(f"[FarmEventHandler] 摇杆滑动: {JOYSTICK_CENTER} -> {joystick_end}")
            
            # 执行摇杆滑动操作 - 确保所有参数都是整数类型
            x1 = int(JOYSTICK_CENTER[0])
            y1 = int(JOYSTICK_CENTER[1])
            x2 = int(joystick_end[0])
            y2 = int(joystick_end[1])
            duration = int(SWIPE_DURATION)
            
            print(f"[FarmEventHandler] 执行摇杆操作: ({x1},{y1}) -> ({x2},{y2}), duration={duration}ms")
            
            # 使用安全滑动函数（基于 touch_down/move/up 实现）
            if _safe_swipe(context, x1, y1, x2, y2, duration):
                print(f"[FarmEventHandler] 摇杆操作成功")
            else:
                print(f"[FarmEventHandler] 摇杆操作失败，等待后重新检测...")
                time.sleep(0.3)
                continue
            
            # 短暂等待后继续检测
            time.sleep(MOVE_CHECK_INTERVAL)
        
        print(f"[FarmEventHandler] 移动超时！未能到达目标位置")
        if DEBUG_ENABLED:
            set_debug_info("Move timeout!")
            set_debug_target(None)
        return False
    
    def _check_repair_button(self, context: Context, image=None) -> tuple:
        """
        检测修理按钮是否出现
        
        参数:
            context: MAA 上下文
            image: 屏幕截图（可选）
            
        返回:
            tuple: (是否出现, box) 
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            image = context.tasker.controller.cached_image
        
        print("[FarmEventHandler] 检测修理按钮...")
        
        reco_result = context.run_recognition(
            "Farm_CheckRepairButton",
            image,
            {
                "Farm_CheckRepairButton": {
                    "recognition": "TemplateMatch",
                    "template": REPAIR_BUTTON_TEMPLATE,
                    "roi": REPAIR_BUTTON_ROI,
                    "threshold": 0.7,
                }
            }
        )
        
        if _reco_hit(reco_result):
            box = _reco_box(reco_result)
            print(f"[FarmEventHandler] 检测到修理按钮: {box}")
            return (True, box)
        
        print("[FarmEventHandler] 未检测到修理按钮")
        return (False, None)
    
    def _click_repair_button(self, context: Context, box: list) -> bool:
        """
        点击修理按钮
        
        参数:
            context: MAA 上下文
            box: 修理按钮的 box
            
        返回:
            bool: 是否成功点击
        """
        if box is None:
            return False
        
        controller = context.tasker.controller
        center = _box_center(box)
        
        print(f"[FarmEventHandler] 点击修理按钮: {center}")
        controller.post_click(center[0], center[1]).wait()
        
        time.sleep(POST_ACTION_WAIT)
        return True

    def _check_water_button(self, context: Context, image=None) -> Tuple[bool, Optional[list]]:
        """
        检测浇水按钮是否出现

        返回:
            (是否出现, box)
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            image = context.tasker.controller.cached_image

        reco_result = context.run_recognition(
            "Farm_CheckWaterButton",
            image,
            {
                "Farm_CheckWaterButton": {
                    "recognition": "TemplateMatch",
                    "template": WATER_BUTTON_TEMPLATE,
                    "roi": WATER_BUTTON_ROI,
                    "threshold": WATER_BUTTON_THRESHOLD,
                }
            },
        )

        if _reco_hit(reco_result):
            return True, _reco_box(reco_result)
        return False, None

    def _click_water_button(self, context: Context, box: Optional[list]) -> bool:
        """点击浇水按钮"""
        if box is None:
            return False
        center = _box_center(box)
        if center is None:
            return False
        try:
            context.tasker.controller.post_click(int(center[0]), int(center[1])).wait()
        except Exception as e:
            print(f"[FarmEventHandler] 点击浇水按钮异常: {e}")
            return False
        time.sleep(POST_ACTION_WAIT)
        return True

    def _is_plot_wet(self, context: Context, plot_pos: Tuple[int, int], image=None) -> bool:
        """
        使用 ColorMatch 判断坑位是否已湿润。

        ROI：以坑位中心为左上角，宽 40 高 25。
        满足任意一组颜色范围即视为湿润。
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            time.sleep(0.03)
            image = context.tasker.controller.cached_image

        x, y = int(plot_pos[0]), int(plot_pos[1])
        roi = _clamp_roi([x, y, PLOT_WET_ROI_W, PLOT_WET_ROI_H])

        any_of = []
        for upper, lower in PLOT_WET_COLOR_RANGES:
            any_of.append({
                "recognition": "ColorMatch",
                "roi": roi,
                "upper": upper,
                "lower": lower,
                "count": PLOT_WET_COUNT,
            })

        reco_result = context.run_recognition(
            "Farm_PlotWetCheck",
            image,
            {
                "Farm_PlotWetCheck": {
                    "recognition": "Or",
                    "any_of": any_of,
                }
            },
        )

        return _reco_hit(reco_result)

    def _detect_soil_moisture_state(self, context: Context, plot_pos: Tuple[int, int], image=None) -> str:
        """
        识别当前坑位土壤湿度状态。

        Returns:
            "wet" | "dry" | "unknown"
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            time.sleep(0.03)
            image = context.tasker.controller.cached_image

        x, y = int(plot_pos[0]), int(plot_pos[1])
        # 以坑位中心为圆心、半径 MOISTURE_OCR_RADIUS 的外接方做 OCR（湿润/缺水/干燥）
        # 当前浇水主逻辑以 ColorMatch 为主，OCR 作为辅助“湿润”判定（防止重复浇水）。
        radius = int(MOISTURE_OCR_RADIUS)
        roi = _clamp_roi([
            x - radius,
            y - radius,
            radius * 2,
            radius * 2,
        ])

        # 优先判定“湿润”
        wet_reco = context.run_recognition(
            "Farm_Moisture_Wet",
            image,
            {
                "Farm_Moisture_Wet": {
                    "recognition": "OCR",
                    "expected": ["湿润"],
                    "roi": roi,
                }
            },
        )
        if _reco_hit(wet_reco):
            return "wet"

        # 再判定“缺水/干燥”
        dry_reco = context.run_recognition(
            "Farm_Moisture_Dry",
            image,
            {
                "Farm_Moisture_Dry": {
                    "recognition": "OCR",
                    "expected": ["缺水", "干燥"],
                    "roi": roi,
                }
            },
        )
        if _reco_hit(dry_reco):
            return "dry"

        return "unknown"

    def _handle_watering_all_plots(self, context: Context) -> bool:
        """
        全农场浇水：遍历16个坑位，将土壤浇到“湿润”，再额外浇1次。

        期间若出现奖励弹窗则优先领取（Sub_Getreward）。
        """
        print("\n" + "=" * 40)
        print("[浇水] 开始全农场浇水任务")
        print("=" * 40)

        controller = context.tasker.controller
        max_water_per_plot = 5      # 最大浇水次数
        moisture_retry = 3          # 湿度识别重试次数
        def capture_image_safe() -> Optional[np.ndarray]:
            """安全截图（失败返回 None）"""
            try:
                controller = context.tasker.controller
                controller.post_screencap().wait()
                time.sleep(0.03)
                return controller.cached_image
            except Exception:
                return None

        def is_wet_combined(plot_xy: Tuple[int, int], image=None) -> bool:
            """湿润判定：ColorMatch OR OCR(湿润)"""
            if image is None:
                image = capture_image_safe()
            if image is None:
                return False
            try:
                if self._is_plot_wet(context, plot_xy, image):
                    return True
            except Exception:
                pass
            try:
                return self._detect_soil_moisture_state(context, plot_xy, image) == "wet"
            except Exception:
                return False

        # 统一的坑位表：记录坐标、初始是否湿润、处理状态与失败原因
        plot_table: List[dict] = []
        start_image = capture_image_safe()

        for idx, plot in enumerate(FARM_PLOT_CENTERS, start=1):
            wet0 = is_wet_combined(plot, start_image)
            plot_table.append({
                "idx": idx,
                "pos": plot,
                "wet": wet0,           # 当前判定
                "done": wet0,          # 已完成（初始湿润直接完成）
                "attempts": 0,         # 浇水尝试次数（跨 pass）
                "fail_reason": None,   # 失败原因
            })

        print("\n[浇水] 开局坑位湿润判定：")
        for it in plot_table:
            print(f"  - 坑位{it['idx']}: {it['pos']} -> {'湿润(跳过)' if it['wet'] else '未湿润(待浇)'}")

        def process_plot(entry: dict, pass_name: str) -> bool:
            """处理单个坑位 entry；成功会更新 entry 并返回 True，否则记录 fail_reason 并返回 False。"""
            plot_idx = entry["idx"]
            plot_x, plot_y = entry["pos"]

            if entry.get("done"):
                return True

            print(f"\n[浇水] ({pass_name}) 坑位 {plot_idx}/16 -> {(plot_x, plot_y)}")

            # 任意时刻优先处理奖励弹窗（避免遮挡/误点）
            if self._check_reward_popup(context):
                print("[浇水] 检测到奖励弹窗，先领取奖励")
                self._run_sub_getreward(context)

            # 1) 移动到坑位附近：脚底点对齐坑位中心上方
            if not self._move_character_to_target(
                context,
                (plot_x, plot_y),
                target_offset=(0, -20),
                use_feet=True,
            ):
                entry["fail_reason"] = "move_failed"
                return False

            time.sleep(0.35)

            # 2) 进入交互态：点坑位 + 必要时向下走触发 UI
            for _ in range(max(1, moisture_retry)):
                try:
                    controller.post_click(int(plot_x), int(plot_y)).wait()
                except Exception:
                    pass
                time.sleep(0.2)

                # 若此时已经湿润（ColorMatch OR OCR），说明无需浇水
                frame = capture_image_safe()
                if is_wet_combined((plot_x, plot_y), frame):
                    entry["wet"] = True
                    entry["done"] = True
                    entry["fail_reason"] = None
                    print("[浇水] 判定该坑位已湿润（ColorMatch/OCR），跳过浇水")
                    return True

            # 走到容差范围内但还没触发 UI：向下走一段补救
            _joystick_nudge_down(context, times=1, duration_ms=500, wait_s=0.25)
            try:
                controller.post_click(int(plot_x), int(plot_y)).wait()
            except Exception:
                pass
            time.sleep(0.25)

            # 3) 浇水直到 ColorMatch 判定湿润
            water_count = 0
            while water_count < max_water_per_plot:
                if self._check_reward_popup(context):
                    print("[浇水] 检测到奖励弹窗，先领取奖励")
                    self._run_sub_getreward(context)

                found, box = self._check_water_button(context)
                if not found:
                    # 尝试点击坑位唤起按钮
                    try:
                        controller.post_click(int(plot_x), int(plot_y)).wait()
                    except Exception:
                        pass
                    time.sleep(0.2)
                    found, box = self._check_water_button(context)

                if not found:
                    water_count += 1
                    entry["attempts"] += 1
                    time.sleep(0.15)
                    continue

                if not self._click_water_button(context, box):
                    water_count += 1
                    entry["attempts"] += 1
                    time.sleep(0.15)
                    continue

                water_count += 1
                entry["attempts"] += 1
                time.sleep(WATER_CLICK_WAIT)

                frame = capture_image_safe()
                if is_wet_combined((plot_x, plot_y), frame):
                    # 从未湿润 -> 湿润：按原需求额外再浇 1 次补满进度条
                    found2, box2 = self._check_water_button(context)
                    if not found2:
                        try:
                            controller.post_click(int(plot_x), int(plot_y)).wait()
                        except Exception:
                            pass
                        time.sleep(0.2)
                        found2, box2 = self._check_water_button(context)

                    if found2:
                        print("[浇水] 已湿润，额外再浇 1 次补满进度条")
                        self._click_water_button(context, box2)
                        time.sleep(WATER_CLICK_WAIT)

                    entry["wet"] = True
                    entry["done"] = True
                    entry["fail_reason"] = None
                    return True

            entry["fail_reason"] = "not_wet_after_watering"
            return False

        # 第一遍：处理所有未完成坑位
        for it in plot_table:
            if not it.get("done"):
                process_plot(it, "第一遍")

        # 第二遍：只重试第一遍仍未完成的坑位
        remaining = [it for it in plot_table if not it.get("done")]
        if remaining:
            print("\n[浇水] 第一遍仍未完成的坑位：")
            for it in remaining:
                print(f"  - 坑位{it['idx']}: {it['pos']} reason={it.get('fail_reason')}")

            for it in remaining:
                process_plot(it, "第二遍")

        # 全部坑位完成后兜底领取奖励
        if self._check_reward_popup(context):
            print("[浇水] 结束前检测到奖励弹窗，领取奖励")
            self._run_sub_getreward(context)

        still_left = [it for it in plot_table if not it.get("done")]
        if still_left:
            print("\n[浇水] 尚有未浇水的坑位：")
            for it in still_left:
                print(f"  - 坑位{it['idx']}: {it['pos']} reason={it.get('fail_reason')} attempts={it.get('attempts')}")
            print("[浇水] 尚有未浇水的坑位")
            return False

        print("\n[浇水] 全农场浇水任务结束")
        return True
    
    def _run_sub_getreward(self, context: Context) -> bool:
        """
        调用 Sub_Getreward 节点获取奖励
        
        参数:
            context: MAA 上下文
            
        返回:
            bool: 是否成功
        """
        print("[FarmEventHandler] 调用 Sub_Getreward 获取奖励...")
        
        try:
            # 兼容不同 MaaFramework Python API：优先走 tasker，其次回退旧接口
            result = None
            if hasattr(context, "tasker"):
                tasker = context.tasker
                # 1) tasker.run_pipeline(name)
                if hasattr(tasker, "run_pipeline") and callable(getattr(tasker, "run_pipeline")):
                    result = tasker.run_pipeline("Sub_Getreward")
                # 2) tasker.post_pipeline(name).wait()
                elif hasattr(tasker, "post_pipeline") and callable(getattr(tasker, "post_pipeline")):
                    job = tasker.post_pipeline("Sub_Getreward")
                    result = job.wait() if job is not None else None

            print(f"[FarmEventHandler] Sub_Getreward 执行结果: {result}")
            return True
        except Exception as e:
            print(f"[FarmEventHandler] Sub_Getreward 执行失败: {e}")
            return False
    
    def _handle_waterwheel_repair(self, context: Context) -> bool:
        """
        处理水车修理事件
        
        流程:
        1. 使用 YOLOv8 检测角色位置
        2. 操纵角色移动到水车附近 [1035, 243]
        3. 检测并点击修理按钮
        4. 调用 Sub_Getreward 获取奖励
        
        参数:
            context: MAA 上下文
            
        返回:
            bool: 是否成功完成修理
        """
        print("\n" + "=" * 40)
        print("[水车修理] 开始处理水车修理事件")
        print("=" * 40)
        
        # 步骤 1: 移动角色到水车附近
        print("\n[水车修理] 步骤 1: 移动角色到水车位置...")
        if not self._move_character_to_target(context, WATERWHEEL_TARGET_POS):
            print("[水车修理] 移动到水车位置失败")
            return False
        
        # 等待界面稳定
        time.sleep(DEFAULT_WAIT)
        
        # 步骤 2: 检测修理按钮（原地尝试3次）
        print("\n[水车修理] 步骤 2: 检测修理按钮...")
        max_static_attempts = 3  # 原地检测次数
        found = False
        box = None
        
        print(f"[水车修理] 在当前位置尝试检测修理按钮（最多{max_static_attempts}次）...")
        for attempt in range(max_static_attempts):
            found, box = self._check_repair_button(context)
            if found:
                print(f"[水车修理] 第 {attempt + 1} 次检测到修理按钮！")
                break
            print(f"[水车修理] 第 {attempt + 1}/{max_static_attempts} 次，未检测到修理按钮")
            time.sleep(0.3)
        
        # 如果原地没找到，尝试向上移动并检测（最多3次）
        if not found:
            print("[水车修理] 原地未检测到修理按钮，尝试向上移动...")
            max_move_attempts = 3  # 最多向上移动次数
            controller = context.tasker.controller
            
            # 计算向上移动的摇杆终点（从中心向上滑动）
            joystick_up_end = (JOYSTICK_CENTER[0], JOYSTICK_CENTER[1] - JOYSTICK_RADIUS)
            move_duration = 500  # 每次移动持续时间（毫秒）
            
            for move_attempt in range(max_move_attempts):
                print(f"[水车修理] 向上移动第 {move_attempt + 1}/{max_move_attempts} 次...")
                
                # 执行向上移动 - 使用安全滑动函数
                x1 = int(JOYSTICK_CENTER[0])
                y1 = int(JOYSTICK_CENTER[1])
                x2 = int(joystick_up_end[0])
                y2 = int(joystick_up_end[1])
                
                if _safe_swipe(context, x1, y1, x2, y2, move_duration):
                    print(f"[水车修理] 摇杆向上操作完成")
                else:
                    print(f"[水车修理] 摇杆操作失败")
                
                # 等待移动完成
                time.sleep(0.5)
                
                # 检测修理按钮
                found, box = self._check_repair_button(context)
                if found:
                    print(f"[水车修理] 移动后第 {move_attempt + 1} 次检测到修理按钮！")
                    break
                print(f"[水车修理] 移动后第 {move_attempt + 1} 次仍未检测到修理按钮")
        
        # 如果所有尝试后仍未找到修理按钮，返回失败
        if not found:
            print("[水车修理] 尝试所有方法后仍未检测到修理按钮")
            return False
        
        # 步骤 3: 点击修理按钮
        print("\n[水车修理] 步骤 3: 点击修理按钮...")
        if not self._click_repair_button(context, box):
            print("[水车修理] 点击修理按钮失败")
            return False
        
        # 等待修理完成和奖励弹窗
        time.sleep(DEFAULT_WAIT)
        
        # 步骤 4: 获取奖励
        print("\n[水车修理] 步骤 4: 获取奖励...")
        self._run_sub_getreward(context)
        
        print("\n[水车修理] 水车修理完成！")
        return True
    
    def _handle_worm_catching(self, context: Context, params: dict) -> bool:
        """
        处理捉虫事件
        
        逻辑:
        1. 使用 YOLOv8 检测所有虫子和角色位置
        2. 找到距离角色最近的虫子
        3. 操纵角色移动到虫子附近
        4. 检测并点击捕捉按钮
        5. 等待捕捉动画完成
        6. 重复直到没有虫子
        7. 无虫子时每隔5秒检测一次共3轮
        
        参数:
            context: MAA 上下文
            params: 额外参数
            
        返回:
            bool: 是否成功完成捕捉
        """
        print("\n" + "=" * 40)
        print("[捉虫] 开始捉虫任务")
        print("=" * 40)
        
        caught_count = 0  # 已捕捉的虫子数量
        no_bugs_rounds = 0  # 连续无虫子的检测轮数
        
        while True:
            # ========== 全局终止条件：奖励弹窗/可执行 Sub_Getreward ==========
            # 用户要求：验证到奖励弹窗也就是 "Sub_Getreward" 节点成功执行意味着虫子已捉完，可直接结束捉虫任务
            if self._check_reward_popup(context):
                print("[捉虫] 检测到奖励弹窗（可领取奖励），直接结束捉虫任务")
                self._run_sub_getreward(context)
                return True

            # 检测场上所有目标
            girl_pos, bugs_list = self._detect_all_targets(context)
            
            # ========== 角色检测失败处理（避免卡死） ==========
            if girl_pos is None:
                # 情况A：能检测到虫子，但检测不到角色 -> 向上推动摇杆两次，每次再检查
                if len(bugs_list) > 0:
                    _joystick_nudge_up(context, times=2, duration_ms=300, wait_s=0.2)

                    # 重新检测一次
                    girl_pos, bugs_list = self._detect_all_targets(context)
                    if girl_pos is None:
                        print("[捉虫] 错误：检测到虫子但连续尝试后仍无法检测到角色，退出捉虫任务")
                        return False
                else:
                    # 情况B：角色和虫子都检测不到 -> 尝试执行 Sub_Getreward，能执行说明虫子已捉完
                    print("[捉虫] 角色与虫子均未检测到，尝试执行 Sub_Getreward 判断是否已结束...")
                    try:
                        if self._check_reward_popup(context):
                            self._run_sub_getreward(context)
                            return True
                        # 即便 OCR 未命中，也尝试直接 run_pipeline（用户要求以其成功作为最终判定）
                        if self._run_sub_getreward(context):
                            return True
                    except Exception:
                        pass

                    print("[捉虫] 错误：角色与虫子均未检测到且无法执行 Sub_Getreward，退出捉虫任务")
                    return False
            
            # 检查是否有虫子
            if len(bugs_list) == 0:
                no_bugs_rounds += 1
                print(f"[捉虫] 未检测到虫子 (第 {no_bugs_rounds}/{NO_BUGS_CHECK_ROUNDS} 轮)")
                
                # 检查是否匹配到 Sub_Getreward（奖励弹窗）-> 直接结束
                if self._check_reward_popup(context):
                    print("[捉虫] 检测到奖励弹窗，捉虫任务完成！")
                    self._run_sub_getreward(context)
                    return True
                
                if no_bugs_rounds >= NO_BUGS_CHECK_ROUNDS:
                    print(f"[捉虫] 连续 {NO_BUGS_CHECK_ROUNDS} 轮未检测到虫子，任务完成")
                    break
                
                print(f"[捉虫] 等待 {NO_BUGS_CHECK_INTERVAL} 秒后重新检测...")
                time.sleep(NO_BUGS_CHECK_INTERVAL)
                continue
            
            # 有虫子，重置无虫子计数器
            no_bugs_rounds = 0
            
            # 找到最近的虫子
            nearest_bug, distance = self._find_nearest_bug(girl_pos, bugs_list)
            print(f"[捉虫] 检测到 {len(bugs_list)} 只虫子，最近的距离: {distance:.1f}px")
            print(f"[捉虫] 角色位置: {girl_pos}, 目标虫子: {nearest_bug}")
            
            # 捕捉最近的虫子
            success = self._catch_single_worm(context, nearest_bug)
            if success:
                caught_count += 1
                print(f"[捉虫] ✓ 成功捕捉第 {caught_count} 只虫子！")
            else:
                print("[捉虫] ✗ 捕捉失败，尝试下一只")
            
            # 短暂等待后继续
            time.sleep(0.5)
        
        print(f"\n[捉虫] 捉虫任务结束，共捕捉 {caught_count} 只虫子")
        return caught_count > 0
    
    def _detect_all_targets(self, context: Context, image=None) -> Tuple[Optional[tuple], List[tuple]]:
        """
        使用 YOLOv8 模型同时检测角色和所有虫子位置
        
        参数:
            context: MAA 上下文
            image: 屏幕截图（可选）
            
        返回:
            Tuple[Optional[tuple], List[tuple]]: (角色位置, 虫子位置列表)
                - 角色位置: (x, y) 或 None
                - 虫子位置列表: [(x, y), ...]
        """
        if image is None:
            try:
                context.tasker.controller.post_screencap().wait()
                time.sleep(0.05)
                image = context.tasker.controller.cached_image
            except Exception as e:
                print(f"[捉虫] 截图失败: {e}")
                return None, []
        
        # 更新调试画面
        if DEBUG_ENABLED and image is not None:
            update_debug_frame(image)
        
        girl_pos = None
        bugs_list = []
        debug_detections = []
        
        # 检测所有目标（不指定 expected，获取所有类别）
        reco_result = context.run_recognition(
            "Farm_DetectAll",
            image,
            {
                "Farm_DetectAll": {
                    "recognition": "NeuralNetworkDetect",
                    "model": FARMING_MODEL_PATH,
                    "threshold": YOLO_THRESHOLD,
                    "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
                }
            }
        )
        
        if reco_result and hasattr(reco_result, 'all_results') and reco_result.all_results:
            for result in reco_result.all_results:
                box = result.box
                center = (box[0] + box[2] // 2, box[1] + box[3] // 2)
                confidence = result.score if hasattr(result, 'score') else 0.0
                label = result.label if hasattr(result, 'label') else 'unknown'
                cls_index = result.cls_index if hasattr(result, 'cls_index') else -1
                
                if cls_index == YOLO_GIRL_INDEX or label == 'girl':
                    girl_pos = center
                    debug_detections.append({
                        "label": "girl",
                        "box": list(box),
                        "confidence": confidence
                    })
                elif cls_index == YOLO_BUGS_INDEX or label == 'bugs':
                    bugs_list.append(center)
                    debug_detections.append({
                        "label": "bugs",
                        "box": list(box),
                        "confidence": confidence
                    })
        
        # 更新调试可视化
        if DEBUG_ENABLED:
            update_debug_detections(debug_detections)
        
        return girl_pos, bugs_list
    
    def _find_nearest_bug(self, girl_pos: tuple, bugs_list: List[tuple]) -> Tuple[tuple, float]:
        """
        找到距离角色最近的虫子
        
        参数:
            girl_pos: 角色位置 (x, y)
            bugs_list: 虫子位置列表 [(x, y), ...]
            
        返回:
            Tuple[tuple, float]: (最近虫子位置, 距离)
        """
        if not bugs_list:
            return None, float('inf')
        
        # 使用 numpy 计算距离
        girl_array = np.array(girl_pos)
        bugs_array = np.array(bugs_list)
        
        # 计算所有虫子到角色的距离
        distances = np.sqrt(np.sum((bugs_array - girl_array) ** 2, axis=1))
        
        # 找到最近的虫子
        nearest_idx = np.argmin(distances)
        nearest_bug = bugs_list[nearest_idx]
        nearest_distance = distances[nearest_idx]
        
        return nearest_bug, nearest_distance
    
    def _check_catch_button(self, context: Context, image=None) -> Tuple[bool, Optional[list]]:
        """
        检测捕捉按钮是否出现
        
        参数:
            context: MAA 上下文
            image: 屏幕截图（可选）
            
        返回:
            Tuple[bool, Optional[list]]: (是否出现, box)
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            image = context.tasker.controller.cached_image
        
        reco_result = context.run_recognition(
            "Farm_CheckCatchButton",
            image,
            {
                "Farm_CheckCatchButton": {
                    "recognition": "TemplateMatch",
                    "template": CATCH_BUTTON_TEMPLATE,
                    "roi": CATCH_BUTTON_ROI,
                    "threshold": 0.7,
                }
            }
        )
        
        if _reco_hit(reco_result):
            box = _reco_box(reco_result)
            return (True, box)
        
        return (False, None)
    
    def _check_reward_popup(self, context: Context) -> bool:
        """
        检查是否出现奖励弹窗（Sub_Getreward）
        
        参数:
            context: MAA 上下文
            
        返回:
            bool: 是否出现奖励弹窗
        """
        context.tasker.controller.post_screencap().wait()
        image = context.tasker.controller.cached_image
        
        reco_result = context.run_recognition(
            "Farm_CheckReward",
            image,
            {
                "Farm_CheckReward": {
                    "recognition": "OCR",
                    "expected": ["获得物品", "获得道具"],
                    "roi": [78, 15, 1014, 254],
                }
            }
        )
        
        return _reco_hit(reco_result)
    
    def _catch_single_worm(self, context: Context, worm_pos: tuple) -> bool:
        """
        捕捉单个虫子
        
        流程:
        1. 移动角色到虫子附近
        2. 检测捕捉按钮
        3. 点击捕捉按钮
        4. 等待捕捉动画完成
        
        参数:
            context: MAA 上下文
            worm_pos: 虫子位置 (x, y)
            
        返回:
            bool: 是否成功捕捉
        """
        print(f"[捉虫] 开始追捕虫子: {worm_pos}")
        
        # 设置调试目标
        if DEBUG_ENABLED:
            set_debug_target(worm_pos)
            set_debug_info(f"Chasing bug at {worm_pos}")
        
        # 移动到虫子附近（使用较大的容差，因为虫子会移动）
        start_time = time.time()
        max_chase_time = 10.0  # 最大追捕时间
        
        while time.time() - start_time < max_chase_time:
            # === 一轮循环严格按：检查捕捉按钮 -> YOLO -> 移动1次 -> 循环 ===
            # 先截图一次，本轮模板匹配与 YOLO 共用同一张图，减少延迟
            try:
                context.tasker.controller.post_screencap().wait()
                time.sleep(0.02)  # 轻微延迟，确保 cached_image 刷新
                loop_image = context.tasker.controller.cached_image
            except Exception as e:
                print(f"[捉虫] 截图失败: {e}")
                time.sleep(0.1)
                continue

            # ========== 1) 循环中断点：优先检查捕捉按钮 ==========
            # 任何 YOLO/移动之前先模板匹配按钮；命中则点击、等动画，然后回到循环开头继续（continue）
            found, box = self._check_catch_button(context, loop_image)
            if found:
                print("[捉虫] 【中断】检测到捕捉按钮！立即点击并回到循环")

                center = _box_center(box)
                print(f"[捉虫] 点击捕捉按钮: {center}")
                try:
                    context.tasker.controller.post_click(int(center[0]), int(center[1])).wait()
                except Exception as e:
                    print(f"[捉虫] 点击捕捉按钮异常: {e}")

                print(f"[捉虫] 等待捕捉动画 ({CATCH_ANIMATION_WAIT}秒)...")
                time.sleep(CATCH_ANIMATION_WAIT)

                # 若弹出奖励弹窗，视为捕捉成功
                if self._check_reward_popup(context):
                    if DEBUG_ENABLED:
                        set_debug_target(None)
                        set_debug_info("Bug caught!")
                    return True

                # 未出现奖励弹窗：继续下一轮（可能点空/未捕到/虫子仍在）
                continue

            # ========== 2) YOLO 检测（使用同一张截图） ==========
            girl_pos, bugs_list = self._detect_all_targets(context, loop_image)

            if girl_pos is None:
                # 用户要求：检测不到角色但能检测到虫子 -> 向上推动两次，每次再检查；仍失败则退出
                if len(bugs_list) > 0:
                    _joystick_nudge_up(context, times=2, duration_ms=300, wait_s=0.2)
                    # 重新截图并再检测一次
                    try:
                        context.tasker.controller.post_screencap().wait()
                        time.sleep(0.02)
                        retry_image = context.tasker.controller.cached_image
                    except Exception as e:
                        print(f"[捉虫] 截图失败: {e}")
                        return False

                    girl_pos, bugs_list = self._detect_all_targets(context, retry_image)
                    if girl_pos is None:
                        print("[捉虫] 错误：检测到虫子但连续尝试后仍无法检测到角色，退出本次追捕")
                        return False
                else:
                    # 用户要求：角色和虫子都检测不到 -> 尝试执行 Sub_Getreward，能执行说明捉完了
                    print("[捉虫] 角色与虫子均未检测到，尝试执行 Sub_Getreward 判断是否已结束...")
                    if self._check_reward_popup(context) or self._run_sub_getreward(context):
                        if DEBUG_ENABLED:
                            set_debug_target(None)
                            set_debug_info("Bug caught!")
                        return True
                    print("[捉虫] 错误：角色与虫子均未检测到且无法执行 Sub_Getreward，退出本次追捕")
                    return False
            
            # 如果没有虫子了，可能已经被捕捉或者跑了
            if len(bugs_list) == 0:
                print("[捉虫] 虫子消失了，检查是否有奖励弹窗...")
                if self._check_reward_popup(context) or self._run_sub_getreward(context):
                    print("[捉虫] 检测到奖励弹窗/成功执行 Sub_Getreward，认为虫子已捉完")
                    if DEBUG_ENABLED:
                        set_debug_target(None)
                    return True
                # 可能虫子暂时被遮挡或者跑到视野外
                time.sleep(0.5)
                continue
            
            # 找到最近的虫子（虫子可能移动了位置）
            nearest_bug, distance = self._find_nearest_bug(girl_pos, bugs_list)
            
            # 更新调试目标
            if DEBUG_ENABLED:
                set_debug_target(nearest_bug)
                set_debug_info(f"Distance: {distance:.1f}px")
            
            # ========== 3) 移动 1 次 ==========
            joystick_end = _calculate_joystick_direction(girl_pos, nearest_bug)

            x1 = int(JOYSTICK_CENTER[0])
            y1 = int(JOYSTICK_CENTER[1])
            x2 = int(joystick_end[0])
            y2 = int(joystick_end[1])

            if not _safe_swipe(context, x1, y1, x2, y2, 300):
                print("[捉虫] 摇杆操作失败，跳过本次移动")
                time.sleep(MOVE_CHECK_INTERVAL)
                continue

            # ========== 4) 循环（下一轮会先检查捕捉按钮） ==========
            time.sleep(MOVE_CHECK_INTERVAL)
        
        print("[捉虫] 追捕超时")
        if DEBUG_ENABLED:
            set_debug_target(None)
            set_debug_info("Chase timeout")
        return False
    
    def _detect_bugs_positions(self, context: Context, image=None) -> List[tuple]:
        """
        使用 YOLOv8 模型检测所有虫子位置
        
        参数:
            context: MAA 上下文
            image: 屏幕截图（可选）
            
        返回:
            List[tuple]: 虫子位置列表 [(x, y), ...]
        """
        _, bugs_list = self._detect_all_targets(context, image)
        return bugs_list


# 为了保持 API 兼容性，提供一个简化的别名类
class FarmWaterwheelRepair(FarmEventHandler):
    """
    水车修理专用 Action（简化调用）
    
    Pipeline 调用示例:
    {
        "custom_action": "FarmWaterwheelRepair"
    }
    """
    
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        # 强制设置 event_type 为 waterwheel
        argv.custom_action_param = json.dumps({"event_type": "waterwheel"})
        return super().run(context, argv)


class FarmWormCatching(FarmEventHandler):
    """
    捉虫专用 Action（简化调用）
    
    Pipeline 调用示例:
    {
        "custom_action": "FarmWormCatching"
    }
    """
    
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        # 强制设置 event_type 为 worms
        params = _parse_param(argv.custom_action_param)
        params["event_type"] = "worms"
        argv.custom_action_param = json.dumps(params)
        return super().run(context, argv)


class FarmWateringAll(FarmEventHandler):
    """
    全农场浇水专用 Action（简化调用）

    Pipeline 调用示例:
    {
        "custom_action": "FarmWateringAll"
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        params = _parse_param(argv.custom_action_param)
        params["event_type"] = "watering"
        argv.custom_action_param = json.dumps(params)
        return super().run(context, argv)
