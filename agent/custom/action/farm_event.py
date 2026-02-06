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
import builtins
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Dict, Any
import numpy as np
import cv2

# 硬样本录制器：识别失败时保存前后若干帧（仅图片）
from .screenshot_collector import HardCaseFrameRecorder

# Tkinter 和 PIL 用于调试窗口（在某些环境中可能不可用，例如嵌入式 Python）
try:
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    _HAS_TKINTER = True
except Exception:
    _HAS_TKINTER = False

# ==================== 调试配置 ====================
# 全局调试开关 - 设置为 True 开启调试功能，False 关闭所有调试功能
# 注意：Tkinter GUI 在 MaaFramework Agent 环境中可能导致崩溃
# 如果遇到"调试适配器意外终止"错误，请设置为 False
DEBUG_ENABLED = True 

# 调试显示模式：
# - "opencv": 使用 OpenCV cv2.imshow 显示 ADB 截图，并用 cv2 直接画框（推荐）
# - "tk": 旧版 Tkinter+PIL 合成显示（开销大，仅保留兼容）
DEBUG_VIEW_MODE = "opencv"

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

# ==================== focus 回调日志 ====================
_RAW_PRINT = builtins.print
_FOCUS_CONTEXT = None
_FOCUS_CONTEXT_LOCK = threading.Lock()
_FOCUS_MESSAGE_TYPE = "Custom.Focus"


def _set_focus_context(context: Optional[Context]) -> Optional[Context]:
    """设置当前 focus 回调上下文，返回之前的上下文。"""
    global _FOCUS_CONTEXT
    with _FOCUS_CONTEXT_LOCK:
        prev = _FOCUS_CONTEXT
        _FOCUS_CONTEXT = context
    return prev


def _get_focus_context() -> Optional[Context]:
    """获取当前 focus 回调上下文。"""
    with _FOCUS_CONTEXT_LOCK:
        return _FOCUS_CONTEXT


def _emit_focus_callback_by_pipeline(message: str, context: Optional[Context]) -> bool:
    """通过临时 pipeline 发送 focus 消息（展示到 UI 终端）。"""
    if not context:
        return False
    tasker = getattr(context, "tasker", None)
    if tasker is None:
        return False

    pipeline = {
        "_Focus_Callback": {
            "recognition": "DirectHit",
            "action": "DoNothing",
            "focus": {"Node.Recognition.Succeeded": message},
            "max_hit": 1,
        }
    }

    try:
        if hasattr(tasker, "post_pipeline") and callable(getattr(tasker, "post_pipeline")):
            job = tasker.post_pipeline(pipeline)
            if job is not None and hasattr(job, "wait"):
                job.wait()
            return True
        if hasattr(tasker, "run_pipeline") and callable(getattr(tasker, "run_pipeline")):
            return bool(tasker.run_pipeline(pipeline))
    except Exception:
        return False
    return False


def _emit_focus_callback(message: str, context: Optional[Context] = None) -> None:
    """尽量发送 focus 回调，不影响主流程。"""
    if message is None:
        return
    msg = str(message)
    if not msg:
        return
    ctx = context or _get_focus_context()

    # 优先通过临时 pipeline 发送到 UI 终端
    if _emit_focus_callback_by_pipeline(msg, ctx):
        return

    # 兜底：通过 Toolkit.report_message 走官方回调通道
    # 《2.3-回调协议》约定：type 使用 "Custom.*" 前缀，detail 为 JSON 字符串。
    detail = {"focus": msg}
    detail_json = json.dumps(detail, ensure_ascii=False)
    try:
        from maa.toolkit import Toolkit

        Toolkit.report_message(_FOCUS_MESSAGE_TYPE, detail_json)
    except Exception:
        return


def _focus_print(*args, **kwargs):
    """保持 print 行为，并附加 focus 回调消息。"""
    _RAW_PRINT(*args, **kwargs)
    try:
        sep = kwargs.get("sep", " ")
        message = sep.join(str(arg) for arg in args)
        if message:
            _emit_focus_callback(message)
    except Exception:
        pass


# 覆盖模块内 print，使所有日志附带 focus 回调
print = _focus_print

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

# 风车修理路径点
WINDMILL_TARGET_POS = (1118, 468)  # 风车附近目标位置（靠近修理按钮）
WINDMILL_ENTRY_POS = (1033, 262)   # 右侧栅栏上方入口点（再往右为空气墙）
FARM_WATERING_START_POS = (627, 287)  # 农场右上方默认站位，用于修理后回到浇水起点

# 浇水相关
WATER_BUTTON_ROI = [1022, 435, 206, 244]  # 右下角浇水按钮区域
WATER_BUTTON_TEMPLATE = f"{_FARM_IMG_DIR}/浇水按钮.png"
WATER_BUTTON_THRESHOLD = 0.7
WATER_CLICK_WAIT = 0.8  # 点击浇水按钮后的等待时间（秒）
SUB_GETREWARD_TARGET = (621, 616)  # 与 utils.json 的 Sub_Getreward 目标保持一致

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
# 土壤湿度 OCR UI 的精确 ROI（基于用户标注）
# 对于中心点 (752, 348)，[651, 235, 186, 124] 能完整覆盖“湿润/缺水/正常”文本区域。
# 推导得到的通用偏移关系：
#   ROI 宽度:  186, 高度: 124
#   ROI 中心相对坑位中心: Δx ≈ -8, Δy ≈ -51
MOISTURE_OCR_ROI_W = 186
MOISTURE_OCR_ROI_H = 124
MOISTURE_OCR_OFFSET_X = -8
MOISTURE_OCR_OFFSET_Y = -51

# 坑位“湿润”检测（TemplateMatch，替代 ColorMatch）
# 通过匹配“湿润状态下坑位底部水面”的模板判断是否湿润：
# - 模板：farm/水坑截图（判断湿润）.png
# - 位置：位于坑位中心点下方约 20px 左右
# - 由于坑位周围都是水，可能出现重复命中；只要命中一次即可认为“湿润”
PLOT_WET_WATER_TEMPLATE = f"{_FARM_IMG_DIR}/水坑截图（判断湿润）.png"
PLOT_WET_TEMPLATE_OFFSET_Y = 20
PLOT_WET_TEMPLATE_MARGIN = 50  # 以 (center_x, center_y+offset) 为中心，向外扩张的 ROI 半径
PLOT_WET_TEMPLATE_THRESHOLD = 0.7

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
MOVE_TOLERANCE = 20  # 到达目标位置的容差（像素）
MOVE_NEAR_DISTANCE = 60  # 进入近距离阈值（像素）
MOVE_CHECK_INTERVAL = 0.1  # 移动过程中检测间隔（秒）- 降低以提高帧率
MOVE_TIMEOUT = 15  # 移动超时时间（秒）
SWIPE_DURATION = 500  # 摇杆滑动持续时间（毫秒）
SWIPE_DURATION_FINE = 250  # 近距离精定位摇杆持续时间（毫秒）
NEAR_DEBOUNCE_WAIT = 0.2  # 进入近距离后的防抖等待（秒）
GIRL_POS_EMA_ALPHA = 0.35  # 角色脚底点 EMA 平滑系数
GIRL_PREDICT_STEP = 10     # 角色丢失时沿移动方向预测步长（像素）
GIRL_POS_EMA_MAX_DELTA = 35  # EMA 与原始位置偏差过大时回退原始位置（像素）


@dataclass
class YoloDet:
    """单个 YOLO 检测结果（与 MaaFramework reco_result 字段对齐）。"""
    box: List[int]
    cls_index: int
    score: float
    label: str = ""


@dataclass
class YoloFrameResults:
    """单帧 YOLO 解析结果（用于复用，避免重复推理）。"""
    all: List[YoloDet] = field(default_factory=list)
    girls: List[YoloDet] = field(default_factory=list)
    bugs: List[YoloDet] = field(default_factory=list)
    girl_best: Optional[YoloDet] = None


@dataclass
class FrameContext:
    """
    单帧上下文：同一张截图内复用所有识别结果，确保“截一次图 -> 跑完必要检测 -> 动作/循环”。\n
    - image: 本帧截图（BGR）\n
    - cache: 同帧缓存（包含 YOLO、OCR、TemplateMatch 等 run_recognition 结果）\n
    """
    image: np.ndarray
    cache: Dict[str, Any] = field(default_factory=dict)
    t_capture: float = field(default_factory=time.time)

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


def _rects_intersect(a: List[int], b: List[int]) -> bool:
    """
    判断两个矩形是否相交。
    矩形格式为 [x, y, w, h]，坐标基于整张截图。
    """
    if a is None or b is None:
        return False
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    # 分离轴：若在任一轴上无重叠，则不相交
    if ax2 <= bx or bx2 <= ax:
        return False
    if ay2 <= by or by2 <= ay:
        return False
    return True


def _build_moisture_ocr_roi(plot_pos: Tuple[int, int]) -> List[int]:
    """
    根据坑位中心构建土壤湿度 OCR 的精确 ROI。
    
    依据用户提供样本：
        坑位中心 (752, 348) -> ROI [651, 235, 186, 124]
    推导出通用关系:
        ROI 宽度:  MOISTURE_OCR_ROI_W
        ROI 高度:  MOISTURE_OCR_ROI_H
        ROI 中心 = 坑位中心 + (MOISTURE_OCR_OFFSET_X, MOISTURE_OCR_OFFSET_Y)
    """
    px, py = int(plot_pos[0]), int(plot_pos[1])
    w, h = int(MOISTURE_OCR_ROI_W), int(MOISTURE_OCR_ROI_H)
    cx = px + MOISTURE_OCR_OFFSET_X
    cy = py + MOISTURE_OCR_OFFSET_Y
    x = cx - w // 2
    y = cy - h // 2
    return _clamp_roi([x, y, w, h])


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
        # 使用字符串类型注解以避免在缺失 Tkinter 的环境下导入失败
        self._root: Optional["tk.Tk"] = None
        self._canvas: Optional["tk.Canvas"] = None
        self._photo_image: Optional["ImageTk.PhotoImage"] = None
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


class OpenCVDebugViewer:
    """
    OpenCV 调试查看器：显示 ADB 截图并用 cv2 直接画框画字。
    作为默认的调试可视化方案。
    """

    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._current_image: Optional[np.ndarray] = None
        self._detections: List[dict] = []
        self._target_pos: Optional[Tuple[int, int]] = None
        self._info_text: str = ""
        self._frame_count = 0

        self._window_name = "Farm Debug Viewer - OpenCV"
        self._last_fps_time = time.time()
        self._show_frames = 0
        self._show_fps = 0.0

    def push_frame(self, image: np.ndarray):
        if image is None or getattr(image, "size", 0) == 0:
            return
        with self._lock:
            self._current_image = image.copy()
            self._frame_count += 1

    def push_detections(self, detections: List[dict]):
        with self._lock:
            self._detections = detections.copy() if detections else []

    def set_target(self, pos: Optional[Tuple[int, int]]):
        with self._lock:
            self._target_pos = pos

    def set_info(self, text: str):
        with self._lock:
            self._info_text = text or ""

    def start(self):
        if self._thread and self._thread.is_alive():
            print("[OpenCVDebugViewer] 已在运行")
            return
        print("[OpenCVDebugViewer] 启动...")
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, name="OpenCVDebugViewer", daemon=True)
        self._thread.start()
        time.sleep(0.2)

    def stop(self):
        print("[OpenCVDebugViewer] 请求停止...")
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass
        print("[OpenCVDebugViewer] 已停止")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        except Exception:
            pass

        while not self._stop_flag.is_set():
            with self._lock:
                img = None if self._current_image is None else self._current_image.copy()
                dets = self._detections.copy()
                target = self._target_pos
                info = self._info_text
                frame_count = self._frame_count

            if img is None:
                time.sleep(0.05)
                continue

            for det in dets:
                label = det.get("label", "unknown")
                box = det.get("box", [0, 0, 0, 0])
                conf = float(det.get("confidence", 0.0) or 0.0)
                color = DEBUG_COLORS.get(label, (0, 255, 0))  # RGB
                bgr = (int(color[2]), int(color[1]), int(color[0]))
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(img, (x, y), (x + w, y + h), bgr, 2)
                text = f"{label} {conf:.2f}" if conf > 0 else str(label)
                cv2.putText(img, text, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

            if target:
                tx, ty = int(target[0]), int(target[1])
                tcolor = DEBUG_COLORS.get("target", (255, 0, 255))
                tbgr = (int(tcolor[2]), int(tcolor[1]), int(tcolor[0]))
                cv2.drawMarker(img, (tx, ty), tbgr, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            self._show_frames += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self._show_fps = self._show_frames / (now - self._last_fps_time)
                self._show_frames = 0
                self._last_fps_time = now

            header = f"OpenCVFPS:{self._show_fps:.1f} PushFrameCount:{frame_count} Dets:{len(dets)}"
            if info:
                header += f" | {info}"
            cv2.putText(img, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            try:
                cv2.imshow(self._window_name, img)
                cv2.waitKey(1)
            except Exception:
                time.sleep(0.05)


# ==================== 全局调试实例管理 ====================

_debug_viewer = None


def get_debug_viewer():
    """获取当前调试查看器实例"""
    return _debug_viewer


def start_debug_viewer(context: Optional[Context] = None):
    """启动调试查看器（非阻塞）"""
    global _debug_viewer
    if not DEBUG_ENABLED:
        return None
    if _debug_viewer and _debug_viewer.is_running():
        return _debug_viewer

    # 统一以 OpenCV 为主，必要时可切到 Tk 模式
    mode = (DEBUG_VIEW_MODE or "opencv").lower()

    if mode == "tk" and _HAS_TKINTER:
        print("[DebugViewer] 使用 Tkinter 调试窗口")
        _debug_viewer = TkDebugViewer()
    else:
        if mode not in ("opencv", "tk"):
            print(f"[DebugViewer] 未知调试模式 {mode}，已回退到 OpenCV")
        _debug_viewer = OpenCVDebugViewer()

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
    return start_debug_viewer(context)


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
    - event_type: "windmill" - 风车修理
    - event_type: "watering" - 全农场浇水
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
        prev_focus_context = _set_focus_context(context)

        print("=" * 60)
        print("[FarmEventHandler] 农场事件处理器启动")
        
        # 启动调试可视化（数据推送模式）
        if DEBUG_ENABLED:
            print("[FarmEventHandler] 调试模式已启用，启动 Tkinter 调试窗口...")
            start_debug_viewer(context)  # 非阻塞，在独立线程运行（绑定当前 ADB 设备）
            time.sleep(0.5)  # 等待窗口创建
            print("[FarmEventHandler] 调试窗口已启动")
        
        # 解析参数
        params = _parse_param(argv.custom_action_param)
        event_type = params.get("event_type", "waterwheel")

        # 硬样本录制器（用于保存“识别丢失/误检导致动作失败”时的前后帧）
        # 低开销：常态仅保留最近 5 帧的环形缓冲；仅在失败触发时写盘。
        self._hardcase_recorder = HardCaseFrameRecorder(
            prefix="farm_fix",
            save_dir="training/hard cases",
            img_format="jpg",
            quality=95,
            pre_frames=5,
            post_frames=5,
        )
        
        print(f"[FarmEventHandler] 事件类型: {event_type}")
        print(f"[FarmEventHandler] 完整参数: {params}")
        
        try:
            if event_type == "waterwheel":
                # 水车修理
                success = self._handle_waterwheel_repair(context)
            elif event_type == "windmill":
                # 风车修理
                success = self._handle_windmill_repair(context)
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
            _set_focus_context(prev_focus_context)

    def _hardcase_push(self, image: np.ndarray):
        """向硬样本录制器推入一帧（不影响主流程）。"""
        try:
            if hasattr(self, "_hardcase_recorder") and self._hardcase_recorder is not None:
                self._hardcase_recorder.push_frame(image)
        except Exception:
            pass

    def _hardcase_trigger_and_capture_post(
        self,
        context: Context,
        *,
        reason: str,
        post_frames: int = 5,
        delay_s: float = 0.03,
    ):
        """
        触发保存“前后帧”并同步补齐后续帧（避免失败后立刻 return 导致拿不到 post 帧）。
        仅保存图片到 training/hard cases。
        """
        try:
            if not hasattr(self, "_hardcase_recorder") or self._hardcase_recorder is None:
                return
            self._hardcase_recorder.trigger(reason=reason)

            # 同步补齐后续帧：失败发生后，额外截取 post_frames 次
            for _ in range(max(0, int(post_frames))):
                try:
                    controller = context.tasker.controller
                    controller.post_screencap().wait()
                    time.sleep(max(0.0, float(delay_s)))
                    img = controller.cached_image
                    if img is not None:
                        self._hardcase_recorder.push_frame(img)
                except Exception:
                    break
        except Exception:
            pass

    def _run_recognition_cached(
        self,
        context: Context,
        *,
        task_name: str,
        image: np.ndarray,
        pipeline_override: dict,
        frame: Optional[FrameContext] = None,
    ):
        """
        同帧缓存版 run_recognition：在同一张 image 上重复调用相同识别时，直接复用结果。\n
        说明：缓存粒度以 (task_name + pipeline_override) 为 key；只在 frame 提供时启用缓存。
        """
        if frame is None:
            return context.run_recognition(task_name, image, pipeline_override)

        try:
            key = f"reco::{task_name}::{json.dumps(pipeline_override, sort_keys=True, ensure_ascii=False)}"
        except Exception:
            # 兜底：不可序列化就不缓存
            return context.run_recognition(task_name, image, pipeline_override)

        if key in frame.cache:
            return frame.cache[key]

        res = context.run_recognition(task_name, image, pipeline_override)
        frame.cache[key] = res
        return res

    def _yolo_detect_all_once(self, context: Context, frame: FrameContext) -> YoloFrameResults:
        """
        单帧仅一次 YOLO 推理：不传 expected，让 NeuralNetworkDetect 返回所有类别的结果。\n
        解析策略：优先 reco_result.all_results；否则回退 best_result。\n
        结果会缓存在 frame.cache['yolo']，确保同一帧复用不重复推理。
        """
        cached = frame.cache.get("yolo")
        if isinstance(cached, YoloFrameResults):
            return cached

        # 统计：用于验证“每帧仅一次 YOLO 推理调用”
        try:
            self._yolo_invoke_total = getattr(self, "_yolo_invoke_total", 0) + 1
            if DEBUG_ENABLED:
                print(f"[FarmEventHandler][YOLO] invoked total={self._yolo_invoke_total}")
        except Exception:
            pass

        pipeline_override = {
            "Farm_YOLO_All": {
                "recognition": "NeuralNetworkDetect",
                "model": FARMING_MODEL_PATH,
                "labels": YOLO_LABELS,
                "threshold": YOLO_THRESHOLD,
                "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
                "order_by": "Score",
                "index": 0,
            }
        }

        reco_result = self._run_recognition_cached(
            context,
            task_name="Farm_YOLO_All",
            image=frame.image,
            pipeline_override=pipeline_override,
            frame=frame,
        )

        parsed = YoloFrameResults()

        if not _reco_hit(reco_result):
            frame.cache["yolo"] = parsed
            return parsed

        det_list = []
        # 多框优先：all_results
        if hasattr(reco_result, "all_results") and reco_result.all_results:
            try:
                for r in reco_result.all_results:
                    box = getattr(r, "box", None)
                    if box is None:
                        continue
                    cls_index = int(getattr(r, "cls_index", -1))
                    score = float(getattr(r, "score", 0.0))
                    label = str(getattr(r, "label", "")) if getattr(r, "label", None) is not None else ""
                    det_list.append(YoloDet(box=list(box), cls_index=cls_index, score=score, label=label))
            except Exception:
                det_list = []

        # 兜底：仅 best_result
        if not det_list and hasattr(reco_result, "best_result") and reco_result.best_result:
            r = reco_result.best_result
            box = getattr(r, "box", None)
            if box is not None:
                cls_index = int(getattr(r, "cls_index", -1))
                score = float(getattr(r, "score", 0.0))
                label = str(getattr(r, "label", "")) if getattr(r, "label", None) is not None else ""
                det_list.append(YoloDet(box=list(box), cls_index=cls_index, score=score, label=label))

        parsed.all = det_list
        parsed.girls = [d for d in det_list if d.cls_index == YOLO_GIRL_INDEX or d.label == "girl"]
        parsed.bugs = [d for d in det_list if d.cls_index == YOLO_BUGS_INDEX or d.label == "bugs"]
        if parsed.girls:
            parsed.girl_best = max(parsed.girls, key=lambda d: d.score)

        frame.cache["yolo"] = parsed
        return parsed
    
    def _detect_girl_position(self, context: Context, image=None, *, frame: Optional[FrameContext] = None) -> tuple:
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
        
        # 保存调试截图到文件（仅第一次）
        if DEBUG_FILE_MODE and not hasattr(self, '_debug_screenshot_saved'):
            import os
            debug_dir = os.path.join(_PROJECT_ROOT, DEBUG_OUTPUT_DIR)
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "input_screenshot.png")
            cv2.imwrite(debug_path, image)
            print(f"[FarmEventHandler] 调试截图已保存到: {debug_path}")
            self._debug_screenshot_saved = True

        frame_ctx = frame if frame is not None else FrameContext(image=image)
        yolo = self._yolo_detect_all_once(context, frame_ctx)

        if yolo.girl_best is None:
            print("[FarmEventHandler] 未检测到角色")
            if DEBUG_ENABLED:
                update_debug_detections([])
                update_debug_frame()
            return None

        box = yolo.girl_best.box
        center = _box_center(box)
        confidence = yolo.girl_best.score
        print(f"[FarmEventHandler] [OK] 检测到角色 (girl): box={box}, center={center}, confidence={confidence:.3f}")

        if DEBUG_ENABLED:
            update_debug_detections([{"label": "girl", "box": list(box), "confidence": confidence}])

        return center

    def _detect_girl_box_and_center(
        self,
        context: Context,
        image=None,
        *,
        frame: Optional[FrameContext] = None,
    ) -> Tuple[Optional[list], Optional[tuple]]:
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

        frame_ctx = frame if frame is not None else FrameContext(image=image)
        yolo = self._yolo_detect_all_once(context, frame_ctx)

        if yolo.girl_best is None:
            if DEBUG_ENABLED:
                update_debug_detections([])
            return None, None

        box = yolo.girl_best.box
        center = _box_center(box)
        confidence = yolo.girl_best.score

        if DEBUG_ENABLED:
            update_debug_detections([{"label": "girl", "box": list(box), "confidence": confidence}])

        return box, center
    
    def _detect_all_objects(self, context: Context, image=None, *, frame: Optional[FrameContext] = None) -> dict:
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
        
        frame_ctx = frame if frame is not None else FrameContext(image=image)
        yolo = self._yolo_detect_all_once(context, frame_ctx)

        results = {
            "girl": [(d.box, d.score) for d in yolo.girls],
            "bugs": [(d.box, d.score) for d in yolo.bugs],
        }

        if DEBUG_ENABLED:
            debug_detections = [{"label": d.label or (YOLO_LABELS[d.cls_index] if 0 <= d.cls_index < len(YOLO_LABELS) else "unknown"),
                                 "box": list(d.box),
                                 "confidence": float(d.score)} for d in yolo.all]
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
        # 记录“本次移动到达”的原因，供外层逻辑判断（如浇水流程）
        self._last_move_reach_reason = None

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
                    # 硬样本环形缓冲：推入初始帧
                    self._hardcase_push(initial_image)
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
        # 角色位置平滑与预测（基于“脚底点”）
        ema_pos: Optional[Tuple[float, float]] = None
        last_seen_pos: Optional[Tuple[int, int]] = None
        last_move_dir: Optional[Tuple[float, float]] = None

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

                # 硬样本环形缓冲：每次截图都推入一帧（低开销，常态只保留最近 5 帧）
                if current_image is not None:
                    self._hardcase_push(current_image)

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
                    # 触发硬样本保存：截图失败导致动作失败
                    self._hardcase_trigger_and_capture_post(context, reason="screencap_fail")
                    return False

                # 退避等待（逐步加长），减少 IPC/截图压力
                backoff = min(0.6, MOVE_CHECK_INTERVAL * (1 + screencap_fail_count / 5))
                time.sleep(backoff)
                continue
            
            # 检测当前角色位置：单帧仅一次 YOLO 推理，并复用结果
            frame_ctx = FrameContext(image=current_image)
            yolo = self._yolo_detect_all_once(context, frame_ctx)
            box = yolo.girl_best.box if yolo.girl_best is not None else None
            # 推送当前帧的检测结果到调试窗口（在调试框中画框）
            if DEBUG_ENABLED:
                debug_detections = []
                for d in yolo.all:
                    label = d.label or (YOLO_LABELS[d.cls_index] if 0 <= d.cls_index < len(YOLO_LABELS) else "unknown")
                    debug_detections.append(
                        {
                            "label": label,
                            "box": list(d.box),
                            "confidence": float(d.score),
                        }
                    )
                update_debug_detections(debug_detections)
            if box is not None:
                # 记录最后一次看到的 box，供 OCR ROI 等后续逻辑复用（避免额外 YOLO）
                self._last_girl_box = list(box)
            center_pos = _box_center(box) if box is not None else None

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

                # 1) 基于“最后一次看到的位置 + 移动方向”做位置预测
                predicted_pos = None
                if last_seen_pos is not None and last_move_dir is not None:
                    px = int(last_seen_pos[0] + last_move_dir[0] * GIRL_PREDICT_STEP)
                    py = int(last_seen_pos[1] + last_move_dir[1] * GIRL_PREDICT_STEP)
                    predicted_pos = (px, py)

                # 2) 如果预测位置已进入容差范围，视为到达（可能被 UI 遮挡）
                if predicted_pos is not None:
                    pred_dist = _distance(predicted_pos, target_pos)
                    if pred_dist <= MOVE_TOLERANCE:
                        print(f"[FarmEventHandler] 预测位置已进入容差范围，判定到达: {predicted_pos}")
                        if DEBUG_ENABLED:
                            set_debug_info(f"Predicted reach: {predicted_pos}")
                            set_debug_target(None)
                        self._last_move_reach_reason = "predicted"
                        return True

                # 3) 若 OCR 能识别出坑位文字（湿润/缺水/干燥），也视为已到达交互范围
                try:
                    ocr_state = self._detect_soil_moisture_state(
                        context,
                        target_pos,
                        current_image,
                        girl_box=box,
                        frame=frame_ctx,
                    )
                    if ocr_state != "unknown":
                        print(f"[FarmEventHandler] OCR 命中文字({ocr_state})，判定已到达坑位交互范围")
                        if DEBUG_ENABLED:
                            set_debug_info(f"OCR hit: {ocr_state}")
                            set_debug_target(None)
                        self._last_move_reach_reason = "ocr"
                        return True
                except Exception:
                    pass

                # 复用捉虫逻辑：检测不到角色时，先向上轻推摇杆两次尝试把角色拉回视野
                _joystick_nudge_up(context, times=2, duration_ms=500, wait_s=0.2)

                # 如果连续丢失次数过多，放弃
                if lost_count >= max_lost_count:
                    print(f"[FarmEventHandler] 连续 {lost_count} 次无法检测到角色，放弃移动")
                    # 触发硬样本保存：识别丢失导致动作失败
                    self._hardcase_trigger_and_capture_post(context, reason="girl_lost")
                    return False

                time.sleep(MOVE_CHECK_INTERVAL)
                continue
            
            # 检测到角色，重置丢失计数器
            lost_count = 0

            # 使用 EMA 平滑“脚底点”位置（基于 current_pos）
            raw_pos = (int(current_pos[0]), int(current_pos[1]))
            if ema_pos is None:
                ema_pos = (float(raw_pos[0]), float(raw_pos[1]))
            else:
                ema_pos = (
                    GIRL_POS_EMA_ALPHA * raw_pos[0] + (1.0 - GIRL_POS_EMA_ALPHA) * ema_pos[0],
                    GIRL_POS_EMA_ALPHA * raw_pos[1] + (1.0 - GIRL_POS_EMA_ALPHA) * ema_pos[1],
                )
            smooth_pos = (int(round(ema_pos[0])), int(round(ema_pos[1])))
            # 若 EMA 偏差过大，回退原始位置，避免水平偏差累积
            if abs(smooth_pos[0] - raw_pos[0]) > GIRL_POS_EMA_MAX_DELTA or abs(smooth_pos[1] - raw_pos[1]) > GIRL_POS_EMA_MAX_DELTA:
                current_pos = raw_pos
            else:
                current_pos = smooth_pos
            last_seen_pos = current_pos
            
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
                self._last_move_reach_reason = "distance"
                return True

            # 进入近距离：先防抖等待并立即识别“坑位 UI 是否弹出”（OCR 信号）
            if dist <= MOVE_NEAR_DISTANCE:
                time.sleep(NEAR_DEBOUNCE_WAIT)
                try:
                    ocr_state = self._detect_soil_moisture_state(
                        context,
                        target_pos,
                        current_image,
                        girl_box=box,
                        frame=frame_ctx,
                    )
                    if ocr_state != "unknown":
                        print(f"[FarmEventHandler] 近距离 OCR 命中文字({ocr_state})，判定已到达")
                        if DEBUG_ENABLED:
                            set_debug_info(f"Near OCR hit: {ocr_state}")
                            set_debug_target(None)
                        self._last_move_reach_reason = "ocr"
                        return True
                except Exception:
                    pass
            
            # 计算摇杆滑动方向
            joystick_end = _calculate_joystick_direction(current_pos, target_pos)
            print(f"[FarmEventHandler] 摇杆滑动: {JOYSTICK_CENTER} -> {joystick_end}")

            # 记录当前移动方向，用于丢失时的位置预测
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            d = math.sqrt(dx * dx + dy * dy)
            if d > 1e-3:
                last_move_dir = (dx / d, dy / d)
            
            # 执行摇杆滑动操作 - 确保所有参数都是整数类型
            x1 = int(JOYSTICK_CENTER[0])
            y1 = int(JOYSTICK_CENTER[1])
            x2 = int(joystick_end[0])
            y2 = int(joystick_end[1])
            # 两段式：远距离用粗定位时长，近距离用精定位时长
            duration = int(SWIPE_DURATION_FINE if dist <= MOVE_NEAR_DISTANCE else SWIPE_DURATION)
            
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
        # 触发硬样本保存：移动超时往往由误检/漏检/控制抖动引起
        self._hardcase_trigger_and_capture_post(context, reason="move_timeout")
        return False
    
    def _check_repair_button(self, context: Context, image=None, *, frame: Optional[FrameContext] = None) -> tuple:
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
        
        reco_result = self._run_recognition_cached(
            context,
            task_name="Farm_CheckRepairButton",
            image=image,
            pipeline_override={
                "Farm_CheckRepairButton": {
                    "recognition": "TemplateMatch",
                    "template": REPAIR_BUTTON_TEMPLATE,
                    "roi": REPAIR_BUTTON_ROI,
                    "threshold": 0.7,
                }
            },
            frame=frame,
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

    def _check_water_button(self, context: Context, image=None, *, frame: Optional[FrameContext] = None) -> Tuple[bool, Optional[list]]:
        """
        检测浇水按钮是否出现

        返回:
            (是否出现, box)
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            image = context.tasker.controller.cached_image

        reco_result = self._run_recognition_cached(
            context,
            task_name="Farm_CheckWaterButton",
            image=image,
            pipeline_override={
                "Farm_CheckWaterButton": {
                    "recognition": "TemplateMatch",
                    "template": WATER_BUTTON_TEMPLATE,
                    "roi": WATER_BUTTON_ROI,
                    "threshold": WATER_BUTTON_THRESHOLD,
                }
            },
            frame=frame,
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

    def _is_plot_wet(
        self,
        context: Context,
        plot_pos: Tuple[int, int],
        image=None,
        *,
        frame: Optional[FrameContext] = None,
    ) -> bool:
        """
        判断坑位是否已湿润（TemplateMatch，替代原 ColorMatch）。

        思路：
        - 使用“湿润状态下坑位底部水面”模板进行匹配
        - 只要在坑位附近 ROI 内命中一次，就认为该坑位已湿润
        - 由于坑位周围都是水，可能出现多个命中；重复命中可忽略（只看 hit）
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            time.sleep(0.03)
            image = context.tasker.controller.cached_image

        cx, cy = int(plot_pos[0]), int(plot_pos[1])
        # 模板锚点：坑位中心下方约 20px（取水面位置）
        anchor_x = cx
        anchor_y = cy + int(PLOT_WET_TEMPLATE_OFFSET_Y)
        m = int(PLOT_WET_TEMPLATE_MARGIN)
        roi = _clamp_roi([anchor_x - m, anchor_y - m, m * 2, m * 2])

        reco_result = self._run_recognition_cached(
            context,
            task_name="Farm_PlotWetCheck",
            image=image,
            pipeline_override={
                "Farm_PlotWetCheck": {
                    "recognition": "TemplateMatch",
                    "template": PLOT_WET_WATER_TEMPLATE,
                    "roi": roi,
                    "threshold": PLOT_WET_TEMPLATE_THRESHOLD,
                }
            },
            frame=frame,
        )

        return _reco_hit(reco_result)

    def _detect_soil_moisture_state(
        self,
        context: Context,
        plot_pos: Tuple[int, int],
        image=None,
        *,
        girl_box: Optional[list] = None,
        frame: Optional[FrameContext] = None,
    ) -> str:
        """
        识别当前坑位土壤湿度状态。
        
        优先使用基于坑位中心推导出的精确 OCR ROI，
        再退回到以角色 box / 大圆形 ROI 为兜底。
        
        Returns:
            "wet" | "dry" | "unknown"
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            time.sleep(0.03)
            image = context.tasker.controller.cached_image

        def _ocr_state_primary(roi: List[int]) -> str:
            """
            在指定 ROI 内识别湿度状态（不做重叠判断）。
            仅用于“坑位精确 ROI”这一首选路径。
            """
            # 优先判定“湿润”
            wet_reco = self._run_recognition_cached(
                context,
                task_name="Farm_Moisture_Wet",
                image=image,
                pipeline_override={
                    "Farm_Moisture_Wet": {
                        "recognition": "OCR",
                        "expected": ["湿润"],
                        "roi": roi,
                    }
                },
                frame=frame,
            )
            if _reco_hit(wet_reco):
                return "wet"

            # 再判定“缺水/干燥/正常”
            dry_reco = self._run_recognition_cached(
                context,
                task_name="Farm_Moisture_Dry",
                image=image,
                pipeline_override={
                    "Farm_Moisture_Dry": {
                        "recognition": "OCR",
                        "expected": ["缺水", "干燥", "正常"],
                        "roi": roi,
                    }
                },
                frame=frame,
            )
            if _reco_hit(dry_reco):
                return "dry"

            return "unknown"

        def _ocr_state_with_overlap(
            roi: List[int],
            primary_roi: List[int],
        ) -> str:
            """
            在指定 ROI 内识别湿度状态，并要求文字 box 与坑位精确 ROI 有重叠。
            用于基于角色 box / 大圆形 ROI 的兜底识别，避免“错浇”判断到邻坑。
            """
            # 1) 判定“湿润”
            wet_reco = self._run_recognition_cached(
                context,
                task_name="Farm_Moisture_Wet",
                image=image,
                pipeline_override={
                    "Farm_Moisture_Wet": {
                        "recognition": "OCR",
                        "expected": ["湿润"],
                        "roi": roi,
                    }
                },
                frame=frame,
            )
            try:
                candidates = getattr(wet_reco, "filtered_results", None) if wet_reco is not None else None
            except Exception:
                candidates = None
            if candidates:
                for res in candidates:
                    try:
                        text = getattr(res, "text", None)
                        box = getattr(res, "box", None)
                    except Exception:
                        continue
                    if text == "湿润" and box is not None and _rects_intersect(primary_roi, list(box)):
                        return "wet"

            # 2) 判定“缺水/干燥/正常”
            dry_words = ["缺水", "干燥", "正常"]
            dry_reco = self._run_recognition_cached(
                context,
                task_name="Farm_Moisture_Dry",
                image=image,
                pipeline_override={
                    "Farm_Moisture_Dry": {
                        "recognition": "OCR",
                        "expected": dry_words,
                        "roi": roi,
                    }
                },
                frame=frame,
            )
            try:
                candidates = getattr(dry_reco, "filtered_results", None) if dry_reco is not None else None
            except Exception:
                candidates = None
            if candidates:
                for res in candidates:
                    try:
                        text = getattr(res, "text", None)
                        box = getattr(res, "box", None)
                    except Exception:
                        continue
                    if text in dry_words and box is not None and _rects_intersect(primary_roi, list(box)):
                        return "dry"

            return "unknown"

        # 1) 首选：基于坑位中心的精确 ROI（用户标注的 UI 区域）
        primary_roi = _build_moisture_ocr_roi(plot_pos)
        state = _ocr_state_primary(primary_roi)
        if state != "unknown":
            return state

        # 2) 备选：以角色识别 box 为中心扩张的 ROI（需要与坑位 ROI 重叠）
        if girl_box is None:
            try:
                girl_box = getattr(self, "_last_girl_box", None)
            except Exception:
                girl_box = None

        backup_rois: List[List[int]] = []
        if girl_box is not None:
            x, y, w, h = [int(v) for v in girl_box]
            backup_rois.append(_clamp_roi([x - 50, y - 50, w + 100, h + 100]))

        # 3) 最后兜底：以坑位中心为圆心的大 ROI（原始实现）
        px, py = int(plot_pos[0]), int(plot_pos[1])
        radius = int(MOISTURE_OCR_RADIUS)
        backup_rois.append(_clamp_roi([px - radius, py - radius, radius * 2, radius * 2]))

        for roi in backup_rois:
            state = _ocr_state_with_overlap(roi, primary_roi)
            if state != "unknown":
                return state

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
            """湿润判定：模板匹配 OR OCR(湿润)"""
            if image is None:
                image = capture_image_safe()
            if image is None:
                return False
            frame_ctx = FrameContext(image=image)
            try:
                if self._is_plot_wet(context, plot_xy, image, frame=frame_ctx):
                    return True
            except Exception:
                pass
            try:
                state = self._detect_soil_moisture_state(
                    context,
                    plot_xy,
                    image,
                    girl_box=getattr(self, "_last_girl_box", None),
                    frame=frame_ctx,
                )
                if state == "wet":
                    return True
            except Exception:
                pass

            # 兜底：强制使用坑位中心 ROI，避免 girl_box 过期导致 OCR 失效
            try:
                state = self._detect_soil_moisture_state(
                    context,
                    plot_xy,
                    image,
                    girl_box=None,
                    frame=frame_ctx,
                )
                return state == "wet"
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
            reward_frame = capture_image_safe()
            if reward_frame is not None and self._check_reward_popup(context, reward_frame, frame=FrameContext(image=reward_frame)):
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

            # 若移动阶段因 OCR/预测判定已到达，则不要再“向下走触发 UI”
            reached_by_ui = getattr(self, "_last_move_reach_reason", None) in ("ocr", "predicted")

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
            if not reached_by_ui:
                _joystick_nudge_down(context, times=1, duration_ms=500, wait_s=0.25)
            else:
                print("[浇水] 已由移动阶段 OCR/预测确认到达，跳过向下推动以避免互斥逻辑")
            try:
                controller.post_click(int(plot_x), int(plot_y)).wait()
            except Exception:
                pass
            time.sleep(0.25)

            # 3) 浇水直到 ColorMatch 判定湿润，并在首次湿润后额外再浇 2 次补满进度条
            water_count = 0
            wet_reached = False          # 是否已经从“未湿润”变为“湿润”
            extra_after_wet = 2          # 首次湿润后额外补浇次数

            screencap_fail_in_loop = 0  # 循环内截图失败计数
            max_screencap_fail_in_loop = 10  # 最大允许的连续截图失败次数

            while water_count < max_water_per_plot:
                # 单帧：每轮循环只截一次图，然后复用该帧进行所有检测
                frame_img = capture_image_safe()
                if frame_img is None:
                    screencap_fail_in_loop += 1
                    print(f"[浇水] 截图失败 ({screencap_fail_in_loop}/{max_screencap_fail_in_loop})")
                    if screencap_fail_in_loop >= max_screencap_fail_in_loop:
                        print("[浇水] 连续截图失败过多，退出浇水循环")
                        entry["fail_reason"] = "screencap_failed"
                        break
                    time.sleep(0.1)
                    continue
                screencap_fail_in_loop = 0  # 截图成功，重置计数
                frame_ctx = FrameContext(image=frame_img)

                # 推送当前帧到调试窗口（避免画面不更新）
                if DEBUG_ENABLED:
                    push_debug_frame(frame_img)

                if self._check_reward_popup(context, frame_ctx.image, frame=frame_ctx):
                    print("[浇水] 检测到奖励弹窗，先领取奖励")
                    self._run_sub_getreward(context)
                    time.sleep(0.2)
                    continue

                found, box = self._check_water_button(context, frame_ctx.image, frame=frame_ctx)
                if not found:
                    # 尝试点击坑位唤起按钮（会改变画面，因此这里不复用旧帧）
                    try:
                        controller.post_click(int(plot_x), int(plot_y)).wait()
                    except Exception:
                        pass
                    time.sleep(0.2)
                    # 重新截一帧再检测
                    frame_img2 = capture_image_safe()
                    if frame_img2 is None:
                        water_count += 1
                        entry["attempts"] += 1
                        continue
                    frame_ctx = FrameContext(image=frame_img2)
                    found, box = self._check_water_button(context, frame_ctx.image, frame=frame_ctx)

                if not found:
                    # 找不到浇水按钮时，检测是否已经湿润（可能坑位本来就是湿润状态）
                    check_frame = frame_ctx.image if frame_ctx else capture_image_safe()
                    if check_frame is not None and is_wet_combined((plot_x, plot_y), check_frame):
                        print("[浇水] 找不到浇水按钮但检测到已湿润，标记为完成")
                        entry["wet"] = True
                        entry["done"] = True
                        entry["fail_reason"] = None
                        return True
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

                # 尚未检测到“湿润” -> 检查是否已经变为湿润（状态切换点）
                if not wet_reached:
                    frame_after = capture_image_safe()
                    if is_wet_combined((plot_x, plot_y), frame_after):
                        wet_reached = True
                        print("[浇水] 检测到坑位已湿润，进入补浇阶段（额外再浇 2 次）")
                        # 不立即返回，继续 while 循环，在湿润状态下再浇 extra_after_wet 次
                    continue

                # 已经进入“湿润后补浇”阶段
                if wet_reached:
                    extra_after_wet -= 1
                    print(f"[浇水] 湿润后补浇，剩余次数: {max(extra_after_wet, 0)}")
                    if extra_after_wet <= 0:
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

        # 第一遍完成即可退出（所有坑位湿润）
        remaining = [it for it in plot_table if not it.get("done")]
        if not remaining:
            print("\n[浇水] 第一遍完成，全部坑位已湿润，结束浇水任务")
            return True

        # 第二遍：只重试第一遍仍未完成的坑位
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

        # 终局复核：对仍未完成的坑位再做一次湿润检测（避免偶发识别失败导致误判）
        still_left = [it for it in plot_table if not it.get("done")]
        if still_left:
            print("\n[浇水] 终局复核未完成坑位湿润状态...")
            for it in still_left:
                frame = capture_image_safe()
                if is_wet_combined(it["pos"], frame):
                    it["wet"] = True
                    it["done"] = True
                    it["fail_reason"] = None

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
            result = None
            tasker = getattr(context, "tasker", None)

            # 优先使用 tasker.post_pipeline().wait()，确保执行完成
            if tasker is not None and hasattr(tasker, "post_pipeline") and callable(getattr(tasker, "post_pipeline")):
                job = tasker.post_pipeline("Sub_Getreward")
                result = job.wait() if job is not None else None
            # 次优先：tasker.run_pipeline(name)
            elif tasker is not None and hasattr(tasker, "run_pipeline") and callable(getattr(tasker, "run_pipeline")):
                result = tasker.run_pipeline("Sub_Getreward")
            # 兜底：Context.run_task（可能是异步）
            elif hasattr(context, "run_task") and callable(getattr(context, "run_task")):
                result = context.run_task("Sub_Getreward", None)

            print(f"[FarmEventHandler] Sub_Getreward 执行结果: {result}")

            # 轻微等待，给弹窗关闭留出时间
            time.sleep(0.2)

            # 如果奖励弹窗仍在，直接点击弹窗按钮兜底关闭
            if self._check_reward_popup(context):
                try:
                    controller = context.tasker.controller
                    controller.post_click(int(SUB_GETREWARD_TARGET[0]), int(SUB_GETREWARD_TARGET[1])).wait()
                    time.sleep(0.3)
                except Exception as e:
                    print(f"[FarmEventHandler] 兜底点击奖励弹窗失败: {e}")

            # 这里无法可靠判断 result 语义，保持“只要不抛异常就视为成功”
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
    
    def _handle_windmill_repair(self, context: Context) -> bool:
        """
        处理风车修理事件
        
        流程:
        1. 使用 YOLOv8 检测角色位置
        2. 先移动到右侧栅栏上方入口点 [1063, 295]
        3. 再从入口点移动到风车附近 [1157, 457]
        4. 检测并点击修理按钮（必要时向下轻推一次补救）
        5. 调用 Sub_Getreward 获取奖励
        6. 修理完成后，从风车位置回到入口点
        7. 再从入口点回到农场右上方默认站位 [627, 287]，方便衔接后续浇水
        """
        print("\n" + "=" * 40)
        print("[风车修理] 开始处理风车修理事件")
        print("=" * 40)
        
        # 步骤 1: 先移动到右侧栅栏上方入口点，避免从农田区域直接撞栅栏
        print("\n[风车修理] 步骤 1: 移动到右侧栅栏上方入口点...")
        if not self._move_character_to_target(context, WINDMILL_ENTRY_POS):
            print("[风车修理] 移动到入口点失败")
            return False
        
        # 步骤 2: 从入口点移动到风车附近
        print("\n[风车修理] 步骤 2: 从入口点移动到风车位置...")
        if not self._move_character_to_target(context, WINDMILL_TARGET_POS):
            print("[风车修理] 移动到风车位置失败")
            return False
        
        # 等待界面稳定
        time.sleep(DEFAULT_WAIT)
        
        # 步骤 3: 检测修理按钮（原地尝试3次）
        print("\n[风车修理] 步骤 3: 检测修理按钮...")
        max_static_attempts = 3  # 原地检测次数
        found = False
        box = None
        
        print(f"[风车修理] 在当前位置尝试检测修理按钮（最多{max_static_attempts}次）...")
        for attempt in range(max_static_attempts):
            found, box = self._check_repair_button(context)
            if found:
                print(f"[风车修理] 第 {attempt + 1} 次检测到修理按钮！")
                break
            print(f"[风车修理] 第 {attempt + 1}/{max_static_attempts} 次，未检测到修理按钮")
            time.sleep(0.3)
        
        # 如果原地没找到，尝试向下移动一步并检测
        if not found:
            print("[风车修理] 原地未检测到修理按钮，尝试向下移动一步...")
            joystick_down_end = (JOYSTICK_CENTER[0], JOYSTICK_CENTER[1] + JOYSTICK_RADIUS)
            move_duration = 500  # 每次移动持续时间（毫秒）
            
            # 执行向下移动 - 使用安全滑动函数
            x1 = int(JOYSTICK_CENTER[0])
            y1 = int(JOYSTICK_CENTER[1])
            x2 = int(joystick_down_end[0])
            y2 = int(joystick_down_end[1])
            
            if _safe_swipe(context, x1, y1, x2, y2, move_duration):
                print("[风车修理] 摇杆向下操作完成")
            else:
                print("[风车修理] 摇杆操作失败")
            
            # 等待移动完成
            time.sleep(0.5)
            
            # 再次检测修理按钮（向下移动后尝试最多2次）
            max_move_attempts = 2
            for move_attempt in range(max_move_attempts):
                found, box = self._check_repair_button(context)
                if found:
                    print(f"[风车修理] 向下移动后第 {move_attempt + 1} 次检测到修理按钮！")
                    break
                print(f"[风车修理] 向下移动后第 {move_attempt + 1} 次仍未检测到修理按钮")
                time.sleep(0.3)
        
        # 如果所有尝试后仍未找到修理按钮，返回失败
        if not found:
            print("[风车修理] 尝试所有方法后仍未检测到修理按钮")
            return False
        
        # 步骤 4: 点击修理按钮
        print("\n[风车修理] 步骤 4: 点击修理按钮...")
        if not self._click_repair_button(context, box):
            print("[风车修理] 点击修理按钮失败")
            return False
        
        # 等待修理完成和奖励弹窗
        time.sleep(DEFAULT_WAIT)
        
        # 步骤 5: 获取奖励
        print("\n[风车修理] 步骤 5: 获取奖励...")
        self._run_sub_getreward(context)
        
        # 步骤 6: 从风车位置回到入口点，避免停在栅栏内侧
        print("\n[风车修理] 步骤 6: 修理完成后回到入口点...")
        if not self._move_character_to_target(context, WINDMILL_ENTRY_POS):
            print("[风车修理] 回到入口点失败（忽略并继续返回起始位置）")
        else:
            time.sleep(DEFAULT_WAIT * 0.5)
        
        # 步骤 7: 从入口点回到农场右上方默认站位，方便衔接后续浇水逻辑
        print("\n[风车修理] 步骤 7: 从入口点回到农场右上方默认站位...")
        if not self._move_character_to_target(context, FARM_WATERING_START_POS):
            print("[风车修理] 回到默认站位失败")
        else:
            time.sleep(DEFAULT_WAIT * 0.5)
        
        print("\n[风车修理] 风车修理完成！")
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

        def capture_frame_safe() -> Optional[FrameContext]:
            """单帧截图：一轮循环内复用该帧进行 reward/OCR/YOLO 等检测。"""
            try:
                context.tasker.controller.post_screencap().wait()
                time.sleep(0.05)
                img = context.tasker.controller.cached_image
                if img is None:
                    return None
                return FrameContext(image=img)
            except Exception:
                return None
        
        while True:
            # 单帧：截一次图 -> 在这一帧上跑完必要检测
            frame = capture_frame_safe()
            if frame is None:
                time.sleep(0.2)
                continue

            # ========== 全局终止条件：奖励弹窗/可执行 Sub_Getreward ==========
            # 用户要求：验证到奖励弹窗也就是 "Sub_Getreward" 节点成功执行意味着虫子已捉完，可直接结束捉虫任务
            if self._check_reward_popup(context, frame.image, frame=frame):
                print("[捉虫] 检测到奖励弹窗（可领取奖励），直接结束捉虫任务")
                self._run_sub_getreward(context)
                return True

            # 检测场上所有目标
            girl_pos, bugs_list = self._detect_all_targets(context, frame.image)
            
            # ========== 角色检测失败处理（避免卡死） ==========
            if girl_pos is None:
                # 情况A：能检测到虫子，但检测不到角色 -> 向上推动摇杆两次，每次再检查
                if len(bugs_list) > 0:
                    _joystick_nudge_up(context, times=2, duration_ms=300, wait_s=0.2)

                    # 重新截一帧再检测一次
                    frame2 = capture_frame_safe()
                    girl_pos, bugs_list = self._detect_all_targets(context, frame2.image) if frame2 else (None, [])
                    if girl_pos is None:
                        print("[捉虫] 错误：检测到虫子但连续尝试后仍无法检测到角色，退出捉虫任务")
                        return False
                else:
                    # 情况B：角色和虫子都检测不到 -> 尝试执行 Sub_Getreward，能执行说明虫子已捉完
                    print("[捉虫] 角色与虫子均未检测到，尝试执行 Sub_Getreward 判断是否已结束...")
                    try:
                        if self._check_reward_popup(context, frame.image, frame=frame):
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
                if self._check_reward_popup(context, frame.image, frame=frame):
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
                print(f"[捉虫] [OK] 成功捕捉第 {caught_count} 只虫子！")
            else:
                print("[捉虫] [FAIL] 捕捉失败，尝试下一只")
            
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
        
        frame_ctx = FrameContext(image=image)
        yolo = self._yolo_detect_all_once(context, frame_ctx)

        girl_pos: Optional[tuple] = None
        if yolo.girl_best is not None:
            b = yolo.girl_best.box
            girl_pos = (int(b[0] + b[2] // 2), int(b[1] + b[3] // 2))

        bugs_list: List[tuple] = []
        for d in yolo.bugs:
            b = d.box
            bugs_list.append((int(b[0] + b[2] // 2), int(b[1] + b[3] // 2)))

        if DEBUG_ENABLED:
            debug_detections = []
            for d in yolo.all:
                label = d.label or (YOLO_LABELS[d.cls_index] if 0 <= d.cls_index < len(YOLO_LABELS) else "unknown")
                debug_detections.append({"label": label, "box": list(d.box), "confidence": float(d.score)})
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
    
    def _check_catch_button(
        self,
        context: Context,
        image=None,
        *,
        frame: Optional[FrameContext] = None,
    ) -> Tuple[bool, Optional[list]]:
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
        
        reco_result = self._run_recognition_cached(
            context,
            task_name="Farm_CheckCatchButton",
            image=image,
            pipeline_override={
                "Farm_CheckCatchButton": {
                    "recognition": "TemplateMatch",
                    "template": CATCH_BUTTON_TEMPLATE,
                    "roi": CATCH_BUTTON_ROI,
                    "threshold": 0.7,
                }
            },
            frame=frame,
        )
        
        if _reco_hit(reco_result):
            box = _reco_box(reco_result)
            return (True, box)
        
        return (False, None)
    
    def _check_reward_popup(self, context: Context, image=None, *, frame: Optional[FrameContext] = None) -> bool:
        """
        检查是否出现奖励弹窗（Sub_Getreward）
        
        参数:
            context: MAA 上下文
            
        返回:
            bool: 是否出现奖励弹窗
        """
        if image is None:
            context.tasker.controller.post_screencap().wait()
            image = context.tasker.controller.cached_image
        if image is None:
            return False
        
        reco_result = self._run_recognition_cached(
            context,
            task_name="Farm_CheckReward",
            image=image,
            pipeline_override={
                "Farm_CheckReward": {
                    "recognition": "OCR",
                    "expected": [
                        "获得物品",
                        "获得道具",
                        "道具",
                        "本勒",
                        "LEVEL",
                        "等级提升"
                    ],
                    "roi": [78, 15, 1014, 254],
                }
            },
            frame=frame,
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


class FarmWindmillRepair(FarmEventHandler):
    """
    风车修理专用 Action（简化调用）
    
    Pipeline 调用示例:
    {
        "custom_action": "FarmWindmillRepair"
    }
    """
    
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        # 强制设置 event_type 为 windmill
        argv.custom_action_param = json.dumps({"event_type": "windmill"})
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
