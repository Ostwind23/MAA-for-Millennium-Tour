"""
钓鱼调试可视化模块
使用 OpenCV 窗口显示检测结果 + 记录调试日志

功能:
- 显示黄色箭头检测过程（最左/最右像素点、连线、中点、方向线）
- 显示 QTE 按钮检测结果
- 显示操控杆区域和中心点
- 显示当前阶段状态
- 将调试日志写入项目根目录下的 debug 文件夹
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import Optional, Tuple, List, Set

# ==================== 日志记录器 ====================

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))
_DEBUG_DIR = os.path.join(_PROJECT_ROOT, "debug")


class FishingLogger:
    """
    简单的文件日志记录器
    - 日志目录：<项目根>/debug
    - 文件名：YYYYMMDD_HHMMSS_fishing.log
    - 每条日志立即 flush，尽量保证即使进程被 kill 也能保留最后一条日志
    """

    def __init__(self) -> None:
        self._file_path: Optional[str] = None

    def _ensure_file(self) -> None:
        if self._file_path is not None:
            return
        try:
            os.makedirs(_DEBUG_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_fishing.log"
            self._file_path = os.path.join(_DEBUG_DIR, filename)
        except Exception as e:
            print(f"[FishingDebug] 日志文件初始化失败: {e}")
            self._file_path = None

    def log(self, message: str) -> None:
        """将一条日志写入文件（附带时间戳）"""
        self._ensure_file()
        if not self._file_path:
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            line = f"[{ts}] {message}\n"
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # 某些平台可能不支持 fsync，忽略即可
                    pass
        except Exception as e:
            print(f"[FishingDebug] 写入日志失败: {e}")


_logger = FishingLogger()


def log(message: str) -> None:
    """对外暴露的日志接口"""
    _logger.log(message)

# ==================== 可视化颜色配置 (BGR) ====================
COLOR_GREEN = (0, 255, 0)       # 绿色 - 检测到的像素点
COLOR_CYAN = (255, 255, 0)      # 青色 - 连线
COLOR_RED = (0, 0, 255)         # 红色 - 中点
COLOR_MAGENTA = (255, 0, 255)   # 紫色 - 方向线
COLOR_YELLOW = (0, 255, 255)    # 黄色 - 文字/高亮
COLOR_GRAY = (128, 128, 128)    # 灰色 - 已点击的按钮
COLOR_WHITE = (255, 255, 255)   # 白色 - ROI 边框
COLOR_BLUE = (255, 0, 0)        # 蓝色 - 操控杆中心


class FishingDebugViewer:
    """钓鱼调试查看器"""
    
    WINDOW_NAME = "Fishing Debug"
    
    def __init__(self):
        self._enabled = False
        self._last_image = None
        self._current_phase = "Idle"
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def start(self):
        """启动调试窗口"""
        self._enabled = True
        try:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, 960, 540)
        except Exception as e:
            print(f"[FishingDebug] 创建窗口失败: {e}")
            self._enabled = False
    
    def stop(self):
        """关闭调试窗口"""
        self._enabled = False
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except Exception:
            pass
    
    def set_phase(self, phase: str):
        """设置当前阶段"""
        self._current_phase = phase
    
    def draw_arrow_detection(
        self, 
        image: np.ndarray, 
        left_point: Tuple[int, int], 
        right_point: Tuple[int, int], 
        mid_point: Tuple[float, float], 
        joystick_center: Tuple[int, int], 
        angle: float,
        roi: List[int] = None,
        yellow_mask: np.ndarray = None
    ):
        """
        绘制箭头检测过程
        
        参数:
            image: 原始图像
            left_point: 最左侧黄色像素点坐标
            right_point: 最右侧黄色像素点坐标
            mid_point: 连线中点坐标
            joystick_center: 操控杆中心坐标
            angle: 计算出的角度
            roi: 检测区域 [x, y, w, h]
            yellow_mask: 黄色像素掩码（可选，用于显示检测到的黄色区域）
        """
        if not self._enabled:
            return
        
        vis = image.copy()
        
        # 画 ROI 区域
        if roi:
            cv2.rectangle(vis, (roi[0], roi[1]), 
                         (roi[0] + roi[2], roi[1] + roi[3]), 
                         COLOR_WHITE, 1)
        
        # 画操控杆中心
        cv2.circle(vis, joystick_center, 5, COLOR_BLUE, -1)
        cv2.circle(vis, joystick_center, 50, COLOR_BLUE, 1)
        
        # 画最左和最右的黄色像素点
        cv2.circle(vis, left_point, 8, COLOR_GREEN, -1)
        cv2.circle(vis, right_point, 8, COLOR_GREEN, -1)
        
        # 画连线
        cv2.line(vis, left_point, right_point, COLOR_CYAN, 2)
        
        # 画中点
        mid_int = (int(mid_point[0]), int(mid_point[1]))
        cv2.circle(vis, mid_int, 10, COLOR_RED, -1)
        
        # 画从操控杆中心到中点的方向线
        cv2.line(vis, joystick_center, mid_int, COLOR_MAGENTA, 3)
        
        # 画延长线（显示拖拽方向）
        direction_length = 120
        rad = np.radians(angle)
        end_x = int(joystick_center[0] + direction_length * np.cos(rad))
        end_y = int(joystick_center[1] + direction_length * np.sin(rad))
        cv2.arrowedLine(vis, joystick_center, (end_x, end_y), COLOR_YELLOW, 2, tipLength=0.2)
        
        # 显示信息
        self._draw_info(vis, [
            f"Phase: {self._current_phase}",
            f"Arrow Angle: {angle:.1f} deg",
            f"Left: {left_point}",
            f"Right: {right_point}",
            f"Mid: ({mid_point[0]:.0f}, {mid_point[1]:.0f})"
        ])
        
        self._show(vis)
    
    def draw_qte_buttons(
        self, 
        image: np.ndarray, 
        buttons: List[dict], 
        clicked_buttons: Set[Tuple[int, int]],
        button_timers: dict = None
    ):
        """
        绘制 QTE 按钮检测
        
        参数:
            image: 原始图像
            buttons: 检测到的按钮列表
            clicked_buttons: 已点击的按钮中心坐标集合
            button_timers: 按钮计时器字典（可选）
        """
        if not self._enabled:
            return
        
        vis = image.copy()
        
        for i, btn in enumerate(buttons):
            center = btn['center']
            radius = btn['radius']
            
            # 判断是否已点击
            is_clicked = center in clicked_buttons
            color = COLOR_GRAY if is_clicked else COLOR_GREEN
            
            # 画按钮圆
            cv2.circle(vis, center, radius, color, 2)
            cv2.circle(vis, center, 5, color, -1)
            
            # 显示按钮编号
            cv2.putText(vis, f"#{i+1}", 
                       (center[0] - 10, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 如果有计时器，显示剩余时间
            if button_timers and center in button_timers:
                elapsed = button_timers.get(center, {}).get('elapsed', 0)
                cv2.putText(vis, f"{elapsed:.1f}s", 
                           (center[0] - 20, center[1] + radius + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
        
        # 显示信息
        self._draw_info(vis, [
            f"Phase: QTE",
            f"Buttons: {len(buttons)}",
            f"Clicked: {len(clicked_buttons)}"
        ])
        
        self._show(vis)
    
    def draw_joystick_drag(
        self, 
        image: np.ndarray, 
        joystick_center: Tuple[int, int],
        drag_angle: float,
        drag_radius: int,
        strategy: str
    ):
        """
        绘制摇杆拖拽方向
        
        参数:
            image: 原始图像
            joystick_center: 操控杆中心
            drag_angle: 拖拽角度
            drag_radius: 拖拽半径
            strategy: 策略名称
        """
        if not self._enabled:
            return
        
        vis = image.copy()
        
        # 画操控杆中心
        cv2.circle(vis, joystick_center, 5, COLOR_BLUE, -1)
        cv2.circle(vis, joystick_center, drag_radius, COLOR_BLUE, 1)
        
        # 画拖拽方向
        rad = np.radians(drag_angle)
        end_x = int(joystick_center[0] + drag_radius * np.cos(rad))
        end_y = int(joystick_center[1] + drag_radius * np.sin(rad))
        cv2.arrowedLine(vis, joystick_center, (end_x, end_y), COLOR_MAGENTA, 3, tipLength=0.15)
        
        # 显示信息
        self._draw_info(vis, [
            f"Phase: Struggle",
            f"Strategy: {strategy}",
            f"Drag Angle: {drag_angle:.1f} deg"
        ])
        
        self._show(vis)
    
    def draw_bite_detection(
        self, 
        image: np.ndarray, 
        joystick_roi: List[int],
        joystick_center: Tuple[int, int],
        detected: bool,
        color_lower: List[int] = None,
        color_upper: List[int] = None
    ):
        """
        绘制上钩检测状态
        
        参数:
            image: 原始图像
            joystick_roi: 操控杆区域
            joystick_center: 操控杆中心
            detected: 是否检测到上钩
            color_lower: BGR 颜色下限（用于显示检测到的像素）
            color_upper: BGR 颜色上限（用于显示检测到的像素）
        """
        if not self._enabled:
            return
        
        vis = image.copy()
        
        # 如果提供了颜色范围，显示检测到的蓝色像素
        blue_pixel_count = 0
        if color_lower is not None and color_upper is not None:
            roi_img = image[joystick_roi[1]:joystick_roi[1]+joystick_roi[3], 
                          joystick_roi[0]:joystick_roi[0]+joystick_roi[2]]
            lower = np.array(color_lower)
            upper = np.array(color_upper)
            mask = cv2.inRange(roi_img, lower, upper)
            blue_pixel_count = cv2.countNonZero(mask)
            
            # 在原图上用亮色高亮显示检测到的像素
            mask_full = np.zeros(image.shape[:2], dtype=np.uint8)
            mask_full[joystick_roi[1]:joystick_roi[1]+joystick_roi[3], 
                     joystick_roi[0]:joystick_roi[0]+joystick_roi[2]] = mask
            vis[mask_full > 0] = [0, 255, 255]  # 用黄色高亮显示检测到的蓝色像素
        
        # 画 ROI
        color = COLOR_GREEN if detected else COLOR_WHITE
        cv2.rectangle(vis, 
                     (joystick_roi[0], joystick_roi[1]),
                     (joystick_roi[0] + joystick_roi[2], joystick_roi[1] + joystick_roi[3]),
                     color, 2)
        
        # 画中心点
        cv2.circle(vis, joystick_center, 5, COLOR_BLUE, -1)
        
        # 显示信息
        status = "BITE DETECTED!" if detected else "Waiting..."
        info_lines = [
            f"Phase: WaitBite",
            f"Status: {status}",
            f"Blue Pixels: {blue_pixel_count}"
        ]
        self._draw_info(vis, info_lines)
        
        self._show(vis)
    
    def draw_phase_info(self, image: np.ndarray, info_lines: List[str]):
        """
        绘制通用阶段信息
        
        参数:
            image: 原始图像
            info_lines: 信息行列表
        """
        if not self._enabled:
            return
        
        vis = image.copy()
        self._draw_info(vis, info_lines)
        self._show(vis)
    
    def _draw_info(self, image: np.ndarray, lines: List[str]):
        """在图像左上角绘制信息文字"""
        y = 30
        for line in lines:
            # 绘制黑色背景以提高可读性
            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(image, (5, y - text_h - 5), (15 + text_w, y + 5), (0, 0, 0), -1)
            cv2.putText(image, line, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)
            y += 30
    
    def _show(self, image: np.ndarray):
        """显示图像"""
        if self._enabled:
            try:
                cv2.imshow(self.WINDOW_NAME, image)
                cv2.waitKey(1)
            except Exception as e:
                print(f"[FishingDebug] 显示图像失败: {e}")


# ==================== 全局调试器实例 ====================
_debug_viewer: Optional[FishingDebugViewer] = None


def get_debug_viewer() -> FishingDebugViewer:
    """获取全局调试器实例"""
    global _debug_viewer
    if _debug_viewer is None:
        _debug_viewer = FishingDebugViewer()
    return _debug_viewer


def start_debug():
    """启动调试模式"""
    get_debug_viewer().start()


def stop_debug():
    """停止调试模式"""
    get_debug_viewer().stop()


def is_debug_enabled() -> bool:
    """检查调试模式是否启用"""
    viewer = get_debug_viewer()
    return viewer.enabled
