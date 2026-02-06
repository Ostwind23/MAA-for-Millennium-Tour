"""
自动钓鱼 Custom Action
实现钓鱼的自动化流程：下杆、等待上钩、拉扯、QTE、结果处理

钓鱼流程说明:
1. 选择饵料界面 - OCR 检测数字确认在选饵界面
2. 下杆阶段 - 点击操控杆下杆
3. 等待上钩 - ColorMatch 检测操控杆区域的蓝色像素
4. 拉扯阶段 - 颜色扫描检测黄色箭头方向，根据策略拖动操控杆
5. QTE阶段 - 动态检测鱼形按钮，根据按钮出现时间计算点击时机
6. 结果处理 - 黑屏检测 + 点击继续

策略说明:
- aggressive: 逆向拉扯，进度快但体力消耗大
- conservative: 顺向拉扯，进度慢但省体力

阶段检测策略（优化性能）:
- WaitBite: 只做 ColorMatch 蓝色检测
- Struggle: 箭头方向检测、QTE 触发检测、黑屏检测（不做 OCR）
- QTE: 鱼按钮检测、黑屏检测（不做箭头检测和 OCR）
- Result: 黑屏结束后点击
"""

from maa.custom_action import CustomAction
from maa.context import Context
import json
import time
import math
import os
from typing import Optional, List, Tuple, Set, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import cv2

# 导入调试模块
from . import fishing_debug


# ==================== QTE 按钮槽位数据结构 ====================
@dataclass
class QTEButtonSlot:
    """QTE 按钮槽位，存储单个按钮的检测和点击信息"""
    center: Tuple[int, int]           # 按钮中心位置
    button_radius: int                # 按钮半径（从模板匹配 box 计算）
    circle_radius: float = 0.0        # 首次检测到的外圆半径（0 表示未检测到圆圈）
    first_detect_time: float = 0.0    # 首次检测时间戳
    target_click_time: float = 0.0    # 计算好的点击时间戳
    clicked: bool = False             # 是否已点击
    confidence: float = 0.0           # 模板匹配置信度

# ==================== 路径配置 ====================
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))
_IMAGE_DIR = os.path.join(_PROJECT_ROOT, "assets", "resource", "image")

# 钓鱼相关图片目录
_FISHING_IMG_DIR = "fishing"

# ==================== 屏幕尺寸 ====================
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# ==================== 操控杆配置 ====================
# 操控杆区域 [x, y, w, h]（紧贴操控杆边缘）
JOYSTICK_ROI = [928, 369, 247, 247]
JOYSTICK_CENTER = (
    JOYSTICK_ROI[0] + JOYSTICK_ROI[2] // 2,  # 中心 X = 1051
    JOYSTICK_ROI[1] + JOYSTICK_ROI[3] // 2   # 中心 Y = 492
)
# 拖拽半径 = ROI 宽高的一半（直接拖到圆圈边缘）
JOYSTICK_RADIUS = min(JOYSTICK_ROI[2], JOYSTICK_ROI[3]) // 2  # = 123

# ==================== 上钩检测配置（ColorMatch）====================
# 蓝色像素检测参数
# 用户提供的蓝色 RGB [29, 86, 136]，±20 范围
BITE_DETECT_ROI = [1034, 476, 182, 179]  # 上钩检测专用区域
# MaaFramework ColorMatch 使用 RGB 格式
BITE_COLOR_LOWER_RGB = [9, 66, 116]    # RGB 下限 (29-20, 86-20, 136-20)
BITE_COLOR_UPPER_RGB = [49, 106, 156]  # RGB 上限 (29+20, 86+20, 136+20)
# OpenCV 调试显示使用 BGR 格式
BITE_COLOR_LOWER_BGR = [116, 66, 9]    # BGR 下限
BITE_COLOR_UPPER_BGR = [156, 106, 49]  # BGR 上限
BITE_COLOR_COUNT = 100  # 像素数阈值
BITE_TIMEOUT = 20.0  # 等待上钩超时时间（秒）

# ==================== 箭头检测配置（颜色扫描）====================
# 黄色箭头检测区域（比操控杆区域各向扩大，包含外围圆弧）
# 基础扩展 60px，再各向额外扩大 20%（≈+25px）
_ARROW_EXPAND = 85  # 60 * 1.2 ≈ 72，向上取整到 85 留更多余量
ARROW_DETECT_ROI = [
    JOYSTICK_ROI[0] - _ARROW_EXPAND,
    JOYSTICK_ROI[1] - _ARROW_EXPAND,
    JOYSTICK_ROI[2] + _ARROW_EXPAND * 2,
    JOYSTICK_ROI[3] + _ARROW_EXPAND * 2
]
# 黄色箭头 BGR 颜色范围
# RGB: upper[166, 255, 188], lower[146, 239, 168] → BGR 格式
ARROW_COLOR_LOWER_BGR = np.array([168, 239, 146])  # BGR 下限
ARROW_COLOR_UPPER_BGR = np.array([188, 255, 166])  # BGR 上限
ARROW_MIN_PIXELS = 100  # 最少黄色像素数（防止噪点误判）

# ==================== QTE 配置 ====================
QTE_FISH_TEMPLATE = f"{_FISHING_IMG_DIR}/icon.png"
QTE_CIRCLE_SHRINK_SPEED = 80.0  # 像素/秒（默认值，可通过参数覆盖）
QTE_CLICK_OFFSET = -0.08  # 点击时机提前量（秒），补偿 post_click IPC 延迟
QTE_RADIUS_CORRECTION = 3  # 圆圈半径修正（像素），补偿外缘辉光导致的检测偏大
QTE_MAX_DURATION = 6.0  # QTE 最长持续时间（秒）
QTE_BUTTON_COUNT = 3  # QTE 按钮总数
QTE_TEMPLATE_THRESHOLD = 0.9  # 模板匹配阈值（提高到 0.75 减少误识别）
QTE_CLICK_WINDOW = 1.0  # 点击时间窗口（秒），超过此时间报超时

# 从鱼图标 box 扩展到整个圆形按钮的偏移
QTE_BUTTON_EXPAND_LEFT = 18
QTE_BUTTON_EXPAND_TOP = 24
QTE_BUTTON_EXPAND_WIDTH = 34
QTE_BUTTON_EXPAND_HEIGHT = 51

# 白色圆圈检测配置
QTE_CIRCLE_COLOR_LOWER = np.array([200, 200, 200])  # BGR 白色下限
QTE_CIRCLE_COLOR_UPPER = np.array([255, 255, 255])  # BGR 白色上限
QTE_CIRCLE_SCAN_DIRECTIONS = 16  # 径向扫描方向数
QTE_CIRCLE_MAX_RADIUS = 200  # 扫描最大半径（需覆盖屏幕边缘按钮）
QTE_CIRCLE_MIN_WHITE_RATIO = 0.3  # 某方向上白色像素比例阈值

# ==================== OCR 配置 ====================
# 选择饵料界面 OCR - 检测任意数字
BAIT_SELECT_ROI = [1078, 197, 195, 154]
BAIT_SELECT_EXPECTED = r"\d+"  # 匹配任意数字

# 结算页面 OCR - 检测"特质"或"重量"
RESULT_OCR_ROI = [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT]  # 全屏
RESULT_OCR_EXPECTED = ["特质", "重量"]

# ==================== 黑屏检测配置 ====================
BLACKOUT_DARK_THRESHOLD = 30  # 像素亮度阈值
BLACKOUT_RATIO_THRESHOLD = 0.6  # 暗色像素比例阈值（70%）

# ==================== 时间配置 ====================
CAST_ROD_DELAY = 0.5  # 下杆后等待时间
BITE_CHECK_INTERVAL = 0.05  # 上钩检测间隔
STRUGGLE_CHECK_INTERVAL = 0.05  # 拉扯阶段检测间隔
QTE_CHECK_INTERVAL = 0.03  # QTE 阶段检测间隔（需要更快）
QTE_BUTTON_CLICK_DELAY = 0.1  # QTE 按钮点击后等待
RESULT_CHECK_INTERVAL = 0.3  # 结果界面检测间隔
RESULT_CLICK_DELAY = 1.0  # 点击结果后等待
SWIPE_DURATION = 500  # 摇杆滑动持续时间（毫秒）

# ==================== 调试配置 ====================
DEBUG_ENABLED = True  # 调试开关


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


def _reco_box(reco_result) -> Optional[list]:
    """从 run_recognition 结果中获取 box"""
    if _reco_hit(reco_result):
        return reco_result.best_result.box
    return None


def _box_center(box: list) -> Optional[Tuple[int, int]]:
    """计算 box 的中心坐标"""
    if box is None:
        return None
    return (box[0] + box[2] // 2, box[1] + box[3] // 2)


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


def _debug_log(message: str) -> None:
    """
    调试日志辅助函数：
    - 打印到控制台
    - 同时写入 fishing_debug 提供的日志文件
    """
    try:
        print(message)
    except Exception:
        # 某些环境可能禁止 stdout，忽略
        pass
    try:
        fishing_debug.log(message)
    except Exception:
        # 日志写入失败不应影响主逻辑
        pass


def _safe_swipe(context: Context, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
    """
    安全的滑动操作，使用底层 touch_down/move/up API 实现
    
    参数:
        context: MAA 上下文
        x1, y1: 起点坐标
        x2, y2: 终点坐标
        duration: 滑动持续时间（毫秒）
        
    返回:
        bool: 是否成功
    """
    try:
        controller = context.tasker.controller
        
        # 确保所有参数都是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        duration = int(duration)
        
        # 步骤1: 按下起点
        touch_down_job = controller.post_touch_down(x1, y1, contact=0, pressure=1)
        if not touch_down_job.wait():
            print(f"[Fishing] touch_down 失败")
            return False
        
        # 步骤2: 移动到终点（分多步移动以模拟滑动）
        steps = max(5, duration // 50)
        for i in range(1, steps + 1):
            t = i / steps
            cur_x = int(x1 + (x2 - x1) * t)
            cur_y = int(y1 + (y2 - y1) * t)
            
            touch_move_job = controller.post_touch_move(cur_x, cur_y, contact=0, pressure=1)
            if not touch_move_job.wait():
                print(f"[Fishing] touch_move 失败 (step {i}/{steps})")
                try:
                    controller.post_touch_up(contact=0).wait()
                except:
                    pass
                return False
            
            time.sleep(duration / steps / 1000.0)
        
        # 步骤3: 抬起
        touch_up_job = controller.post_touch_up(contact=0)
        if not touch_up_job.wait():
            print(f"[Fishing] touch_up 失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"[Fishing] 滑动异常: {e}")
        import traceback
        traceback.print_exc()
        try:
            context.tasker.controller.post_touch_up(contact=0).wait()
        except:
            pass
        return False


class AutoFishing(CustomAction):
    """
    自动钓鱼 Custom Action
    
    实现完整的钓鱼自动化流程，包括：
    - 选择饵料
    - 下杆
    - 等待上钩
    - 拉扯阶段（支持顺向/逆向策略）
    - QTE 处理（动态规划）
    - 结果处理
    
    Pipeline 调用示例:
    {
        "custom_action": "AutoFishing",
        "custom_action_param": {
            "strategy": "aggressive",
            "max_rounds": 10,
            "qte_shrink_speed": 100
        }
    }
    
    参数说明:
    - strategy: "aggressive"（逆向拉，快速但耗体力）/ "conservative"（顺向拉，慢但省体力）
    - max_rounds: 最大钓鱼次数，默认 10
    - qte_shrink_speed: QTE 圆圈收缩速度（像素/秒），默认 100
    """
    
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """执行自动钓鱼"""
        print("=" * 60)
        print("[AutoFishing] 自动钓鱼启动")
        
        # 解析参数
        params = _parse_param(argv.custom_action_param)
        strategy = params.get("strategy", "aggressive")
        max_rounds = params.get("max_rounds", 10)
        qte_shrink_speed = params.get("qte_shrink_speed", QTE_CIRCLE_SHRINK_SPEED)
        
        print(f"[AutoFishing] 策略: {strategy}")
        print(f"[AutoFishing] 最大轮数: {max_rounds}")
        print(f"[AutoFishing] QTE收缩速度: {qte_shrink_speed}")
        
        # 保存参数到实例变量
        self._strategy = strategy
        self._qte_shrink_speed = qte_shrink_speed
        
        # 启动调试窗口
        if DEBUG_ENABLED:
            fishing_debug.start_debug()
        
        success_count = 0
        fail_count = 0
        
        try:
            for round_num in range(1, max_rounds + 1):
                print(f"\n[AutoFishing] ===== 第 {round_num}/{max_rounds} 轮钓鱼 =====")
                
                result = self._fishing_loop(context, strategy)
                
                if result:
                    success_count += 1
                    fail_count = 0  # 重置连续失败计数
                    print(f"[AutoFishing] 第 {round_num} 轮钓鱼成功")
                else:
                    fail_count += 1
                    print(f"[AutoFishing] 第 {round_num} 轮钓鱼失败")
                    if fail_count >= 2:
                        print("[AutoFishing] 连续失败，停止钓鱼")
                        break
            
            print(f"\n[AutoFishing] 钓鱼完成: 成功 {success_count} 次")
            return CustomAction.RunResult(success=success_count > 0)
            
        except Exception as e:
            print(f"[AutoFishing] 发生异常: {e}")
            import traceback
            traceback.print_exc()
            return CustomAction.RunResult(success=False)
        
        finally:
            # 关闭调试窗口
            if DEBUG_ENABLED:
                fishing_debug.stop_debug()
    
    def _fishing_loop(self, context: Context, strategy: str) -> bool:
        """
        单次钓鱼循环
        
        返回:
            bool: 是否成功钓到鱼
        """
        # 1. 选择饵料（如果在选饵界面）
        self._select_bait(context)
        
        # 2. 下杆
        print("[AutoFishing] 准备下杆...")
        if not self._cast_rod(context):
            print("[AutoFishing] 下杆失败")
            return False
        
        # 3. 等待上钩
        print("[AutoFishing] 等待鱼上钩...")
        if DEBUG_ENABLED:
            fishing_debug.get_debug_viewer().set_phase("WaitBite")
        
        if not self._wait_for_bite(context):
            print("[AutoFishing] 等待上钩超时")
            return False
        
        # 4. 拉扯阶段（包含 QTE 处理）
        print("[AutoFishing] 进入拉扯阶段...")
        if DEBUG_ENABLED:
            fishing_debug.get_debug_viewer().set_phase("Struggle")
        
        result = self._struggle_phase(context, strategy)
        
        if result == "caught":
            print("[AutoFishing] 成功钓上鱼！")
            # 5. 处理结果
            self._handle_result(context)
            return True
        else:
            print(f"[AutoFishing] 拉扯阶段结果: {result}")
            return False
    
    # ==================== 下杆阶段 ====================
    
    def _cast_rod(self, context: Context) -> bool:
        """点击操控杆下杆"""
        try:
            controller = context.tasker.controller
            print(f"[AutoFishing] 点击操控杆下杆: {JOYSTICK_CENTER}")
            controller.post_click(JOYSTICK_CENTER[0], JOYSTICK_CENTER[1]).wait()
            time.sleep(CAST_ROD_DELAY)
            return True
        except Exception as e:
            print(f"[AutoFishing] 下杆异常: {e}")
            return False
    
    # ==================== 等待上钩阶段 ====================
    
    def _wait_for_bite(self, context: Context, timeout: float = None) -> bool:
        """
        等待鱼上钩，检测操控杆区域蓝色像素
        
        此阶段只做 ColorMatch 检测，不做其他检测以提高性能
        
        返回:
            bool: 是否检测到上钩
        """
        if timeout is None:
            timeout = BITE_TIMEOUT
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 每次循环重新获取 controller 引用，避免 IPC 通信问题
            try:
                controller = context.tasker.controller
                controller.post_screencap().wait()
                time.sleep(0.03)  # 短暂延迟确保截图完成
                image = controller.cached_image
            except Exception as e:
                print(f"[AutoFishing] 等待上钩截图失败: {e}")
                time.sleep(0.1)
                continue
            
            # ColorMatch 检测蓝色像素（MaaFramework 使用 RGB 格式）
            reco = context.run_recognition("bite_check", image, {
                "bite_check": {
                    "recognition": "ColorMatch",
                    "roi": BITE_DETECT_ROI,
                    "lower": BITE_COLOR_LOWER_RGB,
                    "upper": BITE_COLOR_UPPER_RGB,
                    "count": BITE_COLOR_COUNT,
                    "connected": True,  # 要求连通区域
                }
            })
            
            # 调试：打印 reco 返回值 + 像素数量
            detected = reco is not None and reco.hit
            if DEBUG_ENABLED:
                if reco is None:
                    _debug_log("[WaitBite] reco is None!")
                else:
                    count = None
                    try:
                        if hasattr(reco, "best_result") and reco.best_result is not None:
                            count = getattr(reco.best_result, "count", None)
                    except Exception:
                        count = None
                    _debug_log(
                        f"[WaitBite] hit={reco.hit}, "
                        f"best_result={getattr(reco, 'best_result', 'N/A')}, "
                        f"pixel_count={count}"
                    )
            
            # 调试显示（OpenCV 使用 BGR 格式）
            if DEBUG_ENABLED:
                bite_center = (
                    BITE_DETECT_ROI[0] + BITE_DETECT_ROI[2] // 2,
                    BITE_DETECT_ROI[1] + BITE_DETECT_ROI[3] // 2
                )
                fishing_debug.get_debug_viewer().draw_bite_detection(
                    image, BITE_DETECT_ROI, bite_center, detected,
                    color_lower=BITE_COLOR_LOWER_BGR,
                    color_upper=BITE_COLOR_UPPER_BGR
                )
            
            if detected:
                print("[AutoFishing] 检测到鱼上钩！")
                controller.post_click(JOYSTICK_CENTER[0], JOYSTICK_CENTER[1]).wait()
                time.sleep(0.1)
                return True
            
            time.sleep(BITE_CHECK_INTERVAL)
        
        return False
    
    # ==================== 拉扯阶段 ====================
    
    def _struggle_phase(self, context: Context, strategy: str) -> str:
        """
        拉扯阶段
        
        此阶段做的检测：
        - 黄色箭头方向（颜色扫描）
        - QTE 触发（鱼按钮检测）
        - 黑屏检测（判断是否钓上鱼）
        
        不做 OCR 检测以提高性能
        
        返回:
            "caught" - 钓上鱼
            "failed" - 失败
        """
        max_duration = 120.0  # 最长拉扯时间
        start_time = time.time()
        
        while time.time() - start_time < max_duration:
            # 每次循环重新获取 controller 引用，避免 IPC 通信问题
            try:
                controller = context.tasker.controller
                controller.post_screencap().wait()
                time.sleep(0.03)  # 短暂延迟确保截图完成
                image = controller.cached_image
            except Exception as e:
                print(f"[AutoFishing] 拉扯阶段截图失败: {e}")
                time.sleep(0.1)
                continue
            
            # 1. 快速黑屏预筛（≤5ms），若命中再做 OCR 确认结算页面
            if self._check_screen_blackout(image):
                # 等黑屏过渡结束后 OCR 确认
                time.sleep(0.5)
                try:
                    controller.post_screencap().wait()
                    time.sleep(0.03)
                    confirm_img = controller.cached_image
                    if self._check_result_screen(context, confirm_img):
                        print("[AutoFishing] 检测到结算页面，钓上鱼！")
                        return "caught"
                except Exception:
                    pass
                # 黑屏但不是结算，可能是过场动画，继续
            
            # 2. 检测是否进入 QTE（鱼按钮出现）
            qte_buttons = self._detect_all_fish_buttons(context, image)
            if qte_buttons:
                _debug_log(f"[Struggle] 检测到 QTE，共 {len(qte_buttons)} 个按钮")
                if DEBUG_ENABLED:
                    fishing_debug.get_debug_viewer().set_phase("QTE")
                
                # 动态处理 QTE
                qte_result = self._handle_qte_dynamic(context)
                
                if DEBUG_ENABLED:
                    fishing_debug.get_debug_viewer().set_phase("Struggle")
                
                if qte_result == "caught":
                    return "caught"
                # QTE 完成后，等待游戏 UI 完全过渡，避免残留按钮图像被重新检测
                _debug_log("[Struggle] QTE 结束，等待 1.5s 冷却")
                time.sleep(1.5)
                continue
            
            # 3. 检测黄色箭头方向并拖拽
            t_arrow_start = time.time()
            arrow_result = self._detect_arrow_direction_by_color(image)
            if arrow_result is not None:
                angle, debug_info = arrow_result
                
                # 根据策略计算拖拽方向
                if strategy == "aggressive":
                    drag_angle = angle + 180  # 逆向
                else:
                    drag_angle = angle  # 顺向
                
                # 调试显示
                if DEBUG_ENABLED and debug_info:
                    fishing_debug.get_debug_viewer().draw_arrow_detection(
                        image,
                        debug_info['left_point'],
                        debug_info['right_point'],
                        debug_info['mid_point'],
                        JOYSTICK_CENTER,
                        angle,
                        roi=_clamp_roi(ARROW_DETECT_ROI)
                    )
                
                # 记录从箭头检测开始到准备拖拽的耗时
                t_before_drag = time.time()
                detect_to_drag_ms = (t_before_drag - t_arrow_start) * 1000
                _debug_log(
                    f"[Struggle] 箭头检测→拖拽: {detect_to_drag_ms:.0f}ms, "
                    f"angle={angle:.1f}°, drag_angle={drag_angle:.1f}°"
                )
                
                self._drag_joystick(context, drag_angle)
            
            time.sleep(STRUGGLE_CHECK_INTERVAL)
        
        return "failed"
    
    def _detect_arrow_direction_by_color(self, image: np.ndarray) -> Optional[Tuple[float, dict]]:
        """
        通过颜色扫描检测黄色箭头方向（性能优化版）
        
        算法:
        1. 在 ARROW_DETECT_ROI 内提取黄色像素（BGR 颜色空间）
        2. 使用 cv2.findNonZero 高效获取非零像素坐标
        3. 直接用 numpy 向量化操作获取最左/最右像素
        4. 从操控杆中心到中点的方向即为箭头方向
        
        返回:
            (angle, debug_info): 角度（度）和调试信息，None 表示未检测到
        """
        # 裁剪 ROI 区域（使用切片视图，不复制内存）
        roi = _clamp_roi(ARROW_DETECT_ROI)
        roi_img = image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        
        # 直接在 BGR 颜色空间检测黄色
        mask = cv2.inRange(roi_img, ARROW_COLOR_LOWER_BGR, ARROW_COLOR_UPPER_BGR)
        
        # 使用 cv2.findNonZero 比 np.where 更高效
        # 返回 shape (N, 1, 2) 的数组，每个元素是 [x, y]
        points = cv2.findNonZero(mask)
        if points is None or len(points) < ARROW_MIN_PIXELS:
            return None
        
        pixel_count = len(points)
        
        # 直接获取 x 坐标数组（无需 reshape，直接切片）
        x_coords = points[:, 0, 0]
        
        # 使用 numpy 向量化操作找极值索引
        left_idx = x_coords.argmin()
        right_idx = x_coords.argmax()
        
        # 获取对应的 y 坐标并转换为全图坐标
        left_x = int(points[left_idx, 0, 0]) + roi[0]
        left_y = int(points[left_idx, 0, 1]) + roi[1]
        right_x = int(points[right_idx, 0, 0]) + roi[0]
        right_y = int(points[right_idx, 0, 1]) + roi[1]
        
        # 计算中点（使用整数运算加速）
        mid_x = (left_x + right_x) * 0.5
        mid_y = (left_y + right_y) * 0.5
        
        # 计算从操控杆中心到中点的角度
        dx = mid_x - JOYSTICK_CENTER[0]
        dy = mid_y - JOYSTICK_CENTER[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        # 只在调试模式下构造 debug_info 和打印
        debug_info = None
        if DEBUG_ENABLED:
            debug_info = {
                'left_point': (left_x, left_y),
                'right_point': (right_x, right_y),
                'mid_point': (mid_x, mid_y),
                'pixel_count': pixel_count
            }
            print(f"[AutoFishing] 箭头检测: 像素数={pixel_count}, 角度={angle:.1f}°")
        
        return (angle, debug_info)
    
    def _drag_joystick(self, context: Context, angle_deg: float, duration_ms: int = None):
        """
        拖拽摇杆到指定角度方向
        
        参数:
            context: MAA 上下文
            angle_deg: 目标角度（度）
            duration_ms: 滑动持续时间（毫秒）
        """
        if duration_ms is None:
            duration_ms = SWIPE_DURATION
        
        rad = math.radians(angle_deg)
        end_x = int(JOYSTICK_CENTER[0] + JOYSTICK_RADIUS * math.cos(rad))
        end_y = int(JOYSTICK_CENTER[1] + JOYSTICK_RADIUS * math.sin(rad))
        
        _safe_swipe(context, 
                    JOYSTICK_CENTER[0], JOYSTICK_CENTER[1],
                    end_x, end_y, duration_ms)
    
    # ==================== QTE 阶段（动态规划）====================
    
    def _detect_all_fish_buttons(self, context: Context, image: np.ndarray) -> List[dict]:
        """
        检测所有 QTE 鱼形按钮（最多取置信度最高的 3 个）
        
        返回:
            List[dict]: 按钮列表，每个包含 box、center、radius、confidence
        """
        buttons = []
        
        reco = context.run_recognition("qte_fish_detect", image, {
            "qte_fish_detect": {
                "recognition": "TemplateMatch",
                "template": QTE_FISH_TEMPLATE,
                "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],
                "threshold": QTE_TEMPLATE_THRESHOLD,
            }
        })
        
        if reco and reco.hit:
            # 收集所有匹配结果
            results = []
            if hasattr(reco, 'all_results') and reco.all_results:
                results = list(reco.all_results)
            elif reco.best_result:
                results = [reco.best_result]
            
            # 按置信度排序（如果有 score 属性）
            results_with_score = []
            for result in results:
                box = result.box
                center = _box_center(box)
                if center:
                    # 尝试获取置信度
                    score = getattr(result, 'score', 1.0)
                    results_with_score.append({
                        'box': box,
                        'center': center,
                        'radius': max(box[2], box[3]) // 2 + 20,
                        'confidence': score
                    })
            
            # 按置信度降序排序，取前 3 个
            results_with_score.sort(key=lambda x: x['confidence'], reverse=True)
            buttons = results_with_score[:QTE_BUTTON_COUNT]
        
        return buttons
    
    def _detect_circle_radius(self, image: np.ndarray, center: Tuple[int, int], button_radius: int) -> float:
        """
        检测按钮周围的白色圆圈半径（径向扫描法）
        
        从按钮中心向外扫描多个方向，找到白色像素的最大距离
        为避免白云等背景干扰：检测到的边缘往外 10px 如果还是白色，则丢弃该方向结果
        （圆圈厚度约 7px，往外 10px 必然超出圆圈范围）
        
        参数:
            image: 截图
            center: 按钮中心坐标
            button_radius: 按钮半径（扫描起点）
            
        返回:
            float: 检测到的圆圈半径，0 表示未检测到
        """
        cx, cy = center
        h, w = image.shape[:2]
        
        # 从按钮边缘外一点开始扫描
        start_radius = button_radius + 5
        # 不再用屏幕边缘裁切 max_radius —— 每个方向独立处理越界
        max_radius = QTE_CIRCLE_MAX_RADIUS
        
        detected_radii = []  
        skipped_oob = 0  # 统计因越界跳过的方向数
        
        def is_white_pixel(px: int, py: int) -> bool:
            """检查像素是否为白色"""
            if not (0 <= px < w and 0 <= py < h):
                return False
            pixel = image[py, px]
            return (pixel[0] >= QTE_CIRCLE_COLOR_LOWER[0] and 
                    pixel[1] >= QTE_CIRCLE_COLOR_LOWER[1] and 
                    pixel[2] >= QTE_CIRCLE_COLOR_LOWER[2] and
                    pixel[0] <= QTE_CIRCLE_COLOR_UPPER[0] and 
                    pixel[1] <= QTE_CIRCLE_COLOR_UPPER[1] and 
                    pixel[2] <= QTE_CIRCLE_COLOR_UPPER[2])
        
        for i in range(QTE_CIRCLE_SCAN_DIRECTIONS):
            angle = 2 * math.pi * i / QTE_CIRCLE_SCAN_DIRECTIONS
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            last_white_radius = 0
            has_any_inbound = False  # 该方向是否有任何像素在屏幕内
            
            # 沿着这个方向从外向内扫描，找到最外层的白色像素
            for r in range(max_radius, start_radius, -1):
                px = int(cx + r * cos_a)
                py = int(cy + r * sin_a)
                
                if 0 <= px < w and 0 <= py < h:
                    has_any_inbound = True
                    if is_white_pixel(px, py):
                        last_white_radius = r
                        break  # 找到最外层白色像素即可停止
            
            if not has_any_inbound:
                # 该方向的所有扫描点都在屏幕外，跳过不计入统计
                skipped_oob += 1
                continue
            
            if last_white_radius > 0:
                # 验证：检查边缘往外 10px 的像素（圆圈厚度约 7px）
                verify_radius = last_white_radius + 10
                verify_px = int(cx + verify_radius * cos_a)
                verify_py = int(cy + verify_radius * sin_a)
                
                if is_white_pixel(verify_px, verify_py):
                    # 外侧还是白色，可能是背景干扰，丢弃该方向结果
                    continue
                
                # 验证通过，保留结果
                detected_radii.append(last_white_radius)
        
        # 有效方向数 = 总方向 - 越界跳过的方向
        valid_directions = QTE_CIRCLE_SCAN_DIRECTIONS - skipped_oob
        
        # 需要至少一半的 *有效方向* 检测到圆圈（排除越界方向）
        min_required = max(valid_directions // 2, 4)  # 至少 4 个方向
        if len(detected_radii) >= min_required:
            radius = sum(detected_radii) / len(detected_radii)
            if DEBUG_ENABLED:
                _debug_log(
                    f"[QTE-Circle] detected radius={radius:.1f}, "
                    f"directions={len(detected_radii)}/{valid_directions}"
                    f"(total={QTE_CIRCLE_SCAN_DIRECTIONS}, oob={skipped_oob}), "
                    f"center={center}"
                )
            return radius
        
        if DEBUG_ENABLED:
            _debug_log(
                f"[QTE-Circle] insufficient directions: "
                f"{len(detected_radii)}/{valid_directions}"
                f"(total={QTE_CIRCLE_SCAN_DIRECTIONS}, oob={skipped_oob}), "
                f"center={center}"
            )
        return 0.0
    
    def _handle_qte_dynamic(self, context: Context) -> str:
        """
        动态处理 QTE 阶段（优化版：每个按钮只检测一次外圆）
        
        策略:
        1. 使用固定 3 槽位表存储按钮信息
        2. 首次检测到按钮时，同时检测外圆大小，计算精确点击时间
        3. 外圆已检测的按钮后续帧不再检测圆圈
        4. 根据计算好的点击时间戳执行点击
        5. 超过时间窗口（500ms）报超时警告
        
        返回:
            "done" - QTE 完成
            "caught" - 检测到黑屏（钓上鱼）
        """
        # 固定 3 槽位表
        qte_slots: List[Optional[QTEButtonSlot]] = [None, None, None]
        
        def find_slot_by_center(center: Tuple[int, int], tolerance: int = 30) -> int:
            """查找按钮对应的槽位索引，-1 表示未找到"""
            for i, slot in enumerate(qte_slots):
                if slot is not None:
                    dx = abs(slot.center[0] - center[0])
                    dy = abs(slot.center[1] - center[1])
                    if dx < tolerance and dy < tolerance:
                        return i
            return -1
        
        def find_empty_slot() -> int:
            """查找空槽位索引，-1 表示已满"""
            for i, slot in enumerate(qte_slots):
                if slot is None:
                    return i
            return -1
        
        def count_clicked() -> int:
            """统计已点击的按钮数"""
            return sum(1 for slot in qte_slots if slot is not None and slot.clicked)
        
        start_time = time.time()
        
        while time.time() - start_time < QTE_MAX_DURATION:
            # 每次循环重新获取 controller 引用，避免 IPC 通信问题
            try:
                controller = context.tasker.controller
                controller.post_screencap().wait()
                capture_time = time.time()  # 截图完成后立即记录时间戳（不受后续 sleep 影响）
                time.sleep(0.03)  # 短暂延迟确保 IPC 稳定
                image = controller.cached_image
            except Exception as e:
                _debug_log(f"[QTE] 截图失败: {e}")
                time.sleep(0.1)
                continue
            
            # 快速黑屏预筛（≤5ms），若命中再做 OCR 确认
            if self._check_screen_blackout(image):
                time.sleep(0.5)
                try:
                    controller.post_screencap().wait()
                    time.sleep(0.03)
                    confirm_img = controller.cached_image
                    if self._check_result_screen(context, confirm_img):
                        _debug_log("[QTE] 检测到结算页面，钓上鱼！")
                        return "caught"
                except Exception:
                    pass
            
            # 如果 3 个槽位都已分配且都已有圆圈数据，跳过模板匹配以加速
            all_slots_filled = all(s is not None for s in qte_slots)
            all_circles_known = all_slots_filled and all(
                s.circle_radius > 0 or s.clicked for s in qte_slots  # type: ignore
            )
            
            if not all_circles_known:
                # 检测当前所有按钮
                buttons = self._detect_all_fish_buttons(context, image)
            else:
                buttons = []  # 跳过模板匹配，直接进入点击判断
            
            # 如果没有检测到按钮且不是跳过了检测
            if not buttons and not all_slots_filled:
                if count_clicked() >= QTE_BUTTON_COUNT:
                    _debug_log("[QTE] 所有按钮已点击完成")
                    return "done"
                # 可能还在等待新按钮出现
                time.sleep(QTE_CHECK_INTERVAL)
                continue
            
            # 处理检测到的按钮
            for btn in buttons:
                center = btn['center']
                slot_idx = find_slot_by_center(center)
                
                if slot_idx == -1:
                    # 新按钮，尝试分配槽位
                    empty_idx = find_empty_slot()
                    if empty_idx == -1:
                        continue  # 槽位已满，忽略
                    
                    # 创建新槽位
                    button_radius = btn['radius']
                    slot = QTEButtonSlot(
                        center=center,
                        button_radius=button_radius,
                        first_detect_time=capture_time,
                        confidence=btn.get('confidence', 1.0)
                    )
                    
                    # 检测外圆大小
                    circle_radius = self._detect_circle_radius(image, center, button_radius)
                    
                    if circle_radius > 0:
                        # 检测到圆圈，计算点击时间
                        # 修正半径：减去辉光边缘导致的检测偏大
                        corrected_radius = circle_radius - QTE_RADIUS_CORRECTION
                        slot.circle_radius = circle_radius
                        shrink_distance = max(0, corrected_radius - button_radius)
                        wait_time = shrink_distance / self._qte_shrink_speed
                        slot.target_click_time = capture_time + wait_time + QTE_CLICK_OFFSET
                        
                        _debug_log(
                            f"[QTE] 槽位[{empty_idx}]: 新按钮 {center}, "
                            f"外圆={circle_radius:.0f}→{corrected_radius:.0f}px, 按钮R={button_radius}px, "
                            f"shrink={shrink_distance:.0f}px, 等待={wait_time:.2f}s, "
                            f"target_t=+{slot.target_click_time - start_time:.2f}s"
                        )
                    else:
                        # 未检测到圆圈，后续帧继续尝试
                        _debug_log(f"[QTE] 槽位[{empty_idx}]: 新按钮 {center}, 外圆未出现")
                    
                    qte_slots[empty_idx] = slot
                
                else:
                    # 已存在的按钮
                    slot = qte_slots[slot_idx]
                    if slot is None or slot.clicked:
                        continue
                    
                    # 如果外圆还未检测到，再次尝试
                    if slot.circle_radius == 0:
                        circle_radius = self._detect_circle_radius(image, slot.center, slot.button_radius)
                        if circle_radius > 0:
                            corrected_radius = circle_radius - QTE_RADIUS_CORRECTION
                            slot.circle_radius = circle_radius
                            shrink_distance = max(0, corrected_radius - slot.button_radius)
                            wait_time = shrink_distance / self._qte_shrink_speed
                            slot.target_click_time = capture_time + wait_time + QTE_CLICK_OFFSET
                            
                            _debug_log(
                                f"[QTE] 槽位[{slot_idx}]: 外圆检测到 "
                                f"{circle_radius:.0f}→{corrected_radius:.0f}px, "
                                f"shrink={shrink_distance:.0f}px, 等待={wait_time:.2f}s, "
                                f"target_t=+{slot.target_click_time - start_time:.2f}s"
                            )
            
            # 调试显示
            if DEBUG_ENABLED and buttons:
                clicked_set = set(slot.center for slot in qte_slots if slot is not None and slot.clicked)
                fishing_debug.get_debug_viewer().draw_qte_buttons(
                    image, buttons, clicked_set
                )
            
            # ======== 精准定时点击：用 time.sleep() 代替逐帧轮询 ========
            # 按 target_click_time 排序，先点时间最早的
            pending_slots = [
                (i, s) for i, s in enumerate(qte_slots) 
                if s is not None and not s.clicked and s.circle_radius > 0
            ]
            pending_slots.sort(key=lambda x: x[1].target_click_time)
            
            for i, slot in pending_slots:
                now = time.time()
                wait = slot.target_click_time - now
                
                if wait > 0.5:
                    # 距离点击时间还远（>500ms），先跳出去取下一帧
                    # 继续检测其他按钮的圆圈
                    break
                
                if wait > 0:
                    # 距离点击时间 0~500ms，精确 sleep 到目标时刻
                    time.sleep(wait)
                
                # 执行点击（此时应该非常接近 target_click_time）
                click_time = time.time()
                time_diff = click_time - slot.target_click_time
                
                if time_diff <= QTE_CLICK_WINDOW:
                    _debug_log(
                        f"[QTE] 槽位[{i}]: 点击 {slot.center}, "
                        f"time_diff={time_diff*1000:.0f}ms, "
                        f"elapsed=+{click_time - start_time:.2f}s"
                    )
                    controller.post_click(slot.center[0], slot.center[1]).wait()
                    slot.clicked = True
                    time.sleep(QTE_BUTTON_CLICK_DELAY)
                    # 不 break，继续检查下一个按钮（可能也该点了）
                else:
                    # 超时警告
                    _debug_log(
                        f"[QTE] 槽位[{i}]: 超时！{slot.center}, "
                        f"超时={time_diff*1000:.0f}ms"
                    )
                    slot.clicked = True  # 标记为已处理，避免重复报错
            
            # 检查是否所有按钮都已处理
            if all_slots_filled and count_clicked() >= QTE_BUTTON_COUNT:
                _debug_log("[QTE] 所有按钮已点击完成")
                # 等待游戏 QTE 动画结束，避免残留图像被重新检测
                time.sleep(1.0)
                return "done"
            
            # 短暂等待
            time.sleep(QTE_CHECK_INTERVAL)
        
        _debug_log("[QTE] 整体超时")
        return "done"
    
    def _calculate_qte_wait_time_fallback(self, button_radius: int) -> float:
        """
        计算 QTE 点击等待时间（备用方法，当外圆检测失败时使用）
        
        基于按钮半径和收缩速度估算从圆圈出现到收缩到按钮边缘的时间
        
        注意: 此方法假设初始圆圈半径为按钮半径的 2 倍，不如实际检测精确
        
        参数:
            button_radius: 按钮半径
            
        返回:
            float: 等待时间（秒）
        """
        # 假设初始圆圈半径约为按钮半径的 2 倍
        initial_radius = button_radius * 2
        shrink_distance = initial_radius - button_radius
        wait_time = shrink_distance / self._qte_shrink_speed
        
        return max(0, wait_time + QTE_CLICK_OFFSET)
    
    # ==================== 结算页面检测 ====================
    
    def _check_result_screen(self, context: Context, image: np.ndarray) -> bool:
        """
        检测是否进入结算页面（钓上鱼后的结果展示）
        
        通过全屏 OCR 识别"特质"或"重量"来判断
        
        返回:
            bool: 是否在结算页面
        """
        try:
            reco = context.run_recognition("result_check", image, {
                "result_check": {
                    "recognition": "OCR",
                    "expected": RESULT_OCR_EXPECTED,
                    "roi": RESULT_OCR_ROI,
                }
            })
            detected = reco is not None and reco.hit
            if DEBUG_ENABLED and detected:
                text = ""
                try:
                    if hasattr(reco, "best_result") and reco.best_result is not None:
                        text = getattr(reco.best_result, "text", "")
                except Exception:
                    pass
                _debug_log(f"[ResultScreen] 检测到结算页面, OCR text={text}")
            return detected
        except Exception as e:
            if DEBUG_ENABLED:
                _debug_log(f"[ResultScreen] OCR 检测异常: {e}")
            return False
    
    def _check_screen_blackout(self, image: np.ndarray) -> bool:
        """
        检测屏幕是否进入黑屏（过渡动画）
        
        通过检测大面积低亮度像素判断，比 OCR 更快
        
        返回:
            bool: 是否黑屏
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray < BLACKOUT_DARK_THRESHOLD) / gray.size
        is_black = dark_ratio > BLACKOUT_RATIO_THRESHOLD
        # 在调试模式下记录接近阈值的情况，便于后续调整参数
        if DEBUG_ENABLED and (0.4 <= dark_ratio <= 0.8):
            _debug_log(
                f"[Blackout] dark_ratio={dark_ratio:.3f}, "
                f"threshold={BLACKOUT_RATIO_THRESHOLD}, is_black={is_black}"
            )
        return is_black
    
    # ==================== 结果处理 ====================
    
    def _handle_result(self, context: Context) -> bool:
        """
        处理钓鱼结果界面
        
        结算页面已经在 _struggle_phase / _handle_qte_dynamic 中通过 OCR 确认
        直接点击屏幕继续，然后等待回到选饵界面
        
        返回:
            bool: 是否成功处理
        """
        if DEBUG_ENABLED:
            fishing_debug.get_debug_viewer().set_phase("Result")
        
        print("[AutoFishing] 处理结算界面，点击继续...")
        time.sleep(0.5)  # 等待界面稳定
        
        try:
            controller = context.tasker.controller
            controller.post_click(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2).wait()
        except Exception as e:
            print(f"[AutoFishing] 结算界面点击失败: {e}")
            return False
        
        time.sleep(RESULT_CLICK_DELAY)
        
        # 等待回到选饵界面或其他后续界面
        print("[AutoFishing] 等待结算页面消失...")
        for _ in range(20):  # 最多等待 6 秒
            try:
                controller = context.tasker.controller
                controller.post_screencap().wait()
                time.sleep(0.03)
                image = controller.cached_image
            except Exception as e:
                print(f"[AutoFishing] 结算等待截图失败: {e}")
                time.sleep(0.1)
                continue
            
            # 如果不再检测到"特质""重量"，说明结算页面已消失
            if not self._check_result_screen(context, image):
                print("[AutoFishing] 结算页面已消失")
                return True
            
            # 继续点击确保翻页
            controller.post_click(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2).wait()
            time.sleep(RESULT_CHECK_INTERVAL)
        
        print("[AutoFishing] 结算界面处理超时")
        return False
    
    def _select_bait(self, context: Context) -> bool:
        """
        选择饵料
        
        如果在选饵界面（检测到数字），点击确认开始钓鱼
        
        返回:
            bool: 是否成功选择饵料
        """
        try:
            controller = context.tasker.controller
            controller.post_screencap().wait()
            time.sleep(0.03)
            image = controller.cached_image
        except Exception as e:
            print(f"[AutoFishing] 选饵界面截图失败: {e}")
            return False
        
        # OCR 检测选饵界面（检测任意数字）
        reco = context.run_recognition("bait_select", image, {
            "bait_select": {
                "recognition": "OCR",
                "expected": BAIT_SELECT_EXPECTED,
                "roi": BAIT_SELECT_ROI,
            }
        })
        
        if reco and reco.hit:
            print("[AutoFishing] 检测到选饵界面")
            # 点击屏幕中央开始钓鱼
            controller.post_click(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2).wait()
            time.sleep(0.5)
            return True
        
        return False
