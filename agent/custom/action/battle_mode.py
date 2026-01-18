"""
战斗模式管理 Custom Action
用于检测和切换自动/手动战斗状态
"""

import time
import json
import numpy as np
from maa.custom_action import CustomAction
from maa.context import Context


class BattleModeManager(CustomAction):
    """
    战斗模式智能管理器
    
    功能：
    1. 每秒检测当前战斗状态，不符合目标则点击切换
    2. 切换到目标模式后等待3秒（等待角色动画结束）
    3. 手动模式下持续检测并点击跳过按钮直至消失
    4. 幂等性保证：同一战斗中只执行一次实际切换操作
    
    Pipeline 调用示例:
    {
        "action": "Custom",
        "custom_action": "BattleModeManager",
        "custom_action_param": {"target_mode": "auto"}
    }
    
    参数说明:
    - target_mode: 目标模式，"auto"(自动战斗) 或 "manual_skip"(手动+跳过)
    """

    # 图片路径（相对于 image 文件夹）
    AUTO_MODE_TEMPLATE = "battle/自动战斗状态.png"
    MANUAL_MODE_TEMPLATE = "battle/手动战斗状态.png"
    SKIP_BUTTON_TEMPLATE = "battle/跳过战斗按钮.png"
    
    # 右上角战斗模式图标的 ROI
    MODE_ICON_ROI = [901,0,378,263]
    # 跳过按钮的 ROI
    SKIP_BUTTON_ROI = [901,0,378,263]
    
    # 同一战斗的判定阈值（秒）
    _SAME_BATTLE_THRESHOLD: float = 30.0
    # auto 模式下的同一战斗判定阈值（秒）
    _AUTO_SAME_BATTLE_THRESHOLD: float = 180.0
    # 检测间隔（秒）
    _DETECT_INTERVAL: float = 5.0
    # 切换成功后等待时间（秒）- 等待角色动画结束
    _POST_SWITCH_DELAY: float = 3.0
    # 最大切换尝试次数
    _MAX_SWITCH_ATTEMPTS: int = 10
    # 最大跳过尝试次数
    _MAX_SKIP_ATTEMPTS: int = 30
    # 重复执行日志节流（秒）
    _SKIP_LOG_INTERVAL: float = 15.0
    # auto 模式下的低频轮询间隔（秒）
    _AUTO_WATCH_INTERVAL: float = 4.0
    # auto 模式下画面未变化时的胜利检测间隔（秒）
    _AUTO_VICTORY_CHECK_INTERVAL: float = 10.0
    # 画面变化检测采样步长（像素）
    _FRAME_SAMPLE_STEP: int = 32
    # 平均差异阈值（越大越“宽松”）
    _AUTO_FRAME_DIFF_THRESHOLD: float = 3.0
    # 胜利/结算 OCR 检测 ROI（全屏）
    _VICTORY_CHECK_ROI = [
            738,
            16,
            438,
            218
        ]

    def __init__(self):
        super().__init__()
        self._last_execution_time: float = 0.0
        self._last_target_mode: str = ""
        self._last_skip_log_time: float = 0.0
        self._last_auto_frame_sample = None
        self._last_auto_frame_time: float = 0.0
        self._last_auto_victory_check_time: float = 0.0

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """执行战斗模式管理逻辑"""
        
        # 解析参数 - 处理 JSON 字符串
        raw_param = argv.custom_action_param
        if isinstance(raw_param, dict):
            params = raw_param
        elif isinstance(raw_param, str) and raw_param.strip():
            try:
                params = json.loads(raw_param)
            except json.JSONDecodeError:
                print(f"[BattleModeManager] JSON 解析失败: {raw_param}")
                params = {}
        else:
            params = {}
        
        target_mode = params.get("target_mode", "auto")
        current_time = time.time()
        
        # 幂等性检查
        same_battle_threshold = (
            self._AUTO_SAME_BATTLE_THRESHOLD if target_mode == "auto" else self._SAME_BATTLE_THRESHOLD
        )
        time_since_last = current_time - self._last_execution_time
        if (time_since_last < same_battle_threshold and 
            self._last_target_mode == target_mode):
            if target_mode == "auto":
                self._auto_watch_repeat(context)
            if current_time - self._last_skip_log_time >= self._SKIP_LOG_INTERVAL:
                print(f"[BattleModeManager] 跳过重复执行（{time_since_last:.1f}秒内已执行）")
                self._last_skip_log_time = current_time
            return CustomAction.RunResult(success=True)
        
        print(f"[BattleModeManager] 目标模式: {target_mode}")
        
        # 确定目标状态
        want_manual = (target_mode == "manual_skip")
        
        # 步骤1: 循环检测并切换到目标模式
        if not self._switch_to_target_mode(context, want_manual):
            print("[BattleModeManager] 切换模式失败或未检测到战斗界面")
            return CustomAction.RunResult(success=True)
        
        # 步骤2: 等待角色动画结束
        print(f"[BattleModeManager] 已切换到目标模式，等待 {self._POST_SWITCH_DELAY} 秒...")
        time.sleep(self._POST_SWITCH_DELAY)
        
        # 步骤3: 如果是手动跳过模式，持续点击跳过按钮
        if want_manual:
            self._perform_skip_loop(context)
        else:
            self._refresh_auto_frame_sample(context)
        
        # 更新执行状态
        self._last_execution_time = current_time
        self._last_target_mode = target_mode
        
        return CustomAction.RunResult(success=True)

    def _detect_current_mode(self, context: Context) -> tuple[str | None, any]:
        """
        检测当前战斗模式
        返回: (模式字符串, 识别结果对象) - 模式为 "auto", "manual", 或 None（未检测到）
        """
        image = context.tasker.controller.post_screencap().wait().get()
        
        # 检测自动战斗状态
        auto_reco_param = {
            "BattleMode_DetectAuto_Internal": {
                "recognition": "TemplateMatch",
                "template": self.AUTO_MODE_TEMPLATE,
                "roi": self.MODE_ICON_ROI
            }
        }
        
        auto_result = context.run_recognition(
            "BattleMode_DetectAuto_Internal",
            image,
            auto_reco_param
        )
        
        # 检查 hit 属性，而不是仅检查对象是否为 None
        if auto_result is not None and auto_result.hit:
            return ("auto", auto_result)
        
        # 检测手动战斗状态
        manual_reco_param = {
            "BattleMode_DetectManual_Internal": {
                "recognition": "TemplateMatch",
                "template": self.MANUAL_MODE_TEMPLATE,
                "roi": self.MODE_ICON_ROI
            }
        }
        
        manual_result = context.run_recognition(
            "BattleMode_DetectManual_Internal",
            image,
            manual_reco_param
        )
        
        # 检查 hit 属性
        if manual_result is not None and manual_result.hit:
            return ("manual", manual_result)
        
        return (None, None)

    def _click_mode_toggle(self, context: Context, box=None) -> None:
        """点击战斗模式切换按钮"""
        if box is not None:
            # 使用识别到的精确位置
            x = box.x + box.w // 2
            y = box.y + box.h // 2
        else:
            # 使用 ROI 中心作为备选（不推荐）
            x = self.MODE_ICON_ROI[0] + self.MODE_ICON_ROI[2] // 2
            y = self.MODE_ICON_ROI[1] + self.MODE_ICON_ROI[3] // 2
        
        context.tasker.controller.post_click(x, y).wait()
        print(f"[BattleModeManager] 点击切换战斗模式 ({x}, {y})")

    def _switch_to_target_mode(self, context: Context, want_manual: bool) -> bool:
        """
        循环检测并切换到目标模式
        每秒检测一次，不符合则点击切换
        返回: 是否成功切换到目标模式
        """
        target_str = "manual" if want_manual else "auto"
        
        for attempt in range(self._MAX_SWITCH_ATTEMPTS):
            current_mode, reco_result = self._detect_current_mode(context)
            print(f"[BattleModeManager] 第 {attempt + 1} 次检测，当前模式: {current_mode}")
            
            if current_mode is None:
                print("[BattleModeManager] 未检测到战斗模式图标，可能不在战斗中")
                return False
            
            # 检查是否已经是目标模式
            if (want_manual and current_mode == "manual") or \
               (not want_manual and current_mode == "auto"):
                print(f"[BattleModeManager] 已处于目标模式: {target_str}")
                return True
            
            # 不是目标模式，使用识别结果的精确位置点击切换
            box = reco_result.box if reco_result and hasattr(reco_result, 'box') else None
            self._click_mode_toggle(context, box)
            time.sleep(self._DETECT_INTERVAL)
        
        print(f"[BattleModeManager] 达到最大尝试次数 {self._MAX_SWITCH_ATTEMPTS}，切换失败")
        return False

    def _detect_skip_button(self, context: Context):
        """检测跳过按钮，返回识别结果或 None"""
        image = context.tasker.controller.post_screencap().wait().get()
        
        skip_result = context.run_recognition(
            "BattleMode_DetectSkip_Internal",
            image,
            {
                "BattleMode_DetectSkip_Internal": {
                    "recognition": "TemplateMatch",
                    "template": self.SKIP_BUTTON_TEMPLATE,
                    "roi": self.SKIP_BUTTON_ROI,
                    "threshold": 0.7
                }
            }
        )
        
        # 只有当 hit=True 时才返回结果
        if skip_result is not None and skip_result.hit:
            return skip_result
        return None

    def _sample_frame(self, image):
        """对画面进行稀疏采样，用于轻量变化检测"""
        if image is None:
            return None
        try:
            return image[::self._FRAME_SAMPLE_STEP, ::self._FRAME_SAMPLE_STEP, 1].astype(np.int16)
        except Exception:
            return None

    def _detect_mode_on_image(self, context: Context, image) -> tuple[str | None, any]:
        """在指定截图上检测当前战斗模式"""
        auto_reco_param = {
            "BattleMode_DetectAuto_Internal": {
                "recognition": "TemplateMatch",
                "template": self.AUTO_MODE_TEMPLATE,
                "roi": self.MODE_ICON_ROI
            }
        }
        
        auto_result = context.run_recognition(
            "BattleMode_DetectAuto_Internal",
            image,
            auto_reco_param
        )
        
        if auto_result is not None and auto_result.hit:
            return ("auto", auto_result)
        
        manual_reco_param = {
            "BattleMode_DetectManual_Internal": {
                "recognition": "TemplateMatch",
                "template": self.MANUAL_MODE_TEMPLATE,
                "roi": self.MODE_ICON_ROI
            }
        }
        
        manual_result = context.run_recognition(
            "BattleMode_DetectManual_Internal",
            image,
            manual_reco_param
        )
        
        if manual_result is not None and manual_result.hit:
            return ("manual", manual_result)
        
        return (None, None)

    def _detect_victory_settlement(self, context: Context, image) -> bool:
        """检测是否出现胜利/结算相关文案"""
        victory_reco = context.run_recognition(
            "BattleMode_VictoryCheck",
            image,
            {
                "BattleMode_VictoryCheck": {
                    "recognition": "OCR",
                    "expected": "点击.*继续|返回|胜利|结算",
                    "roi": self._VICTORY_CHECK_ROI,
                }
            }
        )
        
        return victory_reco is not None and victory_reco.hit

    def _refresh_auto_frame_sample(self, context: Context) -> None:
        """切换到 auto 后刷新一帧基准画面"""
        context.tasker.controller.post_screencap().wait()
        image = context.tasker.controller.cached_image
        self._last_auto_frame_sample = self._sample_frame(image)
        self._last_auto_frame_time = time.time()

    def _auto_watch_repeat(self, context: Context) -> None:
        """
        auto 模式下的轻量轮询：
        - 低频截帧检测画面变化
        - 若画面未变化且到达间隔，则尝试检测胜利/结算
        """
        now = time.time()
        if now - self._last_auto_frame_time < self._AUTO_WATCH_INTERVAL:
            return
        
        context.tasker.controller.post_screencap().wait()
        image = context.tasker.controller.cached_image
        sample = self._sample_frame(image)
        if sample is None:
            self._last_auto_frame_time = now
            return
        
        if self._last_auto_frame_sample is None:
            self._last_auto_frame_sample = sample
            self._last_auto_frame_time = now
            return
        
        diff = float(np.mean(np.abs(sample - self._last_auto_frame_sample)))
        self._last_auto_frame_time = now
        
        if diff >= self._AUTO_FRAME_DIFF_THRESHOLD:
            self._last_auto_frame_sample = sample
            mode, reco_result = self._detect_mode_on_image(context, image)
            if mode == "manual":
                box = reco_result.box if reco_result and hasattr(reco_result, 'box') else None
                self._click_mode_toggle(context, box)
            return
        
        if now - self._last_auto_victory_check_time >= self._AUTO_VICTORY_CHECK_INTERVAL:
            self._last_auto_victory_check_time = now
            if self._detect_victory_settlement(context, image):
                print("[BattleModeManager] 检测到胜利/结算文案")

    def _perform_skip_loop(self, context: Context) -> None:
        """
        持续检测并点击跳过按钮，直到按钮消失
        每秒检测一次
        """
        print("[BattleModeManager] 开始跳过按钮检测循环")
        
        for attempt in range(self._MAX_SKIP_ATTEMPTS):
            skip_result = self._detect_skip_button(context)
            
            if skip_result is None:
                print(f"[BattleModeManager] 跳过按钮已消失，退出循环（共尝试 {attempt + 1} 次）")
                return
            
            # 点击跳过按钮
            box = skip_result.box
            x = box.x + box.w // 2
            y = box.y + box.h // 2
            context.tasker.controller.post_click(x, y).wait()
            print(f"[BattleModeManager] 第 {attempt + 1} 次点击跳过按钮")
            
            time.sleep(self._DETECT_INTERVAL)
        
        print(f"[BattleModeManager] 达到最大跳过尝试次数 {self._MAX_SKIP_ATTEMPTS}")

