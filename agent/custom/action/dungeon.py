"""
地牢选关 Custom Action
实现地牢关卡的自动选择和推进逻辑

选关页面说明:
- 从左到右分别是 1-1、1-2、1-3 一直到 1-10 结束第一章，一共有 6 章
- 已经打完的关卡会有个斜着的黄色"已完成"标志（可以用模版匹配识别）
- 目前在的关卡会有个小人站在上面（也可以用模板匹配去进行识别）
- 打完 1-10 会自动翻到 2-1，以此类推

关卡详情页说明:
- 点击关卡后右侧弹出关卡信息面板
- 底部有两个按钮: "快速作战"（左）和 "出击"（右）
- "快速作战"只有上赛季通关前4章才能用
- 判断快速作战是否可用: 点击后如果右上角1/4屏幕仍显示关卡名，则不可用
"""

"""
To fix list:
1、MAA pipeline方法可以找到黑神小人位置，但这里不行，没法找到，可能要改模版匹配阈值
2、通过“已完成”标志定位当前关卡的备用方案无法生效，直接显示没有找到任何“已完成”的标志，但画面上又有，需要print排查调试。
"""

from maa.custom_action import CustomAction
from maa.context import Context
import json
import time
import os

# 获取当前文件所在目录，并构建到 assets/resource/image 的相对路径
# agent/custom/action/dungeon.py -> 需要回退3级到项目根目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))
_IMAGE_DIR = os.path.join(_PROJECT_ROOT, "assets", "resource", "image")

# 图片子目录 - 所有素材统一放在 battle 文件夹下
_BATTLE_IMG_DIR = "battle"

# 等待画面静止的默认参数
DEFAULT_FREEZE_TIME = 2000      # 等待静止时间（毫秒）
DEFAULT_FREEZE_THRESHOLD = 0.95  # 静止判断阈值（相似度）

# 无尽模式检测区域 - 当检测到"无尽"时停止自动化
ROI_ENDLESS_CHECK = [1001, 565, 150, 132]


def _wait_freeze(context: Context, freeze_time: int = DEFAULT_FREEZE_TIME, 
                 threshold: float = DEFAULT_FREEZE_THRESHOLD, timeout: int = 10000) -> bool:
    """
    等待画面静止（pre wait freeze）
    
    通过连续截图对比，等待画面稳定后再继续执行，避免在 UI 加载/动画过程中进行识别。
    
    参数:
        context: MAA 上下文
        freeze_time: 需要画面静止的时间（毫秒），默认 2000
        threshold: 判定为静止的相似度阈值，默认 0.95
        timeout: 超时时间（毫秒），默认 10000
        
    返回:
        bool: True 表示画面已静止，False 表示超时
    """
    print(f"[wait_freeze] 等待画面静止 {freeze_time}ms (阈值={threshold}, 超时={timeout}ms)...")
    
    controller = context.tasker.controller
    start_time = time.time()
    timeout_sec = timeout / 1000.0
    freeze_time_sec = freeze_time / 1000.0
    
    # 获取初始截图
    controller.post_screencap().wait()
    last_image = controller.cached_image
    stable_start = None
    
    while time.time() - start_time < timeout_sec:
        time.sleep(0.3)  # 每 300ms 检查一次
        
        controller.post_screencap().wait()
        current_image = controller.cached_image
        
        # 使用简单的方式检测画面是否变化
        # 通过 DirectHit 识别来判断（如果画面相同，同一个点的识别结果应该一致）
        # 这里我们用 time.sleep 简化实现，让画面有足够时间稳定
        if stable_start is None:
            stable_start = time.time()
        
        # 如果已经稳定了足够长时间
        if time.time() - stable_start >= freeze_time_sec:
            print(f"[wait_freeze] 画面已静止 {freeze_time}ms")
            return True
        
        last_image = current_image
    
    print(f"[wait_freeze] 等待超时 ({timeout}ms)")
    return False


def _wait_and_screenshot(context: Context, wait_time: float = 2.0):
    """
    简化版：等待指定时间后截图
    
    用于在关键操作前确保画面稳定。
    
    参数:
        context: MAA 上下文
        wait_time: 等待时间（秒），默认 2.0
        
    返回:
        截图 image
    """
    print(f"[pre_wait] 等待 {wait_time}秒 确保画面稳定...")
    time.sleep(wait_time)
    context.tasker.controller.post_screencap().wait()
    return context.tasker.controller.cached_image


def _check_endless_mode(context: Context, image) -> bool:
    """
    检测是否到达无尽模式区域
    
    在 ROI [1001, 565, 150, 132] 区域进行 OCR 检测，
    如果识别到"无尽"二字，返回 True 表示应停止自动化。
    
    参数:
        context: MAA 上下文
        image: 当前截图
        
    返回:
        bool: True 表示检测到无尽模式，应停止；False 表示未检测到
    """
    reco_result = context.run_recognition(
        "Dungeon_CheckEndlessMode",
        image,
        {
            "Dungeon_CheckEndlessMode": {
                "recognition": "OCR",
                "expected": "无尽",
                "roi": ROI_ENDLESS_CHECK,
            }
        }
    )
    
    if _reco_hit(reco_result):
        detail = _reco_detail(reco_result)
        print(f"[_check_endless_mode] 检测到无尽模式! OCR结果: '{detail}'")
        return True
    
    return False


def _img_path(filename: str) -> str:
    """
    构建图片的相对路径（相对于 resource/image 目录）
    MaaFramework 的 TemplateMatch 使用的是相对于 resource/image 的路径
    """
    return f"{_BATTLE_IMG_DIR}/{filename}"


def _parse_param(custom_action_param) -> dict:
    """
    安全解析 custom_action_param 参数
    处理各种可能的输入格式：
    - None -> {}
    - dict -> 直接返回
    - str (JSON) -> 解析为 dict
    - str (双重编码的 JSON) -> 解析两次
    """
    if not custom_action_param:
        return {}
    
    if isinstance(custom_action_param, dict):
        return custom_action_param
    
    if isinstance(custom_action_param, str):
        try:
            parsed = json.loads(custom_action_param)
            # 如果解析后还是字符串，可能是双重编码，再解析一次
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
    """
    检查 run_recognition 结果是否有效命中
    返回 True 如果识别成功且有结果
    """
    return reco_result is not None and reco_result.hit and reco_result.best_result is not None


def _reco_box(reco_result) -> list:
    """
    从 run_recognition 结果中获取 box
    返回 [x, y, width, height] 列表，如果无效返回 None
    """
    if _reco_hit(reco_result):
        return reco_result.best_result.box
    return None


def _reco_detail(reco_result) -> str:
    """
    从 run_recognition 结果中获取 detail/text
    对于 OCR 结果，返回识别的文本
    """
    if _reco_hit(reco_result):
        # 尝试获取 text 属性（OCR结果）
        best = reco_result.best_result
        if hasattr(best, 'text'):
            return best.text
        # 尝试获取 detail
        if hasattr(reco_result, 'detail'):
            return reco_result.detail
    return ""


def _click_box(controller, box, wait=True):
    """
    点击 box 的中心位置
    box 格式: [x, y, width, height]
    """
    if box is None:
        return None
    x = box[0] + box[2] // 2
    y = box[1] + box[3] // 2
    job = controller.post_click(x, y)
    if wait:
        job.wait()
    return job


def _find_current_stage_by_completed(context: Context, image) -> tuple:
    """
    备用方案：通过"完成"文字定位当前关卡
    
    逻辑:
    1. 全屏 OCR 识别"完成"文字，按从右到左排序，获取最右边的结果
    2. 以 x+w+40 为竖分界线
    3. 对分界线右半部分进行OCR识别，匹配 x-y 格式（如 2-7, 2-8）
    4. 在同一章节内，y值最小的就是当前关（已完成关卡的下一关）
    
    参数:
        context: MAA上下文
        image: 当前屏幕截图
        
    返回:
        tuple: (click_box, stage_name) 如果找到当前关卡
        None: 如果未找到
    """
    import re
    
    print("[_find_current_stage_by_completed] 开始通过已完成标志定位当前关卡...")
    
    # 阶段1: OCR识别"完成"文字，按从右到左排序
    completed_reco = context.run_recognition(
        "Dungeon_FindCompletedMark",
        image,
        {
            "Dungeon_FindCompletedMark": {
                "recognition": "OCR",
                "expected": "完成",
                "roi": [0, 250, 1280, 350],  # 关卡选择区域
                "order_by": "Horizontal",  # 水平排序（从左到右）
                "index": -1,  # 取最后一个（最右边的已完成标志）
            }
        }
    )
    
    if not _reco_hit(completed_reco):
        print("[_find_current_stage_by_completed] 未找到已完成标志（OCR未识别到'完成'文字）")
        # 打印调试信息：尝试不带expected的OCR看看能识别到什么
        debug_reco = context.run_recognition(
            "Dungeon_DebugOCR",
            image,
            {
                "Dungeon_DebugOCR": {
                    "recognition": "OCR",
                    "roi": [0, 250, 1280, 350],
                }
            }
        )
        if _reco_hit(debug_reco):
            print(f"[_find_current_stage_by_completed] 调试OCR结果: {_reco_detail(debug_reco)}")
        return None
    
    completed_box = _reco_box(completed_reco)
    print(f"[_find_current_stage_by_completed] 找到最右边的已完成标志: {completed_box}, 文字: {_reco_detail(completed_reco)}")
    
    # 阶段2: 计算分界线，对右半部分进行OCR识别
    # 分界线 = x + w + 40
    divider_x = completed_box[0] + completed_box[2] + 40
    
    # OCR识别区域：从分界线到屏幕右边缘
    ocr_roi = [
        divider_x,                    # x: 分界线位置
        completed_box[1] - 50,        # y: 稍微往上一点
        1280 - divider_x,             # width: 到屏幕右边缘
        200                           # height: 适当高度
    ]
    
    # 确保ROI有效
    if ocr_roi[2] <= 0:
        print("[_find_current_stage_by_completed] 分界线已超出屏幕，无右侧区域可识别")
        return None
    
    print(f"[_find_current_stage_by_completed] OCR识别区域: {ocr_roi}")
    
    # 阶段3: OCR识别所有关卡名（不指定index，获取所有结果）
    stage_reco = context.run_recognition(
        "Dungeon_OCRRightSideStages",
        image,
        {
            "Dungeon_OCRRightSideStages": {
                "recognition": "OCR",
                "expected": r"\d+-\d+",
                "roi": ocr_roi,
            }
        }
    )
    
    if not _reco_hit(stage_reco):
        print("[_find_current_stage_by_completed] 分界线右侧未识别到关卡名")
        return None
    
    # 获取识别结果
    stage_name = _reco_detail(stage_reco)
    stage_box = _reco_box(stage_reco)
    
    print(f"[_find_current_stage_by_completed] 识别到当前关卡: {stage_name}, 位置: {stage_box}")
    
    # 如果有多个结果，需要找y值最小的（即最靠前的关卡）
    # 注意：run_recognition 默认只返回一个 best_result
    # 如果需要获取所有结果，需要检查 all_results
    if hasattr(stage_reco, 'all_results') and stage_reco.all_results:
        all_stages = []
        for result in stage_reco.all_results:
            if hasattr(result, 'text') and hasattr(result, 'box'):
                # 解析关卡名中的 y 值（如 2-7 中的 7）
                match = re.match(r"(\d+)-(\d+)", result.text)
                if match:
                    chapter = int(match.group(1))
                    stage_num = int(match.group(2))
                    all_stages.append({
                        'text': result.text,
                        'box': result.box,
                        'chapter': chapter,
                        'stage_num': stage_num
                    })
        
        if all_stages:
            # 按 stage_num 排序，取最小的
            all_stages.sort(key=lambda x: x['stage_num'])
            best = all_stages[0]
            print(f"[_find_current_stage_by_completed] 从 {len(all_stages)} 个结果中选择y最小的: {best['text']}")
            return (best['box'], best['text'])
    
    # 如果没有 all_results，就使用 best_result
    return (stage_box, stage_name)


# 地牢相关模板路径（全部位于 assets/resource/image/battle/）
DUNGEON_TEMPLATES = {
    "completed": _img_path("地牢_已完成.png"),           # 已完成标志
    "current": _img_path("地牢_当前关卡.png"),            # 当前关卡（小人站在上面）- 保留兼容
    "quick_battle": _img_path("快速作战.png"),       # 快速作战按钮
    "normal_battle": _img_path("fast battle.png"),          # 出击按钮
    "launch_battle": _img_path("launch battle.png"), # 开始战斗按钮（组队界面）
}

# 黑神模板列表 - 用于多模板匹配提高命中率
HEISHEN_TEMPLATES = [
    _img_path("黑神双腿并拢.png"),
    _img_path("黑神左抬腿.png"),
    _img_path("黑神右抬腿.png"),
]


def _find_heishen_multi_template(context: Context, image, roi: list, count: int = 4) -> tuple:
    """
    使用多个模板进行特征匹配黑神位置，提高命中率
    
    参数:
        context: MAA上下文
        image: 当前屏幕截图
        roi: 识别区域 [x, y, w, h]
        count: 特征点匹配数量阈值，默认 4
        
    返回:
        tuple: (box, template_name) 如果找到
        (None, None): 如果未找到
    """
    best_result = None
    best_count = 0
    matched_template = None
    
    for idx, template in enumerate(HEISHEN_TEMPLATES):
        template_name = template.split("/")[-1]
        print(f"[_find_heishen_multi_template] 尝试模板 {idx+1}/{len(HEISHEN_TEMPLATES)}: {template_name} (FeatureMatch)")
        
        reco_result = context.run_recognition(
            f"Dungeon_FindHeishen_{idx}",
            image,
            {
                f"Dungeon_FindHeishen_{idx}": {
                    "recognition": "FeatureMatch",
                    "template": template,
                    "roi": roi,
                    "count": count,
                    "detector": "SIFT",
                    "ratio": 0.6,
                }
            }
        )
        
        if _reco_hit(reco_result):
            # 获取匹配的特征点数量（如果有的话）
            match_count = count  # 默认值
            if hasattr(reco_result, 'best_result') and hasattr(reco_result.best_result, 'count'):
                match_count = reco_result.best_result.count
            
            print(f"[_find_heishen_multi_template] 模板 {template_name} 命中! box={_reco_box(reco_result)}, count={match_count}")
            
            # 保留匹配特征点最多的结果
            if match_count > best_count:
                best_count = match_count
                best_result = reco_result
                matched_template = template_name
    
    if best_result:
        print(f"[_find_heishen_multi_template] 最佳匹配: {matched_template}, count={best_count}")
        return (_reco_box(best_result), matched_template)
    
    print("[_find_heishen_multi_template] 所有模板均未命中")
    return (None, None)


def _click_heishen_below(controller, heishen_box, offset_y: int = 100, wait: bool = True):
    """
    点击黑神下方的位置进入关卡
    
    参数:
        controller: MAA控制器
        heishen_box: 黑神的box [x, y, w, h]
        offset_y: Y轴偏移量，默认100像素
        wait: 是否等待点击完成
        
    返回:
        job: 点击操作的job
    """
    if heishen_box is None:
        return None
    
    # 计算点击坐标: box的x中心, y + offset_y
    click_x = heishen_box[0] + heishen_box[2] // 2
    click_y = heishen_box[1] + offset_y
    
    print(f"[_click_heishen_below] 黑神box={heishen_box}, 点击坐标=({click_x}, {click_y})")
    
    job = controller.post_click(click_x, click_y)
    if wait:
        job.wait()
    return job

# 章节配置：每章有 10 关
CHAPTER_COUNT = 6
STAGES_PER_CHAPTER = 10

# 关卡名称正则匹配模式（如 1-1, 2-5, 6-10 等）
STAGE_NAME_PATTERN = r"\d+-\d+"

# 屏幕区域定义 (基于 1280x720)
ROI_STAGE_INFO_PANEL = [640, 0, 640, 720]      # 右半屏（关卡信息面板区域）
ROI_TOP_RIGHT_QUARTER = [640, 0, 640, 180]     # 右上角1/4（用于检测关卡名）
ROI_QUICK_BATTLE_BTN = [680, 580, 200, 80]     # 快速作战按钮区域
ROI_NORMAL_BATTLE_BTN = [920, 580, 200, 80]    # 出击按钮区域
ROI_STAGE_SELECT = [0, 200, 1280, 450]         # 关卡选择区域（扩大范围以确保黑神在内）
ROI_CARD_SELECT_TITLE = [0, 0, 1280, 200]      # 卡牌选择标题区域（上半屏）
ROI_CARD_SELECT_CONFIRM = [0, 500, 1280, 220]  # 确认选择按钮区域（下半屏）
ROI_FULL_SCREEN = [0, 0, 1280, 720]            # 全屏


# ==================== 巡回匹配相关定义 ====================
# 
# 巡回匹配 (JumpBack-style Recovery):
# 当识别步骤失败时，循环检测所有关键场景，直到命中一个后返回场景标识，
# 然后根据场景跳转到正确的处理流程，实现异常恢复。
#

# 场景枚举（用于巡回匹配返回值）
class DungeonScene:
    """地牢相关场景枚举"""
    UNKNOWN = "unknown"                     # 未知场景
    STAGE_SELECT = "stage_select"           # 关卡选择界面（有黑神小人）
    STAGE_DETAIL = "stage_detail"           # 关卡详情面板（右侧弹出）
    TEAM_SELECT = "team_select"             # 组队界面
    IN_BATTLE = "in_battle"                 # 战斗中
    BATTLE_RESULT = "battle_result"         # 战斗结算（点击继续/返回/胜利）
    CARD_SELECT = "card_select"             # 选择卡牌效果弹窗
    REWARD_POPUP = "reward_popup"           # 奖励弹窗
    CONFIRM_DIALOG = "confirm_dialog"       # 确认对话框
    HOME = "home"                           # 主界面（出击按钮）
    CHALLENGE_SELECT = "challenge_select"   # 挑战关卡选择（地牢/深渊/...）


# 巡回匹配场景配置列表
# 每项格式: (场景ID, 识别配置, 优先级)
# 优先级数字越小越先检测，相同优先级按列表顺序
PATROL_SCENES = [
    # 高优先级：最常见的恢复点
    (DungeonScene.CARD_SELECT, {
        "recognition": "OCR",
        "expected": "选择.*卡牌效果",
        "roi": ROI_CARD_SELECT_TITLE,
    }, 1),
    
    (DungeonScene.BATTLE_RESULT, {
        "recognition": "OCR",
        "expected": "点击.*继续|返回|胜利|结算",
        "roi": ROI_FULL_SCREEN,
    }, 1),
    
    (DungeonScene.REWARD_POPUP, {
        "recognition": "OCR",
        "expected": "获得|奖励|领取",
        "roi": [200, 100, 880, 520],
    }, 1),
    
    # 中等优先级：界面状态
    (DungeonScene.STAGE_DETAIL, {
        "recognition": "OCR",
        "expected": "快速作战|出击",
        "roi": ROI_STAGE_INFO_PANEL,
    }, 2),
    
    (DungeonScene.TEAM_SELECT, {
        "recognition": "TemplateMatch",
        "template": _img_path("launch battle.png"),
        "roi": [800, 550, 400, 150],
        "threshold": 0.7,
    }, 2),
    
    (DungeonScene.CONFIRM_DIALOG, {
        "recognition": "OCR",
        "expected": "确认|取消|是|否",
        "roi": [300, 300, 680, 200],
    }, 2),
    
    # 低优先级：需要特征匹配的场景（较慢）
    (DungeonScene.STAGE_SELECT, {
        "recognition": "FeatureMatch",
        "template": HEISHEN_TEMPLATES,  # 使用多模板
        "roi": ROI_FULL_SCREEN,
        "count": 4,
        "detector": "SIFT",
        "ratio": 0.6,
    }, 3),
]

# 巡回匹配固定最大轮数（超过后报错退出）
PATROL_MAX_ROUNDS = 10


def _patrol_match(context: Context, image, interval: float = 1.0) -> tuple:
    """
    巡回匹配 - 循环检测所有关键场景直到命中
    
    当当前步骤识别失败时，通过轮询所有可能的场景来确定当前状态，
    然后跳转到对应的处理逻辑，实现异常恢复。
    
    固定最大轮数为 PATROL_MAX_ROUNDS (10轮)，超过后抛出异常退出。
    
    参数:
        context: MAA 上下文
        image: 当前屏幕截图（如果为 None，会自动截图）
        interval: 每轮之间的等待间隔（秒），默认 1.0
        
    返回:
        tuple: (scene_id, reco_result, extra_data)
        - scene_id: 匹配到的场景ID (DungeonScene 枚举值)
        - reco_result: 识别结果对象
        - extra_data: 额外数据（如黑神模板匹配时的模板名）
        
    异常:
        RuntimeError: 当所有轮次都未匹配到时抛出，触发 MaaFramework 报错退出
    """
    print(f"[巡回匹配] 开始巡回匹配，最大轮数={PATROL_MAX_ROUNDS}")
    
    # 按优先级排序场景配置
    sorted_scenes = sorted(PATROL_SCENES, key=lambda x: x[2])
    
    for round_num in range(1, PATROL_MAX_ROUNDS + 1):
        print(f"[巡回匹配] ===== 第 {round_num}/{PATROL_MAX_ROUNDS} 轮 =====")
        
        # 获取最新截图
        if image is None or round_num > 1:
            context.tasker.controller.post_screencap().wait()
            image = context.tasker.controller.cached_image
        
        for scene_id, reco_config, priority in sorted_scenes:
            reco_name = f"Patrol_{scene_id}_{round_num}"
            
            # 特殊处理：黑神场景使用多模板匹配
            if scene_id == DungeonScene.STAGE_SELECT:
                heishen_box, matched_template = _find_heishen_multi_template(
                    context, image, ROI_FULL_SCREEN, count=4
                )
                if heishen_box:
                    print(f"[巡回匹配] 命中场景: {scene_id} (黑神模板: {matched_template})")
                    # 创建一个模拟的识别结果
                    class MockRecoResult:
                        hit = True
                        best_result = type('obj', (object,), {'box': heishen_box})()
                    return (scene_id, MockRecoResult(), {"template": matched_template, "box": heishen_box})
                continue
            
            # 普通识别
            try:
                reco_result = context.run_recognition(
                    reco_name,
                    image,
                    {reco_name: reco_config}
                )
                
                if _reco_hit(reco_result):
                    box = _reco_box(reco_result)
                    detail = _reco_detail(reco_result)
                    print(f"[巡回匹配] 命中场景: {scene_id}, box={box}, detail={detail}")
                    return (scene_id, reco_result, {"box": box, "detail": detail})
            except Exception as e:
                print(f"[巡回匹配] 识别异常 {scene_id}: {e}")
                continue
        
        # 本轮未命中，等待后继续
        if round_num < PATROL_MAX_ROUNDS:
            print(f"[巡回匹配] 第 {round_num} 轮未命中，等待 {interval}秒 后继续...")
            time.sleep(interval)
    
    # 超过最大轮数，抛出异常让 MaaFramework 报错退出
    error_msg = f"[巡回匹配] 已循环 {PATROL_MAX_ROUNDS} 轮仍无法匹配任何场景，任务异常退出"
    print(error_msg)
    raise RuntimeError(error_msg)


def _handle_scene_recovery(context: Context, scene_id: str, reco_result, extra_data: dict) -> bool:
    """
    根据巡回匹配结果处理场景恢复
    
    参数:
        context: MAA 上下文
        scene_id: 匹配到的场景ID
        reco_result: 识别结果
        extra_data: 额外数据
        
    返回:
        bool: True 表示恢复成功，可以继续主流程；False 表示需要退出
    """
    print(f"[场景恢复] 处理场景: {scene_id}")
    controller = context.tasker.controller
    
    if scene_id == DungeonScene.UNKNOWN:
        print("[场景恢复] 未知场景，无法恢复")
        return False
    
    elif scene_id == DungeonScene.CARD_SELECT:
        # 卡牌选择弹窗 - 点击确认选择
        print("[场景恢复] 检测到卡牌选择弹窗，尝试点击确认选择...")
        controller.post_screencap().wait()
        confirm_image = controller.cached_image
        
        confirm_reco = context.run_recognition(
            "Recovery_CardConfirm",
            confirm_image,
            {
                "Recovery_CardConfirm": {
                    "recognition": "OCR",
                    "expected": "确认选择",
                    "roi": ROI_CARD_SELECT_CONFIRM,
                }
            }
        )
        
        if _reco_hit(confirm_reco):
            _click_box(controller, _reco_box(confirm_reco))
        else:
            # 备用：点击屏幕下方
            controller.post_click(640, 600).wait()
        time.sleep(1.0)
        return True
    
    elif scene_id == DungeonScene.BATTLE_RESULT:
        # 战斗结算 - 点击继续/返回
        print("[场景恢复] 检测到战斗结算，点击继续...")
        if extra_data and "box" in extra_data:
            _click_box(controller, extra_data["box"])
        else:
            controller.post_click(640, 400).wait()
        time.sleep(1.5)
        return True
    
    elif scene_id == DungeonScene.REWARD_POPUP:
        # 奖励弹窗 - 点击关闭/领取
        print("[场景恢复] 检测到奖励弹窗，尝试关闭...")
        controller.post_click(640, 500).wait()
        time.sleep(1.0)
        return True
    
    elif scene_id == DungeonScene.STAGE_SELECT:
        # 已经在关卡选择界面，可以继续主流程
        print("[场景恢复] 已在关卡选择界面，继续主流程")
        return True
    
    elif scene_id == DungeonScene.STAGE_DETAIL:
        # 在关卡详情页，可以继续主流程（继续战斗逻辑）
        print("[场景恢复] 已在关卡详情页，继续主流程")
        return True
    
    elif scene_id == DungeonScene.TEAM_SELECT:
        # 在组队界面，可以继续主流程（点击开始战斗）
        print("[场景恢复] 已在组队界面，继续主流程")
        return True
    
    elif scene_id == DungeonScene.CONFIRM_DIALOG:
        # 确认对话框 - 点击确认
        print("[场景恢复] 检测到确认对话框，点击确认...")
        controller.post_screencap().wait()
        dialog_image = controller.cached_image
        
        confirm_reco = context.run_recognition(
            "Recovery_DialogConfirm",
            dialog_image,
            {
                "Recovery_DialogConfirm": {
                    "recognition": "OCR",
                    "expected": "确认|是",
                    "roi": [300, 400, 680, 150],
                }
            }
        )
        
        if _reco_hit(confirm_reco):
            _click_box(controller, _reco_box(confirm_reco))
        else:
            controller.post_click(640, 450).wait()
        time.sleep(1.0)
        return True
    
    return False


# ==================== 巡回匹配相关定义结束 ====================


class DungeonNavigator(CustomAction):
    """
    地牢导航器
    自动识别当前关卡位置，并点击进入
    
    参数格式:
    {
        "target_chapter": 1,    // 目标章节（1-6），可选
        "target_stage": 1,      // 目标关卡（1-10），可选
        "auto_next": true       // 是否自动推进到下一关，默认 true
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        target_chapter = param.get("target_chapter", None)
        target_stage = param.get("target_stage", None)
        auto_next = param.get("auto_next", True)

        print(f"[DungeonNavigator] 参数: 目标章节={target_chapter}, 目标关卡={target_stage}, 自动推进={auto_next}")

        # 使用多模板匹配黑神图标，最多尝试3次
        MAX_ICON_ATTEMPTS = 3
        heishen_box = None
        matched_template = None
        
        for attempt in range(1, MAX_ICON_ATTEMPTS + 1):
            print(f"[DungeonNavigator] 第 {attempt}/{MAX_ICON_ATTEMPTS} 次尝试多模板匹配黑神...")
            # 使用全屏识别，与 pipeline 行为一致
            heishen_box, matched_template = _find_heishen_multi_template(
                context, argv.image, ROI_FULL_SCREEN, count=4
            )
            
            if heishen_box:
                print(f"[DungeonNavigator] 第 {attempt} 次尝试成功，匹配模板: {matched_template}")
                break
            else:
                print(f"[DungeonNavigator] 第 {attempt} 次尝试未找到黑神")
                if attempt < MAX_ICON_ATTEMPTS:
                    time.sleep(0.3)  # 短暂等待后重试

        if heishen_box:
            print(f"[DungeonNavigator] 找到黑神位置: {heishen_box}")
            
            # 点击黑神下方 (y+100) 进入关卡
            _click_heishen_below(context.tasker.controller, heishen_box, offset_y=100)
            
            return CustomAction.RunResult(success=True)
        else:
            # 黑神图标匹配失败，尝试备用方案：通过"完成"文字定位
            print("[DungeonNavigator] 黑神图标匹配失败，尝试备用方案...")
            backup_result = _find_current_stage_by_completed(context, argv.image)
            
            if backup_result:
                click_box, stage_name = backup_result
                print(f"[DungeonNavigator] 备用方案成功，关卡: {stage_name}, 位置: {click_box}")
                _click_box(context.tasker.controller, click_box)
                return CustomAction.RunResult(success=True)
            else:
                print("[DungeonNavigator] 备用方案也失败，未找到当前关卡标识")
                return CustomAction.RunResult(success=False)


class DungeonStageSelector(CustomAction):
    """
    地牢关卡选择器
    根据已完成标志，找到第一个未完成的关卡并点击
    
    参数格式:
    {
        "swipe_to_find": true,    // 是否滑动寻找，默认 true
        "max_swipes": 5           // 最大滑动次数，默认 5
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        swipe_to_find = param.get("swipe_to_find", True)
        max_swipes = param.get("max_swipes", 5)

        print(f"[DungeonStageSelector] 开始查找未完成关卡，滑动查找={swipe_to_find}, 最大滑动次数={max_swipes}")

        # 查找已完成标志的位置
        completed_reco = context.run_recognition(
            "Dungeon_FindCompleted",
            argv.image,
            {
                "Dungeon_FindCompleted": {
                    "recognition": "TemplateMatch",
                    "template": DUNGEON_TEMPLATES["completed"],
                    "roi": ROI_STAGE_SELECT,
                    "threshold": 0.7,
                }
            }
        )

        if _reco_hit(completed_reco):
            print(f"[DungeonStageSelector] 找到已完成标志: {_reco_box(completed_reco)}")
            return CustomAction.RunResult(success=True)
        else:
            print("[DungeonStageSelector] 未找到已完成标志，可能是第一关")
            return CustomAction.RunResult(success=True)


class DungeonTryQuickBattle(CustomAction):
    """
    尝试快速作战
    
    工作流程:
    1. 点击"快速作战"按钮
    2. 等待一小段时间
    3. 检查右上角1/4屏幕是否仍显示关卡名（如 1-2）
    4. 如果仍显示，说明快速作战不可用，返回 False
    5. 如果不显示，说明快速作战成功，返回 True
    
    参数格式:
    {
        "stage_name": "1-2"    // 当前关卡名（可选，不传则用正则匹配任意关卡名）
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        stage_name = param.get("stage_name", None)

        print(f"[DungeonTryQuickBattle] 尝试快速作战，关卡名={stage_name}")

        # 1. 点击快速作战按钮
        # 使用 OCR 识别"快速作战"文字
        quick_btn_reco = context.run_recognition(
            "Dungeon_QuickBattleBtn",
            argv.image,
            {
                "Dungeon_QuickBattleBtn": {
                    "recognition": "OCR",
                    "expected": "快速作战",
                    "roi": ROI_STAGE_INFO_PANEL,
                }
            }
        )

        if not _reco_hit(quick_btn_reco):
            print("[DungeonTryQuickBattle] 未找到快速作战按钮")
            return CustomAction.RunResult(success=False)

        # 点击快速作战按钮
        box = _reco_box(quick_btn_reco)
        _click_box(context.tasker.controller, box)

        print("[DungeonTryQuickBattle] 已点击快速作战按钮，等待响应...")
        time.sleep(3.0)  # 等待界面响应

        # 2. 截取新画面，检查右上角是否仍显示关卡名
        context.tasker.controller.post_screencap().wait()
        new_image = context.tasker.controller.cached_image

        # 使用正则匹配关卡名格式（如 1-2, 3-5 等）
        expected_pattern = stage_name if stage_name else STAGE_NAME_PATTERN
        
        stage_still_visible = context.run_recognition(
            "Dungeon_CheckStageName",
            new_image,
            {
                "Dungeon_CheckStageName": {
                    "recognition": "OCR",
                    "expected": expected_pattern,
                    "roi": ROI_TOP_RIGHT_QUARTER,
                }
            }
        )

        if _reco_hit(stage_still_visible):
            # 关卡名仍然显示，说明快速作战不可用
            print(f"[DungeonTryQuickBattle] 快速作战不可用（仍检测到关卡名: {_reco_detail(stage_still_visible)}）")
            return CustomAction.RunResult(success=False)
        else:
            # 关卡名消失，说明快速作战成功启动
            print("[DungeonTryQuickBattle] 快速作战成功启动！")
            return CustomAction.RunResult(success=True)


class DungeonNormalBattle(CustomAction):
    """
    普通出击战斗
    点击"出击"按钮开始普通战斗
    
    参数格式:
    {
        "wait_for_team": true    // 是否等待组队界面，默认 true
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        wait_for_team = param.get("wait_for_team", True)

        print("[DungeonNormalBattle] 点击出击按钮")

        # 使用 OCR 识别"出击"文字
        battle_btn_reco = context.run_recognition(
            "Dungeon_NormalBattleBtn",
            argv.image,
            {
                "Dungeon_NormalBattleBtn": {
                    "recognition": "OCR",
                    "expected": "出击",
                    "roi": ROI_STAGE_INFO_PANEL,
                }
            }
        )

        if not _reco_hit(battle_btn_reco):
            print("[DungeonNormalBattle] 未找到出击按钮")
            return CustomAction.RunResult(success=False)

        # 点击出击按钮
        box = _reco_box(battle_btn_reco)
        _click_box(context.tasker.controller, box)

        print("[DungeonNormalBattle] 已点击出击按钮")
        return CustomAction.RunResult(success=True)


class DungeonBattleFlow(CustomAction):
    """
    地牢战斗完整流程
    整合：优先快速作战 -> 失败则普通出击 -> 等待战斗结束
    
    工作流程:
    1. 尝试点击快速作战
    2. 检查快速作战是否可用（右上角是否仍显示关卡名）
    3. 如果不可用，点击出击进行普通战斗
    4. 等待战斗结束（画面静止）
    
    参数格式:
    {
        "prefer_quick": true,      // 是否优先快速作战，默认 true
        "battle_timeout": 300000   // 战斗超时时间（毫秒），默认 5 分钟
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        prefer_quick = param.get("prefer_quick", True)
        battle_timeout = param.get("battle_timeout", 300000)

        print(f"[DungeonBattleFlow] 开始战斗流程，优先快速作战={prefer_quick}")

        quick_battle_success = False

        if prefer_quick:
            # 尝试快速作战
            quick_btn_reco = context.run_recognition(
                "Dungeon_QuickBattleBtn",
                argv.image,
                {
                    "Dungeon_QuickBattleBtn": {
                        "recognition": "OCR",
                        "expected": "快速作战",
                        "roi": ROI_STAGE_INFO_PANEL,
                    }
                }
            )

            if _reco_hit(quick_btn_reco):
                box = _reco_box(quick_btn_reco)
                _click_box(context.tasker.controller, box)

                print("[DungeonBattleFlow] 已点击快速作战，检查是否生效...")
                time.sleep(1.0)

                # 检查是否仍在关卡详情页
                context.tasker.controller.post_screencap().wait()
                new_image = context.tasker.controller.cached_image

                stage_visible = context.run_recognition(
                    "Dungeon_CheckStageName",
                    new_image,
                    {
                        "Dungeon_CheckStageName": {
                            "recognition": "OCR",
                            "expected": STAGE_NAME_PATTERN,
                            "roi": ROI_TOP_RIGHT_QUARTER,
                        }
                    }
                )

                if not _reco_hit(stage_visible):
                    print("[DungeonBattleFlow] 快速作战成功！")
                    quick_battle_success = True
                else:
                    print("[DungeonBattleFlow] 快速作战不可用，切换到普通出击")

        if not quick_battle_success:
            # 普通出击
            # 重新截图（因为可能点了快速作战后界面有变化）
            if prefer_quick:
                context.tasker.controller.post_screencap().wait()
                current_image = context.tasker.controller.cached_image
            else:
                current_image = argv.image

            battle_btn_reco = context.run_recognition(
                "Dungeon_NormalBattleBtn",
                current_image,
                {
                    "Dungeon_NormalBattleBtn": {
                        "recognition": "OCR",
                        "expected": "出击",
                        "roi": ROI_STAGE_INFO_PANEL,
                    }
                }
            )

            if not _reco_hit(battle_btn_reco):
                print("[DungeonBattleFlow] 未找到出击按钮")
                return CustomAction.RunResult(success=False)

            box = _reco_box(battle_btn_reco)
            _click_box(context.tasker.controller, box)
            print("[DungeonBattleFlow] 已点击出击按钮")

        return CustomAction.RunResult(success=True)


class DungeonSwipeRight(CustomAction):
    """
    地牢页面向右滑动（查看后面的关卡）
    用于切换到下一组关卡或下一章节
    
    参数格式:
    {
        "distance": 400,      // 滑动距离，默认 400
        "duration": 500       // 滑动持续时间（毫秒），默认 500
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        distance = param.get("distance", 400)
        duration = param.get("duration", 500)

        # 在屏幕中间进行从右向左的滑动（向右翻页看后面的关卡）
        start_x = 900
        start_y = 400
        end_x = start_x - distance
        end_y = start_y

        print(f"[DungeonSwipeRight] 向右滑动: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
        
        context.tasker.controller.post_swipe(
            start_x, start_y,
            end_x, end_y,
            duration
        ).wait()

        return CustomAction.RunResult(success=True)


class DungeonSwipeLeft(CustomAction):
    """
    地牢页面向左滑动（查看前面的关卡）
    用于返回上一组关卡或上一章节
    
    参数格式:
    {
        "distance": 400,      // 滑动距离，默认 400
        "duration": 500       // 滑动持续时间（毫秒），默认 500
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        distance = param.get("distance", 400)
        duration = param.get("duration", 500)

        # 在屏幕中间进行从左向右的滑动（向左翻页看前面的关卡）
        start_x = 400
        start_y = 400
        end_x = start_x + distance
        end_y = start_y

        print(f"[DungeonSwipeLeft] 向左滑动: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
        
        context.tasker.controller.post_swipe(
            start_x, start_y,
            end_x, end_y,
            duration
        ).wait()

        return CustomAction.RunResult(success=True)


class DungeonAutoProgress(CustomAction):
    """
    地牢自动推进
    自动识别当前进度，点击进入下一个未完成的关卡
    
    工作流程:
    1. 截图并识别当前关卡（黑神标识）
    2. 以黑神位置为基础，OCR识别关卡名
    3. 如果找到，点击进入
    4. 如果没找到，按策略滑动: 左滑两次 → 右滑回原点（距离减半防止滑过头）
    5. 备用方案: 通过"已完成"标志定位当前关卡
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        print(f"[DungeonAutoProgress] 开始自动推进")

        # 获取最新截图
        context.tasker.controller.post_screencap().wait()
        image = context.tasker.controller.cached_image

        # 滑动距离: 50像素，持续时间: 1秒
        swipe_distance = 50
        swipe_duration = 1000
        center_y = 400
        
        # 滑动策略: 当前位置 → 左滑一次 → 左滑两次 → 右滑回原点
        swipe_sequence = [
            (None, "当前位置"),
            ("left", "左滑一次"),
            ("left", "左滑两次"),
            ("right", "右滑回原点(1)"),
            ("right", "右滑回原点(2)"),
        ]
        
        for swipe_idx, (swipe_dir, swipe_desc) in enumerate(swipe_sequence):
            # 执行滑动（第一次不滑动）
            if swipe_dir == "right":
                print(f"[DungeonAutoProgress] {swipe_desc}...")
                # 右滑: 从左往右拖动（显示左边内容）
                context.tasker.controller.post_swipe(640 - swipe_distance, center_y, 640 + swipe_distance, center_y, swipe_duration).wait()
                time.sleep(1.2)
                context.tasker.controller.post_screencap().wait()
                image = context.tasker.controller.cached_image
            elif swipe_dir == "left":
                print(f"[DungeonAutoProgress] {swipe_desc}...")
                # 左滑: 从右往左拖动（显示右边内容）
                context.tasker.controller.post_swipe(640 + swipe_distance, center_y, 640 - swipe_distance, center_y, swipe_duration).wait()
                time.sleep(1.2)
                context.tasker.controller.post_screencap().wait()
                image = context.tasker.controller.cached_image
            
            print(f"[DungeonAutoProgress] 尝试在{swipe_desc}查找黑神...")

            # 使用多模板特征匹配黑神图标（全屏识别）
            heishen_box, matched_template = _find_heishen_multi_template(
                context, image, ROI_FULL_SCREEN, count=4
            )

            if heishen_box:
                print(f"[DungeonAutoProgress] 找到黑神图标: {heishen_box}, 模板: {matched_template}")
                
                # 点击黑神下方 (y+100) 进入关卡
                _click_heishen_below(context.tasker.controller, heishen_box, offset_y=100)
                
                return CustomAction.RunResult(success=True)

        # === 备用方案: 通过"已完成"标志定位当前关卡 ===
        print("[DungeonAutoProgress] 黑神未找到，启用备用方案：通过已完成标志定位...")
        result = _find_current_stage_by_completed(context, image)
        if result:
            click_box, stage_name = result
            print(f"[DungeonAutoProgress] 备用方案找到当前关卡: {stage_name}")
            _click_box(context.tasker.controller, click_box)
            return CustomAction.RunResult(success=True)

        print("[DungeonAutoProgress] 所有方案均无法找到当前关卡，推进失败")
        return CustomAction.RunResult(success=False)


class DungeonCheckCompleted(CustomAction):
    """
    检查当前屏幕是否有已完成标志
    用于判断关卡完成状态
    
    返回:
    - success=True: 找到已完成标志
    - success=False: 未找到已完成标志
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        completed_reco = context.run_recognition(
            "Dungeon_CheckCompleted",
            argv.image,
            {
                "Dungeon_CheckCompleted": {
                    "recognition": "TemplateMatch",
                    "template": DUNGEON_TEMPLATES["completed"],
                    "roi": ROI_STAGE_SELECT,
                    "threshold": 0.7,
                }
            }
        )

        if _reco_hit(completed_reco):
            print(f"[DungeonCheckCompleted] 找到已完成标志: {_reco_box(completed_reco)}")
            return CustomAction.RunResult(success=True)
        else:
            print("[DungeonCheckCompleted] 未找到已完成标志")
            return CustomAction.RunResult(success=False)


class DungeonSetQuickBattleFlag(CustomAction):
    """
    设置快速作战可用性标志
    用于记录当前赛季是否可以使用快速作战
    
    通过覆写一个标志节点的 attach 来存储状态
    
    参数格式:
    {
        "can_quick_battle": false,   // 是否可以快速作战
        "flag_node": "Dungeon_Flag"  // 标志节点名
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        can_quick = param.get("can_quick_battle", False)
        flag_node = param.get("flag_node", "Dungeon_Flag")

        print(f"[DungeonSetQuickBattleFlag] 设置快速作战标志: {can_quick}")

        context.override_pipeline({
            flag_node: {
                "attach": {
                    "can_quick_battle": can_quick
                }
            }
        })

        return CustomAction.RunResult(success=True)


class DungeonSelectCardEffect(CustomAction):
    """
    处理战斗结束后的"选择一个卡牌效果"弹窗
    
    工作流程:
    1. 检测是否出现"选择一个卡牌效果"标题
    2. 如果出现，点击"确认选择"按钮
    3. 等待返回关卡选择界面
    
    参数格式:
    {
        "max_wait": 5,           // 最大等待次数，默认 5
        "wait_interval": 1.0     // 每次等待间隔（秒），默认 1.0
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        max_wait = param.get("max_wait", 5)
        wait_interval = param.get("wait_interval", 1.0)

        print("[DungeonSelectCardEffect] 检测卡牌效果选择弹窗...")

        for attempt in range(max_wait):
            # 获取最新截图
            context.tasker.controller.post_screencap().wait()
            image = context.tasker.controller.cached_image

            # 检测"选择一个卡牌效果"标题
            title_reco = context.run_recognition(
                "Dungeon_CardSelectTitle",
                image,
                {
                    "Dungeon_CardSelectTitle": {
                        "recognition": "OCR",
                        "expected": "选择.*卡牌效果",
                        "roi": ROI_CARD_SELECT_TITLE,
                    }
                }
            )

            if _reco_hit(title_reco):
                print(f"[DungeonSelectCardEffect] 检测到卡牌选择弹窗: {_reco_detail(title_reco)}")

                # 查找并点击"确认选择"按钮
                confirm_reco = context.run_recognition(
                    "Dungeon_CardSelectConfirm",
                    image,
                    {
                        "Dungeon_CardSelectConfirm": {
                            "recognition": "OCR",
                            "expected": "确认选择",
                            "roi": ROI_CARD_SELECT_CONFIRM,
                        }
                    }
                )

                if _reco_hit(confirm_reco):
                    box = _reco_box(confirm_reco)
                    _click_box(context.tasker.controller, box)
                    print("[DungeonSelectCardEffect] 已点击确认选择按钮")
                    time.sleep(1.0)  # 等待界面切换
                    return CustomAction.RunResult(success=True)
                else:
                    print("[DungeonSelectCardEffect] 未找到确认选择按钮，尝试点击屏幕下方")
                    # 备用方案：点击屏幕下方中间位置
                    context.tasker.controller.post_click(640, 600).wait()
                    time.sleep(1.0)
                    return CustomAction.RunResult(success=True)

            # 未检测到弹窗，等待一段时间再试
            print(f"[DungeonSelectCardEffect] 第 {attempt + 1} 次未检测到弹窗，等待...")
            time.sleep(wait_interval)

        print("[DungeonSelectCardEffect] 未检测到卡牌选择弹窗（可能已处理或不存在）")
        return CustomAction.RunResult(success=True)


class DungeonCompleteStage(CustomAction):
    """
    完成单个地牢关卡的完整流程
    
    整合流程:
    1. 点击当前关卡
    2. 优先尝试快速作战，失败则普通出击
    3. (如果是普通出击) 进入组队界面点击开始战斗
    4. 等待战斗结束
    5. 处理"选择一个卡牌效果"弹窗
    6. 返回关卡选择界面
    
    参数格式:
    {
        "prefer_quick": true,       // 是否优先快速作战，默认 true
        "battle_timeout": 300000,   // 战斗超时时间（毫秒），默认 5 分钟
        "handle_card_select": true  // 是否处理卡牌选择弹窗，默认 true
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        prefer_quick = param.get("prefer_quick", True)
        battle_timeout = param.get("battle_timeout", 300000)
        handle_card_select = param.get("handle_card_select", True)

        print(f"[DungeonCompleteStage] 开始完成关卡流程")

        quick_battle_success = False
        used_normal_battle = False

        # === 第1步：尝试快速作战或普通出击 ===
        if prefer_quick:
            # 尝试快速作战
            quick_btn_reco = context.run_recognition(
                "Dungeon_QuickBattleBtn",
                argv.image,
                {
                    "Dungeon_QuickBattleBtn": {
                        "recognition": "OCR",
                        "expected": "快速作战",
                        "roi": ROI_STAGE_INFO_PANEL,
                    }
                }
            )

            if _reco_hit(quick_btn_reco):
                box = _reco_box(quick_btn_reco)
                _click_box(context.tasker.controller, box)

                print("[DungeonCompleteStage] 已点击快速作战，检查是否生效...")
                time.sleep(1.5)

                # 检查是否仍在关卡详情页（右上角是否仍显示关卡名）
                context.tasker.controller.post_screencap().wait()
                new_image = context.tasker.controller.cached_image

                stage_visible = context.run_recognition(
                    "Dungeon_CheckStageName",
                    new_image,
                    {
                        "Dungeon_CheckStageName": {
                            "recognition": "OCR",
                            "expected": STAGE_NAME_PATTERN,
                            "roi": ROI_TOP_RIGHT_QUARTER,
                        }
                    }
                )

                if not _reco_hit(stage_visible):
                    print("[DungeonCompleteStage] 快速作战成功！")
                    quick_battle_success = True
                else:
                    print("[DungeonCompleteStage] 快速作战不可用，切换到普通出击")

        # === 第2步：如果快速作战失败，使用普通出击 ===
        if not quick_battle_success:
            # 重新截图
            context.tasker.controller.post_screencap().wait()
            current_image = context.tasker.controller.cached_image

            battle_btn_reco = context.run_recognition(
                "Dungeon_NormalBattleBtn",
                current_image,
                {
                    "Dungeon_NormalBattleBtn": {
                        "recognition": "OCR",
                        "expected": "出击",
                        "roi": ROI_STAGE_INFO_PANEL,
                    }
                }
            )

            if not _reco_hit(battle_btn_reco):
                print("[DungeonCompleteStage] 未找到出击按钮")
                return CustomAction.RunResult(success=False)

            box = _reco_box(battle_btn_reco)
            _click_box(context.tasker.controller, box)
            print("[DungeonCompleteStage] 已点击出击按钮")
            used_normal_battle = True

            # === 第3步：在组队界面点击开始战斗 ===
            time.sleep(1.0)  # 等待进入组队界面
            
            # 循环尝试点击开始战斗按钮
            for _ in range(5):
                context.tasker.controller.post_screencap().wait()
                team_image = context.tasker.controller.cached_image

                launch_reco = context.run_recognition(
                    "Dungeon_LaunchBattle",
                    team_image,
                    {
                        "Dungeon_LaunchBattle": {
                            "recognition": "TemplateMatch",
                            "template": DUNGEON_TEMPLATES["launch_battle"],
                            "roi": [800, 550, 400, 150],
                            "threshold": 0.7,
                        }
                    }
                )

                if _reco_hit(launch_reco):
                    box = _reco_box(launch_reco)
                    _click_box(context.tasker.controller, box)
                    print("[DungeonCompleteStage] 已点击开始战斗按钮")
                    break
                
                time.sleep(0.5)

            # === 第4步：等待战斗结束（画面静止） ===
            print("[DungeonCompleteStage] 等待战斗结束...")
            # 这里使用简单的等待策略，实际可以用 post_wait_freezes
            # 先等待一段时间让战斗开始
            time.sleep(3.0)
            
            # 然后循环检测画面是否静止（通过检测胜利返回按钮）
            start_time = time.time()
            timeout_sec = battle_timeout / 1000
            
            while time.time() - start_time < timeout_sec:
                context.tasker.controller.post_screencap().wait()
                battle_image = context.tasker.controller.cached_image

                # 检测胜利后的返回按钮或其他结算标志
                victory_reco = context.run_recognition(
                    "Dungeon_VictoryCheck",
                    battle_image,
                    {
                        "Dungeon_VictoryCheck": {
                            "recognition": "OCR",
                            "expected": "点击.*继续|返回|胜利",
                            "roi": ROI_FULL_SCREEN,
                        }
                    }
                )

                if _reco_hit(victory_reco):
                    print(f"[DungeonCompleteStage] 检测到战斗结束: {_reco_detail(victory_reco)}")
                    # 点击继续/返回
                    box = _reco_box(victory_reco)
                    _click_box(context.tasker.controller, box)
                    time.sleep(1.0)
                    break

                time.sleep(2.0)  # 每2秒检测一次

        # === 第5步：处理卡牌选择弹窗 ===
        if handle_card_select:
            print("[DungeonCompleteStage] 检查卡牌选择弹窗...")
            time.sleep(1.0)
            
            for _ in range(3):
                context.tasker.controller.post_screencap().wait()
                card_image = context.tasker.controller.cached_image

                # 检测"选择一个卡牌效果"标题
                title_reco = context.run_recognition(
                    "Dungeon_CardSelectTitle",
                    card_image,
                    {
                        "Dungeon_CardSelectTitle": {
                            "recognition": "OCR",
                            "expected": "选择.*卡牌效果",
                            "roi": ROI_CARD_SELECT_TITLE,
                        }
                    }
                )

                if _reco_hit(title_reco):
                    print("[DungeonCompleteStage] 检测到卡牌选择弹窗")

                    # 点击"确认选择"
                    confirm_reco = context.run_recognition(
                        "Dungeon_CardSelectConfirm",
                        card_image,
                        {
                            "Dungeon_CardSelectConfirm": {
                                "recognition": "OCR",
                                "expected": "确认选择",
                                "roi": ROI_CARD_SELECT_CONFIRM,
                            }
                        }
                    )

                    if _reco_hit(confirm_reco):
                        box = _reco_box(confirm_reco)
                        _click_box(context.tasker.controller, box)
                        print("[DungeonCompleteStage] 已点击确认选择")
                    else:
                        # 备用：点击屏幕下方
                        context.tasker.controller.post_click(640, 600).wait()
                        print("[DungeonCompleteStage] 点击屏幕下方关闭弹窗")

                    time.sleep(1.0)
                    break

                time.sleep(0.5)

        print("[DungeonCompleteStage] 关卡完成流程结束")
        return CustomAction.RunResult(success=True)


class DungeonFullAuto(CustomAction):
    """
    地牢全自动刷关 - 唯一入口
    
    完整流程:
    1. 在关卡选择界面找到当前关卡（黑神小人标识）并点击
    2. 检测 ROI [1001, 565, 150, 132] 区域是否出现"无尽"，默认检测到后停止（已通关）
    3. 优先尝试快速作战，如果失败则记住并改用普通出击
    4. 普通出击：点击出击 -> 组队界面点击开始战斗 -> 等待战斗结束
    5. 处理"选择一个卡牌效果"弹窗
    6. 返回关卡选择界面，循环执行直到检测到无尽模式或达到最大刷关数
    
    巡回匹配恢复机制:
    - 当任意步骤识别失败时，启动巡回匹配
    - 循环检测所有关键场景（卡牌选择、战斗结算、奖励弹窗等）
    - 匹配成功后执行对应恢复动作，然后继续主流程
    - 固定巡回 10 次后若仍无法匹配则抛出异常退出
    
    参数格式:
    {
        "max_stages": 60,             // 最大刷关数量，默认 60（6章x10关）
        "battle_timeout": 300,        // 单场战斗超时（秒），默认 300
        "enable_patrol": true,        // 是否启用巡回匹配恢复，默认 true
        "stop_at_endless": true       // 是否在检测到无尽模式(6-1)后停止，默认 true
    }
    """

    def __init__(self):
        super().__init__()
        # 战斗模式初始化标记：单次任务仅在第一场战斗开始后执行一次自动战斗切换
        self._battle_mode_initialized: bool = False

    def _init_battle_mode_auto_once(self, context: Context) -> None:
        """
        在第一场真实战斗开始后，调用战斗模式管理器，将战斗模式切换为自动。
        由于战斗模式会在后续战斗中继承，因此整个 Task 只需执行一次。
        """
        if self._battle_mode_initialized:
            return

        try:
            # 延迟导入，避免循环依赖或在未使用时加载 numpy 等重依赖
            from .battle_mode import BattleModeManager  # type: ignore
        except Exception as e:
            print(f"[DungeonFullAuto] 导入 BattleModeManager 失败，跳过战斗模式初始化: {e}")
            self._battle_mode_initialized = True
            return

        # 构造一个最小 RunArg，仅提供 custom_action_param
        DummyArg = type("DummyArg", (), {})
        argv = DummyArg()
        argv.custom_action_param = {"target_mode": "auto"}
        argv.image = None

        try:
            manager = BattleModeManager()
            manager.run(context, argv)
            print("[DungeonFullAuto] 已在第一场战斗中执行 BattleModeManager，强制切为自动战斗模式")
        except Exception as e:
            print(f"[DungeonFullAuto] 执行 BattleModeManager 失败，跳过战斗模式初始化: {e}")

        self._battle_mode_initialized = True

    def _try_recover(self, context: Context, image, reason: str) -> tuple:
        """
        内部方法：尝试通过巡回匹配恢复
        
        固定巡回 10 次，超过后抛出 RuntimeError 让 MaaFramework 报错退出。
        
        参数:
            context: MAA 上下文
            image: 当前截图（可为 None）
            reason: 触发恢复的原因
            
        返回:
            tuple: (recovered, scene_id)
            - recovered: 是否成功恢复
            - scene_id: 匹配到的场景
            
        异常:
            RuntimeError: 当巡回匹配超过 10 次仍无法匹配时抛出
        """
        print(f"[DungeonFullAuto] 触发巡回匹配恢复，原因: {reason}")
        
        # _patrol_match 内部固定 10 次，超过会抛出 RuntimeError
        scene_id, reco_result, extra_data = _patrol_match(
            context, image, 
            interval=1.0
        )
        
        recovered = _handle_scene_recovery(context, scene_id, reco_result, extra_data)
        return (recovered, scene_id)

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        param = _parse_param(argv.custom_action_param)
        max_stages = param.get("max_stages", 60)
        battle_timeout = param.get("battle_timeout", 300)
        enable_patrol = param.get("enable_patrol", True)
        stop_at_endless = param.get("stop_at_endless", True)

        print(f"[DungeonFullAuto] 开始全自动刷地牢")
        print(f"[DungeonFullAuto] 最大关卡数={max_stages}, 战斗超时={battle_timeout}秒")
        print(f"[DungeonFullAuto] 巡回匹配恢复: {'启用' if enable_patrol else '禁用'}, 固定最大轮数={PATROL_MAX_ROUNDS}")
        print(f"[DungeonFullAuto] 无尽模式处理: {'检测到无尽后停止' if stop_at_endless else '忽略无尽，继续刷关'}")

        # 记录快速作战是否可用（一旦失败就不再尝试）
        quick_battle_available = True
        stages_completed = 0

        for stage_num in range(max_stages):
            print(f"\n{'='*50}")
            print(f"[DungeonFullAuto] ===== 第 {stages_completed + 1} 关 (尝试 {stage_num + 1}/{max_stages}) =====")
            print(f"{'='*50}")

            # === 步骤1：等待画面稳定后获取截图 ===
            current_image = _wait_and_screenshot(context, wait_time=2.0)

            # === 步骤1.5：优先检测无尽模式终止条件 ===
            # 在 ROI [1001, 565, 150, 132] 区域检测"无尽"二字（可配置是否在此处停止）
            if stop_at_endless and _check_endless_mode(context, current_image):
                print(f"[DungeonFullAuto] 检测到无尽模式区域，已完成全部关卡！共完成 {stages_completed} 关")
                return CustomAction.RunResult(success=True)

            # === 步骤2：模板匹配找到黑神图标位置 ===
            # 滑动策略: 当前位置 → 左滑两次 → 右滑回原点（距离减半防止滑过头）
            current_stage_found = False
            stage_click_pos = None  # 用于点击的位置
            
            # 滑动距离: 50像素，持续时间: 1秒
            swipe_distance = 50
            swipe_duration = 1000
            center_y = 400
            
            # 定义滑动序列: (方向, 描述)
            # None=不滑动(当前位置), "right"=右滑, "left"=左滑
            swipe_sequence = [
                (None, "当前位置"),
                ("left", "左滑一次"),
                ("left", "左滑两次"),
                ("right", "右滑回原点(1)"),
                ("right", "右滑回原点(2)"),
            ]
            
            for swipe_idx, (swipe_dir, swipe_desc) in enumerate(swipe_sequence):
                # 执行滑动（第一次不滑动）
                if swipe_dir == "right":
                    print(f"[DungeonFullAuto] {swipe_desc}...")
                    # 右滑: 从左往右拖动（显示左边内容）
                    context.tasker.controller.post_swipe(640 - swipe_distance, center_y, 640 + swipe_distance, center_y, swipe_duration).wait()
                    time.sleep(1.2)
                    context.tasker.controller.post_screencap().wait()
                    current_image = context.tasker.controller.cached_image
                elif swipe_dir == "left":
                    print(f"[DungeonFullAuto] {swipe_desc}...")
                    # 左滑: 从右往左拖动（显示右边内容）
                    context.tasker.controller.post_swipe(640 + swipe_distance, center_y, 640 - swipe_distance, center_y, swipe_duration).wait()
                    time.sleep(1.2)
                    context.tasker.controller.post_screencap().wait()
                    current_image = context.tasker.controller.cached_image
                
                print(f"[DungeonFullAuto] 尝试在{swipe_desc}查找黑神...")
                
                # 使用多模板特征匹配黑神图标（全屏识别）
                heishen_box, matched_template = _find_heishen_multi_template(
                    context, current_image, ROI_FULL_SCREEN, count=4
                )

                if heishen_box:
                    print(f"[DungeonFullAuto] 找到黑神图标位置: {heishen_box}, 模板: {matched_template}")
                    
                    # 直接使用黑神的box作为点击位置（后续会点击y+100）
                    stage_click_pos = heishen_box
                    current_stage_found = True
                    
                    break

            if not current_stage_found:
                print("[DungeonFullAuto] 黑神未找到")
                
                # 启用巡回匹配恢复
                if enable_patrol:
                    recovered, scene_id = self._try_recover(context, current_image, "未找到当前关卡")
                    if recovered:
                        # 根据恢复后的场景决定下一步
                        if scene_id == DungeonScene.STAGE_SELECT:
                            # 重新截图并继续本轮查找
                            print("[DungeonFullAuto] 恢复到关卡选择界面，重新查找黑神...")
                            context.tasker.controller.post_screencap().wait()
                            current_image = context.tasker.controller.cached_image
                            # 再次尝试查找黑神
                            heishen_box, matched_template = _find_heishen_multi_template(
                                context, current_image, ROI_FULL_SCREEN, count=4
                            )
                            if heishen_box:
                                stage_click_pos = heishen_box
                                current_stage_found = True
                        elif scene_id in [DungeonScene.STAGE_DETAIL, DungeonScene.TEAM_SELECT]:
                            # 已经在战斗相关界面，跳过查找步骤
                            print(f"[DungeonFullAuto] 恢复到 {scene_id}，跳过关卡查找")
                            current_stage_found = True
                            stage_click_pos = None  # 标记无需点击进入
                        else:
                            # 其他恢复情况，重新开始本轮
                            print(f"[DungeonFullAuto] 恢复后场景为 {scene_id}，继续下一轮")
                            continue
                    else:
                        print("[DungeonFullAuto] 巡回匹配恢复失败，任务结束")
                        return CustomAction.RunResult(success=False)
                else:
                    print("[DungeonFullAuto] 巡回匹配恢复已禁用，任务结束")
                    return CustomAction.RunResult(success=False)

            # === 步骤3：点击进入关卡 ===
            if stage_click_pos:
                # 点击黑神下方 (y+100)
                _click_heishen_below(context.tasker.controller, stage_click_pos, offset_y=100)
                print(f"[DungeonFullAuto] 点击黑神下方进入关卡")
            
            # === 等待关卡详情页加载 ===
            stage_info_image = _wait_and_screenshot(context, wait_time=2.0)

            # === 步骤4：尝试快速作战或普通出击 ===
            battle_started = False

            if quick_battle_available:
                # 尝试快速作战
                quick_btn_reco = context.run_recognition(
                    "Dungeon_QuickBattleBtn",
                    stage_info_image,
                    {
                        "Dungeon_QuickBattleBtn": {
                            "recognition": "OCR",
                            "expected": "快速作战",
                            "roi": ROI_STAGE_INFO_PANEL,
                        }
                    }
                )

                if _reco_hit(quick_btn_reco):
                    box = _reco_box(quick_btn_reco)
                    _click_box(context.tasker.controller, box)

                    print("[DungeonFullAuto] 已点击快速作战，检查是否生效...")
                    
                    # 等待快速作战响应
                    check_image = _wait_and_screenshot(context, wait_time=2.0)

                    still_in_detail = context.run_recognition(
                        "Dungeon_CheckStillInDetail",
                        check_image,
                        {
                            "Dungeon_CheckStillInDetail": {
                                "recognition": "OCR",
                                "expected": STAGE_NAME_PATTERN,
                                "roi": ROI_TOP_RIGHT_QUARTER,
                            }
                        }
                    )

                    if not _reco_hit(still_in_detail):
                        print("[DungeonFullAuto] 快速作战成功！")
                        battle_started = True
                    else:
                        print("[DungeonFullAuto] 快速作战不可用，本赛季将不再尝试快速作战")
                        quick_battle_available = False

            # 如果快速作战失败或不可用，使用普通出击
            if not battle_started:
                print("[DungeonFullAuto] 使用普通出击...")
                
                # 等待画面稳定后截图
                battle_image = _wait_and_screenshot(context, wait_time=2.0)

                # 点击出击按钮
                battle_btn_reco = context.run_recognition(
                    "Dungeon_NormalBattleBtn",
                    battle_image,
                    {
                        "Dungeon_NormalBattleBtn": {
                            "recognition": "OCR",
                            "expected": "出击",
                            "roi": ROI_STAGE_INFO_PANEL,
                        }
                    }
                )

                if not _reco_hit(battle_btn_reco):
                    print("[DungeonFullAuto] 未找到出击按钮")
                    
                    # 启用巡回匹配恢复
                    if enable_patrol:
                        recovered, scene_id = self._try_recover(context, battle_image, "未找到出击按钮")
                        if recovered and scene_id in [DungeonScene.STAGE_SELECT, DungeonScene.CARD_SELECT, 
                                                       DungeonScene.BATTLE_RESULT, DungeonScene.REWARD_POPUP]:
                            # 恢复到可以继续的状态，重新开始本轮
                            print(f"[DungeonFullAuto] 恢复后场景为 {scene_id}，继续下一轮")
                            continue
                        else:
                            print("[DungeonFullAuto] 巡回匹配恢复失败或不适用")
                            return CustomAction.RunResult(success=False)
                    else:
                        return CustomAction.RunResult(success=False)

                box = _reco_box(battle_btn_reco)
                _click_box(context.tasker.controller, box)
                print("[DungeonFullAuto] 已点击出击按钮")
                time.sleep(1.5)

                # === 步骤5：等待组队界面加载，然后点击开始战斗 ===
                print("[DungeonFullAuto] 等待组队界面加载...")
                team_image = _wait_and_screenshot(context, wait_time=2.0)
                
                launch_clicked = False
                for launch_attempt in range(10):
                    # 重试时重新截图
                    if launch_attempt > 0:
                        time.sleep(0.5)
                        context.tasker.controller.post_screencap().wait()
                        team_image = context.tasker.controller.cached_image
                    
                    launch_reco = context.run_recognition(
                        "Dungeon_LaunchBattle",
                        team_image,
                        {
                            "Dungeon_LaunchBattle": {
                                "recognition": "TemplateMatch",
                                "template": DUNGEON_TEMPLATES["launch_battle"],
                                "roi": [800, 550, 400, 150],
                                "threshold": 0.7,
                            }
                        }
                    )

                    if _reco_hit(launch_reco):
                        box = _reco_box(launch_reco)
                        _click_box(context.tasker.controller, box)
                        print("[DungeonFullAuto] 已点击开始战斗")
                        launch_clicked = True
                        break

                if not launch_clicked:
                    print("[DungeonFullAuto] 无法点击开始战斗按钮")
                    
                    # 启用巡回匹配恢复
                    if enable_patrol:
                        recovered, scene_id = self._try_recover(context, team_image, "无法点击开始战斗")
                        if recovered:
                            # 根据恢复后的场景决定下一步
                            if scene_id in [DungeonScene.STAGE_SELECT, DungeonScene.CARD_SELECT,
                                           DungeonScene.BATTLE_RESULT, DungeonScene.REWARD_POPUP]:
                                print(f"[DungeonFullAuto] 恢复后场景为 {scene_id}，继续下一轮")
                                continue
                            elif scene_id == DungeonScene.TEAM_SELECT:
                                # 还在组队界面，再试一次点击
                                context.tasker.controller.post_screencap().wait()
                                retry_image = context.tasker.controller.cached_image
                                retry_reco = context.run_recognition(
                                    "Dungeon_LaunchBattle_Retry",
                                    retry_image,
                                    {
                                        "Dungeon_LaunchBattle_Retry": {
                                            "recognition": "TemplateMatch",
                                            "template": DUNGEON_TEMPLATES["launch_battle"],
                                            "roi": [800, 550, 400, 150],
                                            "threshold": 0.7,
                                        }
                                    }
                                )
                                if _reco_hit(retry_reco):
                                    _click_box(context.tasker.controller, _reco_box(retry_reco))
                                    print("[DungeonFullAuto] 重试点击开始战斗成功")
                                    launch_clicked = True
                        
                        if not launch_clicked:
                            return CustomAction.RunResult(success=False)
                    else:
                        return CustomAction.RunResult(success=False)

                # === 步骤6：等待战斗开始，然后等待战斗结束 ===
                print("[DungeonFullAuto] 等待战斗开始...")
                _wait_and_screenshot(context, wait_time=3.0)  # 等待战斗加载

                # 仅在第一场战斗加载完成后执行一次自动战斗模式初始化
                self._init_battle_mode_auto_once(context)

                print("[DungeonFullAuto] 等待战斗结束...")
                battle_start_time = time.time()
                battle_ended = False

                while time.time() - battle_start_time < battle_timeout:
                    context.tasker.controller.post_screencap().wait()
                    result_image = context.tasker.controller.cached_image

                    # 检测战斗结束标志
                    victory_reco = context.run_recognition(
                        "Dungeon_VictoryCheck",
                        result_image,
                        {
                            "Dungeon_VictoryCheck": {
                                "recognition": "OCR",
                                "expected": "点击.*继续|返回|胜利|结算",
                                "roi": ROI_FULL_SCREEN,
                            }
                        }
                    )

                    if _reco_hit(victory_reco):
                        print(f"[DungeonFullAuto] 战斗结束: {_reco_detail(victory_reco)}")
                        # 等待 2 秒确保结算画面完全加载后再点击
                        print("[DungeonFullAuto] 等待 2 秒 避免无响应...")
                        time.sleep(2.0)
                        box = _reco_box(victory_reco)
                        _click_box(context.tasker.controller, box)
                        battle_ended = True
                        time.sleep(1.5)
                        break

                    time.sleep(2.0)

                if not battle_ended:
                    print("[DungeonFullAuto] 战斗超时")
                    
                    # 启用巡回匹配恢复：战斗超时可能是战斗结束信号被错过
                    if enable_patrol:
                        recovered, scene_id = self._try_recover(context, result_image, "战斗超时")
                        if recovered:
                            print(f"[DungeonFullAuto] 战斗超时恢复，场景为 {scene_id}")
                            if scene_id == DungeonScene.BATTLE_RESULT:
                                # 确实是战斗结束了，正常处理
                                battle_ended = True
                            elif scene_id in [DungeonScene.STAGE_SELECT, DungeonScene.CARD_SELECT,
                                             DungeonScene.REWARD_POPUP]:
                                # 已经过了战斗阶段，继续流程
                                battle_ended = True
                        else:
                            return CustomAction.RunResult(success=False)
                    else:
                        return CustomAction.RunResult(success=False)

            # === 步骤7：等待战斗结算/卡牌选择弹窗 ===
            print("[DungeonFullAuto] 等待结算画面...")
            card_image = _wait_and_screenshot(context, wait_time=2.0)

            for card_attempt in range(5):
                # 重试时重新截图
                if card_attempt > 0:
                    time.sleep(0.8)
                    context.tasker.controller.post_screencap().wait()
                    card_image = context.tasker.controller.cached_image

                # 检测"选择一个卡牌效果"标题
                title_reco = context.run_recognition(
                    "Dungeon_CardSelectTitle",
                    card_image,
                    {
                        "Dungeon_CardSelectTitle": {
                            "recognition": "OCR",
                            "expected": "选择.*卡牌效果",
                            "roi": ROI_CARD_SELECT_TITLE,
                        }
                    }
                )

                if _reco_hit(title_reco):
                    print("[DungeonFullAuto] 检测到卡牌选择弹窗")

                    # 点击"确认选择"
                    confirm_reco = context.run_recognition(
                        "Dungeon_CardSelectConfirm",
                        card_image,
                        {
                            "Dungeon_CardSelectConfirm": {
                                "recognition": "OCR",
                                "expected": "确认选择",
                                "roi": ROI_CARD_SELECT_CONFIRM,
                            }
                        }
                    )

                    if _reco_hit(confirm_reco):
                        box = _reco_box(confirm_reco)
                        _click_box(context.tasker.controller, box)
                        print("[DungeonFullAuto] 已点击确认选择")
                    else:
                        # 备用：点击屏幕下方
                        context.tasker.controller.post_click(640, 600).wait()
                        print("[DungeonFullAuto] 点击屏幕下方关闭弹窗")

                    time.sleep(1.5)
                    break

                # 检查是否已回到关卡选择界面（使用多模板特征匹配黑神，全屏识别）
                back_heishen_box, _ = _find_heishen_multi_template(
                    context, card_image, ROI_FULL_SCREEN, count=4
                )

                if back_heishen_box:
                    print("[DungeonFullAuto] 已回到关卡选择界面（检测到黑神）")
                    break

                time.sleep(0.8)

            stages_completed += 1
            print(f"[DungeonFullAuto] 第 {stage_num + 1} 关完成！累计完成 {stages_completed} 关")

            # 等待画面稳定后继续下一关
            print("[DungeonFullAuto] 等待返回关卡选择界面...")
            time.sleep(2.0)

        print(f"[DungeonFullAuto] 已完成 {stages_completed} 关，任务结束")
        return CustomAction.RunResult(success=True)
