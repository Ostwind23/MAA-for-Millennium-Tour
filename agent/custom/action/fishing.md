## 自动钓鱼整体设计概览

- **入口类**: `AutoFishing(CustomAction)`
- **核心流程阶段**（状态机）：
  1. **SelectBait**：如果当前在选饵界面，OCR 检测任意数字后点击确认开始钓鱼。
  2. **CastRod**：点击操控杆中心一次，下杆。
  3. **WaitBite**：在屏幕下半部分用 ColorMatch 检测黄色拉扯条（替代原蓝色方案），判断鱼儿上钩。
  4. **Struggle**：拉扯阶段（含体力管理）：
     - 颜色扫描检测黄色箭头方向 → 全幅拖拽摇杆。
     - 拉扯 `STRUGGLE_DRAG_DURATION`(5s) → 松手休息 `STRUGGLE_REST_DURATION`(1.5s) → 循环。
     - 检测是否进入 QTE（降频检测，每 0.5s 一次）。
     - 检测是否黑屏（鱼已钓上）。
  5. **QTE**：检测 3 个鱼形按钮，检测每个按钮外圈白色圆环的半径，计算最佳点击时间并依次点击。含预热帧跳过、fallback 降级等容错机制。
  6. **Result**：通过黑屏 + OCR 确认结算界面，点击屏幕继续。

- **用户参数**：
  - `strategy`: `"aggressive"`（逆向拉扯） / `"conservative"`（顺向拉扯）
  - `max_rounds`: 最大钓鱼轮数
  - `qte_shrink_speed`: QTE 白圈收缩速度（像素/秒）

- **稳定性设计重点**：
  - 所有截图调用都在同一线程中，且每次循环**重新获取 controller**，加短延迟防止 IPC 崩溃。
  - 各阶段检测互不混用（WaitBite 只做 ColorMatch，Struggle/QTE 不做 OCR）。
  - QTE 每个按钮只检测一次外圆半径，后续靠时间戳驱动点击。

---

## 常量与配置

### 屏幕与摇杆

- `SCREEN_WIDTH = 1280`, `SCREEN_HEIGHT = 720`
- `JOYSTICK_ROI = [928, 369, 247, 247]`
- `JOYSTICK_CENTER` 为 ROI 中心 `(1051, 492)`，`JOYSTICK_RADIUS = 123`（ROI 半宽，拖到边缘）。

### 上钩检测（黄色拉扯条）

- `BITE_DETECT_ROI = [2, 260, 1280, 460]`（屏幕下半部分）
- MaaFramework ColorMatch 使用 **RGB** 范围：
  - `BITE_COLOR_LOWER_RGB = [239, 123, 65]`
  - `BITE_COLOR_UPPER_RGB = [255, 143, 85]`
- OpenCV 调试使用对应的 BGR 范围。
- 最少像素数：`BITE_COLOR_COUNT = 200`
- 超时：`BITE_TIMEOUT = 10.0s`
- **设计缘由**：替代原蓝色像素方案。蓝色方案在深蓝色背景钓鱼场景下容易误触发；黄色拉扯条在所有场景下均可靠出现。

### 黄色箭头检测

- `ARROW_DETECT_ROI`：在摇杆 ROI 周围扩展 85px。
- 直接在 BGR 空间用 `cv2.inRange` 检测黄色像素。
- `ARROW_MIN_PIXELS = 100` 防止噪点。
- BGR 范围：`[168, 239, 146]` ~ `[188, 255, 166]`

### 拉扯阶段时间配置

- `STRUGGLE_CHECK_INTERVAL = 0.02s`：主循环帧间隔。
- `STRUGGLE_QTE_CHECK_INTERVAL = 0.5s`：QTE 模板匹配降频间隔，避免每帧做昂贵的 TemplateMatch。
- `STRUGGLE_DRAG_DURATION = 5.0s`：拉扯持续时间，超过后松手休息。
- `STRUGGLE_REST_DURATION = 1.5s`：休息时间，松手恢复体力。

### QTE 配置

- 鱼图标模板：`QTE_FISH_TEMPLATE = "fishing/icon.png"`
- 模板匹配阈值：`QTE_TEMPLATE_THRESHOLD = 0.9`
- 最多 3 个按钮：`QTE_BUTTON_COUNT = 3`
- QTE 最长持续时间：`QTE_MAX_DURATION = 6.0s`
- QTE 轮询间隔：`QTE_CHECK_INTERVAL = 0.03s`
- 收缩速度默认：`QTE_CIRCLE_SHRINK_SPEED = 80.0 像素/秒`
- 点击时机提前量：`QTE_CLICK_OFFSET = -0.08s`（补偿 IPC 延迟）
- 半径修正：`QTE_RADIUS_CORRECTION = 3px`（补偿外缘辉光）
- 点击时间窗口：`QTE_CLICK_WINDOW = 1.0s`
- Fallback 最小方向数：`QTE_FALLBACK_MIN_DIRECTIONS = 4`
- Fallback 超时：`QTE_FALLBACK_TIMEOUT = 2.5s`

### 白色圆圈检测

- BGR 白色范围：`[200, 200, 200]` ~ `[255, 255, 255]`
- 径向扫描方向数：`QTE_CIRCLE_SCAN_DIRECTIONS = 16`
- 最远扫描半径：`QTE_CIRCLE_MAX_RADIUS = 200`
- 最小合格方向比例：有效方向的 3/8（至少 3 个）

### 黑屏检测

- 灰度 < `BLACKOUT_DARK_THRESHOLD = 30` 的暗像素比例大于 `BLACKOUT_RATIO_THRESHOLD = 0.6` 即认为黑屏。

### 调试

- `DEBUG_ENABLED = True` 时启用 OpenCV 窗口和详细日志。

---

## QTE 按钮槽位数据结构

```python
@dataclass
class QTEButtonSlot:
    center: Tuple[int, int]           # 按钮中心位置
    button_radius: int                # 按钮半径
    circle_radius: float = 0.0        # 外圆半径（0=未检测, >0=正常检测, -1=fallback模式）
    first_detect_time: float = 0.0    # 首次检测时间戳
    target_click_time: float = 0.0    # 计算好的点击时间戳
    clicked: bool = False             # 是否已点击
    confidence: float = 0.0           # 模板匹配置信度
    circle_detect_attempts: int = 0   # 外圆检测尝试次数
    max_detected_directions: int = 0  # 历次检测中最大方向数
    warmup_done: bool = False         # 预热帧是否已跳过
```

- `circle_radius` 三态语义：
  - `== 0`：尚未成功检测到外圆。
  - `> 0`：正常检测到的外圆半径。
  - `== -1`：fallback 降级模式（外圆检测失败，用估算值）。
- `max_detected_directions`：区分"圆圈尚未出现"（始终 0-3）和"圆圈存在但检测失败"（4+）。
- `warmup_done`：跳过首帧，避免在圆圈展开动画阶段测量到偏小半径。

---

## 主入口与整体循环

### `AutoFishing.run`

- 解析 `custom_action_param`，支持直接字典、JSON 字符串、双重 JSON 包裹。
- 读取参数：`strategy`，`max_rounds`，`qte_shrink_speed`。
- 初始化实例变量：`self._joystick_held = False`。
- 启动调试窗口（若启用）。
- 进行 `max_rounds` 轮钓鱼：
  - 调用 `_fishing_loop(context, strategy)`。
  - 成功则累加 `success_count`；连续失败达 2 次则提前终止。
- 返回 `RunResult(success=success_count > 0)`。

### `_fishing_loop`

1. **选饵（幂等）**：
   - `_select_bait(context)`：截图一次 → OCR 检测数字 → 命中则点击屏幕中央。

2. **下杆**：
   - `_cast_rod(context)`：点击 `JOYSTICK_CENTER`，等待 `CAST_ROD_DELAY`。

3. **等待上钩**：
   - `_wait_for_bite(context)`：循环检测屏幕下半部分的黄色拉扯条。
   - ColorMatch 检测 `BITE_DETECT_ROI` 中黄色像素数 ≥ `BITE_COLOR_COUNT`。
   - 命中 → 点击 `JOYSTICK_CENTER` 收杆。

4. **拉扯阶段**：
   - `_struggle_phase(context, strategy)`。

5. **结果处理**：
   - 若返回 `"caught"` → `_handle_result(context)` 点击继续。

---

## 关键子模块逻辑

### 1. 黄色箭头方向检测

函数：`_detect_arrow_direction_by_color(image) -> Optional[(angle, debug_info)]`

- 步骤：
  1. 在 `ARROW_DETECT_ROI` 裁剪 ROI。
  2. 用 `cv2.inRange`（BGR）得到黄色掩码。
  3. `cv2.findNonZero(mask)` 获取所有黄色像素坐标。
  4. 若像素数 < `ARROW_MIN_PIXELS` 则认为未检测到。
  5. 提取 `x_coords`，用 `argmin/argmax` 找最左/最右像素，取中点。
  6. 从 `JOYSTICK_CENTER` 指向中点的向量计算角度 `angle = atan2(dy, dx)`。

- 稳定性：像素阈值过滤噪点；不依赖旋转模板，任意角度均适用。

### 2. 摇杆操作

#### 全幅拖拽（拉扯阶段使用）

函数：`_hold_joystick_direction(context, angle_deg)`

- **核心机制**：每次调用都执行完整的 **touch_up → touch_down（中心）→ touch_move（边缘）** 循环。
- 流程：
  1. 若 `_joystick_held = True`：先 `post_touch_up` 释放上一次触摸。
  2. `post_touch_down` 在 `JOYSTICK_CENTER` 按下（从正中心起始）。
  3. `post_touch_move` 到目标方向边缘 `(center + JOYSTICK_RADIUS × direction)`。
  4. 设置 `_joystick_held = True`。
- **设计缘由**：
  - 旧方案（首帧 touch_down + 后续帧仅 touch_move）：当箭头方向变化很小时，连续 touch_move 只在边缘微移，游戏虚拟摇杆无法识别为有效的全幅输入。
  - 新方案每帧都重做"中心→边缘"的完整拖拽行程，确保游戏每帧都能识别到 `JOYSTICK_RADIUS`(123px) 的最大位移。
  - 总耗时约 15~20ms/帧（3 次 touch 操作），远低于旧版 `_safe_swipe` 的 500ms 完整滑动。

#### 释放摇杆

函数：`_release_joystick(context)`

- 若 `_joystick_held = True`：执行 `post_touch_up`，设置 `_joystick_held = False`。
- 在以下时机调用：进入休息、进入 QTE、黑屏检测、退出拉扯阶段。

#### 完整滑动（非拉扯阶段备用）

函数：`_drag_joystick(context, angle_deg, duration_ms)`

- 使用 `_safe_swipe` 分多步 touch_down → touch_move → touch_up，模拟慢速滑动。
- 仅用于非拉扯场景，拉扯阶段始终使用 `_hold_joystick_direction`。

### 3. 拉扯阶段（体力管理版）

函数：`_struggle_phase(context, strategy) -> "caught" | "failed"`

- **体力管理节奏**：
  - 拉扯 `STRUGGLE_DRAG_DURATION`(5.0s) → 松手休息 `STRUGGLE_REST_DURATION`(1.5s) → 循环。
  - 通过 `is_resting` 标志和 `drag_phase_start` 时间戳控制周期切换。
  - 休息时调用 `_release_joystick` 彻底松手（touch_up）。
  - 恢复拉扯时，下一帧的 `_hold_joystick_direction` 自动从中心重新 touch_down。

- **主循环**（最长 120s）：
  1. **节奏控制**（每帧顶部）：
     - 拉扯中：检查是否该休息 → `_release_joystick` → `is_resting = True`。
     - 休息中：检查是否该恢复 → `is_resting = False`。
  2. **安全截图**：重取 controller + `post_screencap` + 30ms 延迟。
  3. **黑屏预筛**（≤5ms）：`_check_screen_blackout` → 命中则等 0.5s 后 OCR 确认结算 → 返回 `"caught"`。
  4. **箭头方向检测**（每帧都做）：
     - `_detect_arrow_direction_by_color` 检测黄色箭头。
     - 仅在**拉扯中**（`not is_resting`）才操作摇杆：
       - `"aggressive"` 策略：`drag_angle = angle + 180`（逆向）。
       - `"conservative"` 策略：`drag_angle = angle`（顺向）。
       - 调用 `_hold_joystick_direction(context, drag_angle)`。
     - 休息中检测箭头但不操作（为恢复后的首帧准备方向）。
  5. **QTE 检测**（降频，每 `STRUGGLE_QTE_CHECK_INTERVAL`(0.5s) 一次）：
     - `_detect_all_fish_buttons` 检测鱼形按钮。
     - 若命中 → `_release_joystick` → `_handle_qte_dynamic` → QTE 完成后 1s 冷却 → 重置拉扯周期。
  6. 帧间隔：`sleep(STRUGGLE_CHECK_INTERVAL)`(20ms)。

- **QTE 后重置**：`is_resting = False`，`drag_phase_start = now`，确保 QTE 结束后立即开始新一轮拉扯。

- **finally**：无论如何退出都调用 `_release_joystick` 确保触摸状态清理。

### 4. QTE 鱼按钮检测（模板匹配）

函数：`_detect_all_fish_buttons(context, image) -> List[dict]`

- 用 `"fishing/icon.png"` 做 TemplateMatch（全屏，阈值 0.9）。
- 每个结果提取 `box` → 计算 `center`、`radius = max(w,h)/2 + 扩展偏移`。
- 按 `confidence` 降序排序，只保留前 3 个。
- 按钮半径通过鱼图标 box 扩展计算：
  - `QTE_BUTTON_EXPAND_LEFT/TOP/WIDTH/HEIGHT` 四个偏移量。

### 5. 白色圆圈半径检测（径向扫描）

函数：`_detect_circle_radius(image, center, button_radius) -> (radius, direction_count)`

- 前提：已知按钮中心和按钮半径。
- 返回值：`(radius, dir_count)`，radius=0 表示未通过检测。

- 步骤：
  1. 起始半径 `button_radius + 5`，最大半径 `QTE_CIRCLE_MAX_RADIUS`(200)。
  2. 16 个方向，每方向**从外向内**扫描，找最外层白色像素。
  3. **OOB 判定改进**：每方向统计屏幕内采样点数，若 < 半数则视为越界跳过（不计入有效方向）。
  4. **3 点投票验证**：检测到的白色边缘往外 `[8, 12, 16]px` 三个验证点，≥2 个仍为白色则判定为背景干扰丢弃。
  5. 合格阈值：有效方向的 3/8（至少 3 个），取平均半径。

- 稳定性：
  - 3 点投票比单点验证更抗噪。
  - 3/8 阈值比原 1/2 更宽松，适配屏幕边缘按钮（OOB 方向较多）。

### 6. QTE 动态规划与点击调度

函数：`_handle_qte_dynamic(context) -> "done" | "caught"`

- 维护 3 个 `QTEButtonSlot` 槽位。

- 辅助函数：
  - `find_slot_by_center(center, tolerance=30)`：容差匹配已知按钮。
  - `find_empty_slot()`：寻找空槽。
  - `count_clicked()`：统计已点击数。

- **主循环**（最长 `QTE_MAX_DURATION`(6s)）：
  1. 安全截图，记录 `capture_time`。
  2. 黑屏检测 → 等 0.5s → OCR 确认 → 返回 `"caught"`。
  3. 优化：所有槽位已填满且都有圆圈数据时，跳过模板匹配。
  4. **新按钮分配**：遍历 `buttons`，无对应槽位则分配新 `QTEButtonSlot`。
  5. **圆圈检测循环**（独立于模板匹配）：
     - 对所有 `circle_radius == 0` 且未点击的槽位执行检测。
     - **预热帧跳过**：每个槽位首次检测机会跳过不执行，等圆圈展开动画完成。
     - 检测成功（`radius > 0`）→ 计算 `shrink_distance`、`wait_time`、`target_click_time`。
     - 检测失败 → 根据条件触发 fallback：
       - **快速 fallback**：`max_detected_directions ≥ 4` 且 `attempts ≥ 2`（圆圈存在但无法通过阈值）。
       - **超时 fallback**：`elapsed ≥ 2.5s`（圆圈始终未出现）。
  6. **精准定时点击**：
     - 按 `target_click_time` 排序待点击槽位。
     - 距点击 > 500ms：跳出回截图循环，继续检测其他按钮。
     - 距点击 0~500ms：`time.sleep(wait)` 精确等待。
     - 到达点击时刻：`post_click` 并标记 `clicked = True`。
     - 超过 `QTE_CLICK_WINDOW`(1.0s)：超时警告，标记为已处理。
  7. 所有 3 个按钮点击完成 → 等待 1s 动画 → 返回 `"done"`。

- **Fallback 机制**（`_apply_fallback`）：
  - 用 `_calculate_qte_wait_time_fallback` 估算等待时间（假设初始半径 = 按钮半径 × 2）。
  - `circle_radius = -1` 标记为 fallback 模式。
  - `target_click_time = max(ideal_click_time, capture_time)`，若已过期则立即点击。

### 7. 黑屏检测与结果处理

- **黑屏检测** `_check_screen_blackout`：
  - 灰度化后统计亮度 < 30 的像素比例，大于 0.6 视为黑屏。
  - 调试模式下记录接近阈值（0.4~0.8）的情况。

- **结算页面检测** `_check_result_screen`：
  - 全屏 OCR 检测"特质"或"重量"关键词。
  - 黑屏 + OCR 二次确认，避免过场动画误判。

- **结果处理** `_handle_result`：
  - 等待 0.5s 界面稳定 → 点击屏幕中央。
  - 循环最多 20 次检测结算页面是否消失，每次点击 + 等 0.3s。
  - 超时 6s 后返回。

---

## 辅助函数

- `_parse_param(custom_action_param) -> dict`：安全解析参数（字典/JSON/双重JSON）。
- `_reco_hit(reco_result) -> bool`：检查识别结果是否命中。
- `_reco_box(reco_result) -> Optional[list]`：提取命中的 box。
- `_box_center(box) -> Optional[Tuple]`：计算 box 中心坐标。
- `_clamp_roi(roi) -> List[int]`：将 ROI 裁剪到屏幕范围内。
- `_debug_log(message)`：同时输出到控制台和日志文件。
- `_safe_swipe(context, x1, y1, x2, y2, duration)`：多步触摸滑动（备用，拉扯阶段不使用）。

---

## 稳定性评估要点（供检查使用）

- **IPC / 截图稳定性**：
  - 所有阶段截图调用都包裹在 try/except 中，异常会打印并通过短暂 sleep 重试。
  - 每一轮循环都重新通过 `context.tasker.controller` 获取 controller 句柄。

- **识别负载控制**：
  - WaitBite 阶段仅做单一 ColorMatch 检测。
  - Struggle 阶段：黑屏检测（每帧）+ 箭头颜色检测（每帧）+ QTE 模板匹配（降频 0.5s/次）。
  - QTE 阶段：黑屏检测 + 模板匹配 + 白圈径向扫描，不做箭头/OCR。

- **摇杆操作可靠性**：
  - 每帧完整 touch_up → touch_down（中心）→ touch_move（边缘），确保全幅位移。
  - 休息时彻底 touch_up 松手；恢复时自动从中心重新开始。
  - 进入 QTE / 黑屏 / 退出时都显式调用 `_release_joystick` 清理触摸状态。

- **QTE 鲁棒性**：
  - 预热帧跳过：避免在圆圈展开动画阶段测量偏小半径 → 过早点击。
  - OOB 改进：半数采样点越界才跳过方向，避免边缘按钮被误判。
  - 3 点投票验证：`[8, 12, 16]px` 外侧验证，比单点更抗白云背景噪声。
  - 3/8 方向阈值：比 1/2 更宽松，适配屏幕边缘按钮。
  - 双层 fallback：快速 fallback（圆圈存在但检测失败）+ 超时 fallback（圆圈未出现）。
  - 圆圈检测独立于模板匹配：即使按钮模板消失，已知槽位仍继续检测圆圈。

- **体力管理**：
  - 拉扯 5s → 休息 1.5s 周期，避免体力耗尽导致鱼逃脱。
  - QTE 完成后自动重置拉扯周期。
  - 休息期间仍持续检测黑屏和 QTE，确保不错过关键事件。

- **错误恢复与退出条件**：
  - 等待/拉扯/QTE/结果处理都有明确超时时间与日志输出。
  - 总体钓鱼轮数有上限，通过 `max_rounds` 控制。
  - 连续失败轮数达 2 次后提前结束。

---

## 建议的后续检查点

- 调整/确认以下参数是否与实机表现匹配：
  - `BITE_COLOR_*`（黄色拉扯条颜色范围）是否覆盖各种时间段/地图。
  - `ARROW_MIN_PIXELS` 在线上环境中的典型像素数分布。
  - `STRUGGLE_DRAG_DURATION` 和 `STRUGGLE_REST_DURATION` 是否适合当前体力消耗速率。
  - QTE 白圈的 `QTE_CIRCLE_COLOR_*` 和 `QTE_CIRCLE_MAX_RADIUS` 是否足够稳健。
  - `QTE_CLICK_OFFSET` 与 `QTE_CLICK_WINDOW` 对整体成功率的影响。
  - `QTE_CIRCLE_SHRINK_SPEED` 默认值（80）是否与实际收缩速度匹配。

- 观察日志：
  - QTE 阶段是否频繁出现"外圆未出现"或 fallback 触发日志。
  - Struggle 阶段箭头检测→拖拽的延迟是否稳定在 15~20ms。
  - 体力管理周期（拉扯/休息）的日志时间戳是否符合预期。
  - WaitBite 阶段是否存在误触发。
