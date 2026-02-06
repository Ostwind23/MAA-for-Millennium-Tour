## 自动钓鱼整体设计概览

- **入口类**: `AutoFishing(CustomAction)`
- **核心流程阶段**（状态机）：
  1. **SelectBait**：如果当前在选饵界面，OCR 检测任意数字后点击确认开始钓鱼。
  2. **CastRod**：点击操控杆中心一次，下杆。
  3. **WaitBite**：在专用 ROI 内用 ColorMatch 检测蓝色像素，判断鱼儿上钩。
  4. **Struggle**：拉扯阶段：
     - 颜色扫描检测黄色箭头方向。
     - 检测是否进入 QTE。
     - 检测是否黑屏（鱼已钓上）。
  5. **QTE**：检测 3 个鱼形按钮，检测每个按钮外圈白色圆环的半径，计算最佳点击时间并依次点击。
  6. **Result**：通过黑屏检测确认结果界面结束，点击屏幕继续。

- **用户参数**：
  - `strategy`: `"aggressive"`（逆向拉扯） / `"conservative"`（顺向拉扯）
  - `max_rounds`: 最大钓鱼轮数
  - `qte_shrink_speed`: QTE 白圈收缩速度（像素/秒）

- **稳定性设计重点**：
  - 所有截图调用都在同一线程中，且每次循环**重新获取 controller**，加短延迟防止 IPC 崩溃。
  - 各阶段检测互不混用（WaitBite 只做 ColorMatch，Struggle/QTE 不做 OCR）。
  - QTE 只为每个按钮检测一次外圈半径，避免重复重计算。

---

## 常量与配置

- **屏幕与摇杆**：
  - `SCREEN_WIDTH = 1280`, `SCREEN_HEIGHT = 720`
  - `JOYSTICK_ROI = [928, 369, 247, 247]`
  - `JOYSTICK_CENTER` 为 ROI 中心，`JOYSTICK_RADIUS` 为 ROI 半径（直接拖到边缘）。

- **上钩检测（蓝色）**：
  - `BITE_DETECT_ROI = [1034, 476, 182, 179]`
  - Maa ColorMatch 使用 **RGB** 范围：
    - `BITE_COLOR_LOWER_RGB = [9, 66, 116]`
    - `BITE_COLOR_UPPER_RGB = [49, 106, 156]`
  - OpenCV 调试使用对应的 BGR 范围。
  - 最少像素数：`BITE_COLOR_COUNT = 100`
  - 超时：`BITE_TIMEOUT = 10.0s`

- **黄色箭头检测**：
  - `ARROW_DETECT_ROI`：在摇杆 ROI 周围扩展一圈。
  - 直接在 BGR 空间用 `cv2.inRange` 检测黄色像素。
  - `ARROW_MIN_PIXELS = 100` 防止噪点。

- **QTE 检测**：
  - 鱼图标模板：`QTE_FISH_TEMPLATE = "fishing/icon.png"`
  - 模板匹配阈值：`QTE_TEMPLATE_THRESHOLD = 0.75`
  - 最多 3 个按钮：`QTE_BUTTON_COUNT = 3`
  - QTE 最长持续时间：`QTE_MAX_DURATION = 6.0s`
  - QTE 轮询间隔：`QTE_CHECK_INTERVAL = 0.03s`
  - 收缩速度默认：`QTE_CIRCLE_SHRINK_SPEED = 10.0 像素/秒`

- **白色圆圈检测**：
  - BGR 白色范围：
    - `QTE_CIRCLE_COLOR_LOWER = [200, 200, 200]`
    - `QTE_CIRCLE_COLOR_UPPER = [255, 255, 255]`
  - 径向扫描方向数：`QTE_CIRCLE_SCAN_DIRECTIONS = 16`
  - 最远扫描半径：`QTE_CIRCLE_MAX_RADIUS = 150`

- **黑屏检测**：
  - 灰度 < `BLACKOUT_DARK_THRESHOLD = 30` 的暗像素比例大于 `BLACKOUT_RATIO_THRESHOLD = 0.6` 即认为黑屏。

- **调试**：
  - `DEBUG_ENABLED = True` 时启用 OpenCV 窗口和详细日志。

---

## 主入口与整体循环

### `AutoFishing.run`

- 解析 `custom_action_param`，支持：
  - 直接字典、JSON 字符串、双重 JSON 包裹。
- 读取参数：
  - `strategy`，`max_rounds`，`qte_shrink_speed`。
- 启动调试窗口（若启用）。
- 进行 `max_rounds` 轮钓鱼：
  - 调用 `_fishing_loop(context, strategy)`。
  - 成功则累加 `success_count`；连续失败达 2 次则提前终止。
- 返回 `RunResult(success=success_count > 0)`。

### `_fishing_loop`

1. **选饵（幂等）**：
   - `_select_bait(context)`：
     - 截图一次，OCR 检测 `BAIT_SELECT_ROI` 内是否存在任意数字。
     - 若命中，则点击屏幕中央开始钓鱼。
     - 否则直接返回 False，但不影响后续流程（认为已处于钓鱼界面）。

2. **下杆**：
   - `_cast_rod(context)`：
     - 直接点击 `JOYSTICK_CENTER`。
     - 等待 `CAST_ROD_DELAY` 后返回 True/False。

3. **等待上钩**：
   - `_wait_for_bite(context)`：
     - 循环直到超时：
       - 每次循环：
         - 重新获取 `controller`，`post_screencap().wait()`，`sleep(0.03)`。
         - 从 `controller.cached_image` 取图。
         - 使用 `ColorMatch` 检测 `BITE_DETECT_ROI` 中蓝色连通像素数是否 >= `BITE_COLOR_COUNT`。
         - 若开启调试，绘制 ROI 和检测状态，并打印 `reco.hit`/`best_result`。
       - 命中则：
         - 打印“检测到鱼上钩”，点击 `JOYSTICK_CENTER` 收杆。
         - 返回 True。
   - 若超时，返回 False。

4. **拉扯阶段**：
   - `_struggle_phase(context, strategy)`：
     - 循环最长 `max_duration = 120s`：
       - 安全截图（同样模式重取 controller + 短延迟）。
       - 先检测黑屏 `_check_screen_blackout`，命中则返回 `"caught"`。
       - 再检测 QTE：
         - `_detect_all_fish_buttons` 返回至少 1 个按钮，则：
           - 切换调试阶段为 QTE。
           - 调用 `_handle_qte_dynamic`，若返回 `"caught"` 则直接结束。
           - 否则 QTE 完成继续拉扯。
       - 否则检测黄色箭头方向 `_detect_arrow_direction_by_color`：
         - 得到 `angle`，根据 `strategy` 决定拖拽角度：
           - `"aggressive"`：`drag_angle = angle + 180`（逆向拉）。
           - `"conservative"`：`drag_angle = angle`（顺向拉）。
         - `_drag_joystick` 用 `_safe_swipe` 进行实际拖拽。
       - 每次循环末尾 `sleep(STRUGGLE_CHECK_INTERVAL)`。
     - 超时未结束则返回 `"failed"`。

5. **结果处理**：
   - 若 `_struggle_phase` 返回 `"caught"`：
     - 打印成功信息，调用 `_handle_result(context)`。
   - 否则视为失败。

---

## 关键子模块逻辑

### 1. 黄色箭头方向检测

函数：`_detect_arrow_direction_by_color(image) -> Optional[(angle, debug_info)]`

- 步骤：
  1. 在 `ARROW_DETECT_ROI` 裁剪 ROI。
  2. 用 `cv2.inRange`（BGR）得到黄色掩码。
  3. `cv2.findNonZero(mask)` 获取所有黄色像素坐标 `points`。
  4. 若像素数 < `ARROW_MIN_PIXELS` 则认为未检测到。
  5. 从 `points` 中：
     - 提取 `x_coords`，用 `argmin/argmax` 找最左/最右像素索引。
     - 得到全图坐标 `left_x, left_y` 和 `right_x, right_y`。
     - 取中点 `mid_x, mid_y`。
  6. 从 `JOYSTICK_CENTER` 指向 `(mid_x, mid_y)` 的向量计算角度 `angle = atan2(dy, dx)`。
  7. 在调试模式下，回传 `debug_info` 做可视化。

- 稳定性要点：
  - 像素阈值过滤噪点。
  - 不依赖旋转模板，任意角度的圆弧都可通过极值点定位。

### 2. 摇杆拖拽

函数：`_drag_joystick(context, angle_deg, duration_ms)`

- 计算终点坐标：
  - `end_x = JOYSTICK_CENTER.x + JOYSTICK_RADIUS * cos(angle)`
  - `end_y = JOYSTICK_CENTER.y + JOYSTICK_RADIUS * sin(angle)`
- 使用 `_safe_swipe` 在多个步长内模拟 touch_down → 多次 touch_move → touch_up。
- 任一步出错都会打印日志并尽量 touch_up 收尾。

### 3. QTE 鱼按钮检测（模板匹配）

函数：`_detect_all_fish_buttons(context, image) -> List[dict]`

- 用单个模板 `"fishing/icon.png"` 做 TemplateMatch：
  - ROI 全屏。
  - 阈值 `0.75`。
- 若识别命中：
  - 优先使用 `all_results`；否则退回 `best_result`。
  - 每个结果：
    - 提取 `box`，计算中心点 `center`。
    - 使用 box 宽高估计按钮半径 `radius = max(w,h)/2 + 20`。
    - 若有 `score` 字段则作为 `confidence`。
  - 按 `confidence` 降序排序，只保留前 3 个结果。

- 输出字段：
  - `box`, `center`, `radius`, `confidence`。

### 4. 白色圆圈半径检测（径向扫描）

函数：`_detect_circle_radius(image, center, button_radius) -> float`

- 前提：
  - 已知按钮中心 `center` 与按钮半径 `button_radius`。

- 步骤：
  1. 设起始半径 `start_radius = button_radius + 5`，最大半径为离屏幕边缘与 `QTE_CIRCLE_MAX_RADIUS` 中较小值。
  2. 定义辅助函数 `is_white_pixel(px, py)` 判断像素是否在白色阈值范围内。
  3. 对 `QTE_CIRCLE_SCAN_DIRECTIONS`（16 个方向）循环：
     - 每个方向角度为 `2π * i / 16`，计算 `(cos, sin)`。
     - 从 `max_radius` 向 `start_radius` 反向扫描：
       - 找到第一个白色像素，记作该方向的 `last_white_radius`（最外层白圈）。
  4. 对于每个方向的 `last_white_radius > 0`：
     - 再往外 10 像素，检查对应点是否仍为白色：
       - 若是白色：判定可能是背景白云，**丢弃该方向结果**。
       - 否则：保留该方向的 `last_white_radius`。
  5. 若最终有效方向数 ≥ 一半（8 个），返回这些半径的平均值作为圆圈半径。
  6. 若不足，则返回 0，表示本帧未能可靠检测到圆圈。

- 稳定性要点：
  - 外侧 10 像素验证结合“圆圈厚度 ≤7 像素”的先验信息，有效排除白色背景。
  - 需要多方向一致性投票，防止局部噪点影响整体判断。
  - 每个按钮只在首次出现时检测一次圆圈，避免重复开销。

### 5. QTE 动态规划与点击调度

函数：`_handle_qte_dynamic(context) -> "done" | "caught"`

- 内部维护 3 个 `QTEButtonSlot` 槽位：
  - `center`, `button_radius`, `circle_radius`, `first_detect_time`, `target_click_time`, `clicked`, `confidence`。

- 辅助函数：
  - `find_slot_by_center(center, tolerance=30)`：在已有槽位中以一定容差找到匹配按钮。
  - `find_empty_slot()`：寻找空槽位。
  - `count_clicked()`：统计已点击按钮数。

- 主循环（最长 `QTE_MAX_DURATION` 秒）：
  1. 安全截图；记录 `capture_time`。
  2. 若检测到黑屏 → 立即返回 `"caught"`。
  3. 调用 `_detect_all_fish_buttons` 获取当前可见按钮列表。
  4. 若没有按钮且 `count_clicked() >= 3` → QTE 完成，返回 `"done"`。
  5. 对每个当前检测到的按钮：
     - 若尚无对应槽位：
       - 分配空槽位，初始化 `QTEButtonSlot`。
       - 调用 `_detect_circle_radius`：
         - 若返回半径 `circle_radius > 0`：
           - 计算 `shrink_distance = circle_radius - button_radius`。
           - 计算等待时间 `wait_time = shrink_distance / qte_shrink_speed`。
           - 设定 `target_click_time = capture_time + wait_time + QTE_CLICK_OFFSET`。
         - 否则：暂不设定点击时间，后续帧可再次检测圆圈。
     - 若已有槽位但 `circle_radius == 0`：
       - 再次尝试 `_detect_circle_radius`，成功后同样计算 `target_click_time`。
  6. 调试模式下调用 `draw_qte_buttons` 显示按钮与点击状态。
  7. 遍历槽位，寻找需要点击的按钮：
     - 跳过 `slot is None` 或 `slot.clicked`。
     - 跳过 `circle_radius == 0`（尚未检测到圆圈）。
     - 计算 `time_diff = capture_time - target_click_time`：
       - 若 `-0.05 ≤ time_diff ≤ QTE_CLICK_WINDOW(0.5)`：
         - 判定为可点击，执行点击并 `slot.clicked = True`，本帧只点击 1 个按钮。
       - 若 `time_diff > QTE_CLICK_WINDOW`：
         - 视为超时，打印警告并 `slot.clicked = True` 防止重复报错。
  8. 每轮末尾 `sleep(QTE_CHECK_INTERVAL)`。

- 稳定性要点：
  - 槽位上限为 3，避免在高噪场景中无节制记忆按钮。
  - 每个按钮只检测一次圆圈，后续只依靠已计算好的时间戳对比，减轻负担。
  - 有明确的点击时间窗口与超时报警，方便调试和参数调整。

### 6. 黑屏检测与结果处理

- 黑屏检测 `_check_screen_blackout`：
  - 灰度化后统计亮度 < 30 的像素比例，大于 0.6 视为黑屏。

- 结果处理 `_handle_result`：
  - 循环截图最多 30 次：
    - 若已经不是黑屏，则点击屏幕中央继续。
    - 若始终是黑屏，则打印超时日志。

---

## 稳定性评估要点（供检查使用）

- **IPC / 截图稳定性**：
  - 所有阶段截图调用都包裹在 try/except 中，异常会打印并通过短暂 sleep 重试，而非直接崩溃。
  - 每一轮循环都重新通过 `context.tasker.controller` 获取 controller 句柄。

- **识别负载控制**：
  - WaitBite 阶段仅做单一 ColorMatch 检测。
  - Struggle 阶段只做：黑屏检测 + QTE 模板匹配 + 黄色箭头颜色检测，不做 OCR。
  - QTE 阶段只做：黑屏检测 + 模板匹配 + 白圈径向扫描，不做箭头/OCR。

- **鲁棒性逻辑**：
  - 上钩蓝色检测限定特定 ROI + 像素计数阈值 + 连通约束。
  - 黄色箭头使用最左/最右黄色像素构造中线，天然旋转不变。
  - QTE 白圈检测结合：
    - 多方向一致性（至少一半方向）
    - 径向从外向内扫描
    - 外侧 10 像素验证排除白云干扰。

- **错误恢复与退出条件**：
  - 等待/拉扯/QTE/结果处理都有明确超时时间与日志输出。
  - 总体钓鱼轮数有上限，通过 `max_rounds` 控制。
  - 连续失败轮数达到阈值后提前结束，避免死循环。

---

## 建议的后续检查点

- 调整/确认以下参数是否与实机表现匹配：
  - `BITE_COLOR_*`（蓝色上钩亮度范围）是否覆盖各种时间段/地图。
  - `ARROW_MIN_PIXELS` 在线上环境中的典型像素数分布。
  - QTE 白圈的 `QTE_CIRCLE_COLOR_*` 和 `QTE_CIRCLE_MAX_RADIUS` 是否足够稳健。
  - `QTE_CLICK_OFFSET` 与 `QTE_CLICK_WINDOW` 对整体成功率的影响。

- 观察日志：
  - QTE 阶段是否频繁出现“外圆未出现”或“超时”日志。
  - WaitBite 阶段 `reco.hit` 是否偶发抖动。

整体来看，`fishing.py` 在结构上是一个清晰的有限状态机，各阶段职责明确、检测模块分离，并在 IPC 与识别负载、噪声鲁棒性方面做了较多防护，具备较好的扩展与调参空间。
