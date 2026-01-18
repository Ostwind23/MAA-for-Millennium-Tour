# 项目改进计划 (Update Wish List)

> 基于对当前项目的深度分析和讨论，整理出以下改进方向和待办事项。
> 本文档既可作为快速浏览的索引，也包含完整的技术细节和实现思路。

---

## 一、当前项目状态

### 1.1 已完成的工作

- ✅ **独立完成 YOLOv8 模型完整流程**：数据采集 → 标注 → 训练 → 部署
- ✅ **模型已接入实际任务链**：
  - `farm_event.py` 用于农场捉虫/修水车等事件
  - 通过 MaaFramework 的 `NeuralNetworkDetect` 识别类型调用
  - 模型文件位于 `assets/resource/model/detect/farming.onnx`
  - 当前支持两个类别：`bugs`（虫子，index=0）和 `girl`（角色，index=1）
- ✅ **使用了训练技巧**：
  - 数据增强（各种参数配置）
  - 余弦退火学习率（Cosine Annealing）
  - 曾尝试修改注意力机制（但效果倒退）
- ✅ **实现了调试可视化窗口**：
  - 基于 Tkinter 的 `TkDebugViewer` 类
  - 支持实时显示检测框、目标位置、摇杆区域等
  - 采用"主线程推送数据 + 独立线程显示"的架构
- ✅ **设计了截图采集器**：
  - `ScreenshotCollector`：单次截图采集
  - `BatchScreenshotCollector`：批量截图采集
  - `ConditionalScreenshotCollector`：条件截图采集（识别成功时才保存）

### 1.2 当前主要问题

#### 问题 1：遮挡导致识别失败，角色卡死

**现象描述**：

- 角色被灌木作物遮挡时（农场地上有已成长的灌木作物）
- 角色靠近坑位时，坑位会弹出"含水量/成熟时间"等 UI，遮挡角色大部分身体
- 识别失败后角色会卡在某个地方无法脱出

**当前补救逻辑**（不够好）：

```python
# farm_event.py 中的处理
if current_pos is None:
    lost_count += 1
    # 复用捉虫逻辑：检测不到角色时，先向上轻推摇杆两次尝试把角色拉回视野
    _joystick_nudge_up(context, times=2, duration_ms=500, wait_s=0.2)
```

这种"碰运气式"的补救有时有效，有时反而让角色更偏。

#### 问题 2：移动逻辑僵硬，坑位附近来回抖动/走错坑

**根本原因**：
摇杆方向计算永远推到最大半径，力度始终最大：

```python
# farm_event.py 第 249-250 行
norm_dx = dx / dist * JOYSTICK_RADIUS
norm_dy = dy / dist * JOYSTICK_RADIUS
```

**导致的问题**：

- 离目标很近时也"猛推"，角色会越过交互范围
- 下一帧检测到又往回推，变成来回进出
- 可能走到相邻的坑位去

#### 问题 3：性能开销大，调试查看器帧率只有 2-3 帧

**瓶颈分析**：

1. **ADB 截图本身就慢**（跨进程/跨设备传图）：

   ```python
   context.tasker.controller.post_screencap().wait()
   image = context.tasker.controller.cached_image
   ```

2. **调试窗口每帧做很多"重活"**：

   ```python
   # farm_event.py 第 520-592 行的刷新流程
   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR→RGB 转换
   pil_image = Image.fromarray(image_rgb)              # numpy→PIL
   overlay = Image.new('RGBA', pil_image.size, ...)    # 创建透明层
   # ... 绘制各种框和文字 ...
   pil_image = Image.alpha_composite(pil_image, overlay)  # 半透明合成（开销大）
   self._photo_image = ImageTk.PhotoImage(pil_image)      # 转 Tk 图片对象
   ```

**对比**：示例 YOLO 项目流畅是因为画面来自本地摄像头/视频文件（读取快），或直接在 GPU/内存里流转。

#### 问题 4：训练提升方向不明

- 尝试过修改注意力机制，但训练参数全面倒退
- 不清楚下一步该怎么提升模型

---

## 二、改进方向

### 2.1 数据层面（优先级最高，决定上限的 70%）

> **核心观点**：模型效果的上限主要由数据决定，而不是模型结构。对于你这个项目，最大的问题不是"模型不够聪明"，而是它没见过足够多的"角色被 UI 挡住"的样子。

#### 2.1.1 硬负样本/失败样本闭环（强烈推荐）

**为什么要做**：
你现在的截图采集器是"随便截"。更聪明的做法是：每次脚本因为"识别不到角色"或"认错了"而失败时，自动把那一刻的截图保存下来。这些"难题"图片比随机图片值钱 10 倍。

**具体实现**：

- [ ] 每次"识别丢失/误检导致动作失败"时，自动保存前后 N 帧截图
- [ ] 记录失败时的上下文信息（sidecar json）：
  - pipeline 节点名称
  - 事件类型（捉虫/浇水/修水车）
  - 当时的阈值设置
  - ROI 区域
  - 模型版本
  - 失败类型（漏检/误检/超时）
- [ ] 标注优先级：先标注这些"难题"样本，再标注随机样本

**落地点**：
可以在 `farm_event.py` 的失败处理分支中调用 `ScreenshotCollector`，或者新增一个 `FailureCaseCollector`。

#### 2.1.2 按"域变化"分层采集

**为什么要做**：
不同的模拟器、不同的画质设置、不同的游戏场景，图片看起来会有差别（这叫 Domain Shift）。模型在 A 模拟器上训练，可能在 B 模拟器上就不太准了。

给每张图片记录来源信息，以后出问题时能快速定位"是哪种情况下模型不行"。

**具体实现**：

- [ ] 给每张图片记录元信息（sidecar json）：

  ```json
  {
    "emulator": "MuMu",
    "resolution": "1280x720",
    "scale": 1.0,
    "scene": "farm",
    "is_motion_blur": false,
    "is_occluded": true,
    "occlusion_type": "plot_ui",
    "timestamp": "2026-01-14T10:30:00"
  }
  ```

- [ ] 建立分桶评估机制：
  - 按模拟器类型分桶
  - 按遮挡程度分桶
  - 按场景类型分桶
- [ ] 便于快速定位"哪种情况下模型不行"

#### 2.1.3 专门补"遮挡"数据（最紧迫）

**为什么这是最紧迫的**：
你当前最大的稳定性问题就是遮挡。对目标检测模型来说，遮挡相当于"关键特征没了"，它就可能直接检测不到，或者框的位置飘。

**具体实现**：

- [ ] **刻意收集并标注两类图**：
  - 角色被坑位信息 UI 挡住的各种程度（30%/50%/70%）
  - 角色半身被灌木挡住、只露头/只露脚
- [ ] **模拟遮挡的数据增强**（比泛用增强更对症）：
  - 随机贴一块半透明 UI 形状在角色附近
  - 模拟坑位弹出框的样式和位置
  - 随机贴灌木形状的遮挡物
- [ ] **标注规则**：
  - 即使只露出一部分，也要标注完整的预期框
  - 记录遮挡程度作为 metadata

#### 2.1.4 标注质量控制

**为什么要做**：
特别是"虫子"这种小目标，框画得偏一点点，对模型影响很大。

**具体实现**：

- [ ] **抽样复核已标注数据**：
  - 每 100 张抽 10 张复核
  - 检查框是否画歪、是否漏标、是否错标
- [ ] **统计分析**：
  - 框尺寸分布（是否有极小框、异常大框）
  - 越界框占比
  - 各类别数量是否均衡
- [ ] **定好一致的标注规则**：
  - "人/虫重叠时"：都标出来？还是只标能看到的？
  - "遮挡时"：标完整框？还是只标可见部分？
  - **建议**：统一标完整框（模型应该学会"补全"）

#### 2.1.5 贴近部署的数据增强

**为什么要这样做**：
数据增强应该模拟"游戏截图会遇到的变化"，而不是模拟"拍照片"那种变化。

**具体实现**：

- [ ] **JPEG 压缩模拟**：

  ```python
  # 模拟截图压缩
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(70, 95)]
  _, encoded = cv2.imencode('.jpg', image, encode_param)
  image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
  ```

- [ ] **downscale → upscale**（模拟低分辨率截图）：

  ```python
  scale = random.uniform(0.7, 0.9)
  small = cv2.resize(image, None, fx=scale, fy=scale)
  image = cv2.resize(small, (orig_w, orig_h))
  ```

- [ ] **Gaussian blur / 运动模糊**：
  - 模拟运动时的画面模糊
  - 模拟截图时机不对导致的模糊
- [ ] **gamma / 亮度波动**：
  - 模拟不同时间段的光效变化
  - 模拟不同模拟器的渲染差异
- [ ] **轻微 UI 透明遮罩**：
  - 模拟游戏内各种 UI 元素的干扰

**不要做的增强**：

- 大角度旋转（游戏里画面不会旋转 90 度）
- 大幅度裁剪（会丢失上下文）
- 颜色通道交换（游戏画面不会这样）

---

### 2.2 模型/训练层面

#### 2.2.1 关注"部署指标"而非只看 mAP

**为什么 mAP 不够**：
mAP（mean Average Precision，平均精度均值）是目标检测的标准评分，但 mAP 高不代表"用起来稳"。

**你应该额外跟踪的指标**：

- [ ] **每帧误检率（FP/frame）**：
  - 定义：平均每帧有多少个"不该有的框"
  - 重要性：误检会导致脚本点错地方
- [ ] **漏检连续帧长度分布**：
  - 定义：连续多少帧检测不到目标
  - 重要性：连续丢失 >3 帧会触发你的 `lost_count` 机制
  - 统计：P(连续丢失 >N 帧) 的分布
- [ ] **首次检测延迟**：
  - 定义：目标出现后多少帧才首次检测到
  - 重要性：影响响应速度

**这些指标比 mAP 更能反映"实际使用稳不稳"**。

#### 2.2.2 输入分辨率/ROI 策略（对小目标关键）

**问题分析**：
你的游戏截图是 1280×720，但 YOLO 模型通常会把图片缩小到 640×640 来处理。虫子本来就小，缩小后就更小了，模型可能"看不清"。

```python
# test_yolo_onnx.py 中的配置
YOLO_INPUT_SIZE = 640  # YOLOv8 输入尺寸
```

**解决方案**：

- [ ] **方案 A：提高输入尺寸**
  - 把输入尺寸调大（如 960 或 1280）
  - 代价：速度变慢（推理时间大约与面积成正比）
  - 适合：对速度要求不高、但需要稳定性的场景

- [ ] **方案 B：按任务/场景动态 ROI**
  - 只截取"虫子/角色常出现的区域"
  - 相当于"放大"了那块区域
  - 你已经在识别配置里传了 `roi` 参数，可以进一步利用：

    ```python
    # farm_event.py 中已有的 ROI 使用
    "roi": [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT],  # 可以改成更小的区域
    ```

  - 适合：目标位置相对固定的场景

**建议**：先尝试方案 B（改动小、见效快），效果不够再考虑方案 A。

#### 2.2.3 按类别独立阈值

**问题分析**：
你现在是 `YOLO_THRESHOLD=0.3` 一刀切，但虫子和角色的检测难度不一样：

- 角色大、特征明显，置信度分布可能集中在 0.6-0.9
- 虫子小、特征少，置信度分布可能集中在 0.3-0.6

用同一个阈值会导致：

- 角色阈值偏低 → 可能增加误检
- 虫子阈值偏高 → 可能增加漏检

**解决方案**：

- [ ] **按类别设置不同阈值**：

  ```python
  YOLO_THRESHOLD_GIRL = 0.4  # 角色可以用更高阈值
  YOLO_THRESHOLD_BUGS = 0.25  # 虫子需要更低阈值才不漏
  ```

- [ ] **或者做简单的置信度校准**：
  - 离线统计各类别的置信度分布
  - 找到各类别的最佳阈值点（使 F1 最大）
  - 高级做法：温度缩放（Temperature Scaling）或 Platt Scaling

---

### 2.3 在线推理/后处理（稳定性来自"时序一致性"）

> **核心观点**：即使单帧检测更准，在线控制仍会抖。建议把"检测→动作"之间加一层轻量时序模块。

#### 2.3.1 目标跟踪/平滑（强烈推荐）

**问题分析**：
模型每一帧单独检测，结果可能"跳来跳去"。比如：

- 第 1 帧：角色在 (100, 200)
- 第 2 帧：角色在 (150, 180)  ← 跳了 50px
- 第 3 帧：角色在 (90, 210)   ← 又跳回去了

其实角色可能根本没动那么多，是检测有波动。

**解决方案**：

- [ ] **对 `girl`（角色）：用滑动平均（EMA）平滑中心点**

  ```python
  # EMA 滑动平均示例
  class PositionSmoother:
      def __init__(self, alpha=0.3):
          self.alpha = alpha  # 平滑系数，越小越平滑
          self.last_pos = None
      
      def smooth(self, new_pos):
          if new_pos is None:
              return self.last_pos
          if self.last_pos is None:
              self.last_pos = new_pos
              return new_pos
          # 加权平均：新位置占 alpha，旧位置占 (1-alpha)
          smoothed = (
              self.alpha * new_pos[0] + (1 - self.alpha) * self.last_pos[0],
              self.alpha * new_pos[1] + (1 - self.alpha) * self.last_pos[1]
          )
          self.last_pos = smoothed
          return smoothed
  ```

- [ ] **对 `bugs`（虫子）：用"最近邻关联 + 轨迹一致性"筛掉跳变误检**
  - 如果新检测到的虫子位置和上一帧差太远（比如 >100px），可能是误检
  - 可以用简单的距离阈值过滤

- [ ] **可选：引入轻量 tracking-by-detection（如 ByteTrack）**
  - ByteTrack 是一个很轻量的跟踪算法
  - 核心思想：把低置信度检测也纳入跟踪，减少漏检
  - 适合：需要同时跟踪多个虫子的场景

#### 2.3.2 迟滞（hysteresis）决策

**问题分析**：
模型偶尔会"闪一下"——某一帧误检到一个不存在的东西，下一帧又没了；或者某一帧漏检，下一帧又检测到了。

如果你的逻辑对每一帧都立刻响应，就会"来回跳"。

**解决方案**：

- [ ] **"出现 ≥2 帧才确认存在"**：

  ```python
  class DetectionConfirmer:
      def __init__(self, appear_threshold=2, disappear_threshold=3):
          self.appear_count = 0
          self.disappear_count = 0
          self.confirmed = False
      
      def update(self, detected):
          if detected:
              self.appear_count += 1
              self.disappear_count = 0
              if self.appear_count >= self.appear_threshold:
                  self.confirmed = True
          else:
              self.disappear_count += 1
              self.appear_count = 0
              if self.disappear_count >= self.disappear_threshold:
                  self.confirmed = False
          return self.confirmed
  ```

- [ ] **"消失 ≥K 帧才确认真的没了"**：
  - 结合你已有的 `lost_count` 机制
  - 把丢失判断从"帧级"升级为"轨迹级"

#### 2.3.3 丢失补救策略优化

**当前问题**：

```python
# farm_event.py 中的当前逻辑
if current_pos is None:
    lost_count += 1
    _joystick_nudge_up(context, times=2, duration_ms=500, wait_s=0.2)  # 一丢就往上走
```

这种"一丢就乱动"的策略属于碰运气，有时有效有时反而更糟。

**改进方案**：

- [ ] **分阶段补救策略**：

  ```python
  if current_pos is None:
      lost_count += 1
      
      if lost_count <= 2:
          # 阶段1：短暂丢失，可能只是遮挡
          # 不改方向，用上一次的位置继续（或者干脆停一下等遮挡消失）
          current_pos = last_known_pos
          time.sleep(0.1)  # 等一下再截图
          
      elif lost_count <= 5:
          # 阶段2：持续丢失，尝试轻微调整
          # 按上一次的移动方向继续一点点
          _joystick_nudge(context, direction=last_direction, strength=0.3)
          
      else:
          # 阶段3：长时间丢失，执行"拉回视野"动作
          _joystick_nudge_up(context, times=2, duration_ms=500, wait_s=0.2)
  ```

- [ ] **用"最后一次看到的位置"撑过短暂遮挡**：
  - 角色被 UI 挡住时，不代表它瞬间消失了
  - 在"丢失的短时间内"继续用上一次位置来做移动决策

#### 2.3.4 用"脚底点"而非"盒子中心"

**为什么**：
UI 往往遮挡上半身，但脚部/下缘更可能还可见。用盒子底边中心作为位置参考更稳定。

**代码已支持，但可能未全面启用**：

```python
# farm_event.py 第 1144-1149 行
box, center_pos = self._detect_girl_box_and_center(context, current_image)
if box is not None and use_feet:
    # feet 点：box 底部中心
    current_pos = (int(box[0] + box[2] // 2), int(box[1] + box[3]))
else:
    current_pos = center_pos
```

- [ ] **确保 `use_feet=True` 被正确传递和使用**
- [ ] **在所有检测角色位置的地方统一使用脚底点**

#### 2.3.5 减少重复识别调用

**当前问题**：

```python
# farm_event.py 中的 _detect_all_objects 方法
for label_idx, label_name in enumerate(YOLO_LABELS):
    reco_result = context.run_recognition(...)  # 每个类别都调用一次
```

每个类别各跑一次 `run_recognition`，在实时循环里会放大延迟/抖动来源。

**改进方案**：

- [ ] **单次推理取全量检测结果，再按类拆分**：
  - 修改识别配置，不指定 `expected`，让模型返回所有检测到的目标
  - 在 Python 层按 `cls_index` 或 `label` 分类
  - 这样只需要一次推理调用

---

### 2.4 移动控制逻辑改进

#### 2.4.1 摇杆力度随距离变化

**当前问题**：

```python
# farm_event.py 第 249-250 行
norm_dx = dx / dist * JOYSTICK_RADIUS  # 永远推到最大半径
norm_dy = dy / dist * JOYSTICK_RADIUS
```

无论离目标多远，都用最大力度推摇杆。

**改进方案**：

- [ ] **力度随距离线性/非线性缩放**：

  ```python
  def _calculate_joystick_direction_v2(current_pos, target_pos):
      dx = target_pos[0] - current_pos[0]
      dy = target_pos[1] - current_pos[1]
      dist = math.sqrt(dx * dx + dy * dy)
      
      if dist < 1:
          return JOYSTICK_CENTER
      
      # 根据距离计算力度比例
      if dist > 150:
          strength = 1.0      # 远距离：全力
      elif dist > 80:
          strength = 0.6      # 中距离：60% 力度
      elif dist > 40:
          strength = 0.35     # 近距离：35% 力度
      else:
          strength = 0.15     # 微调：15% 力度
      
      # 应用力度
      effective_radius = JOYSTICK_RADIUS * strength
      norm_dx = dx / dist * effective_radius
      norm_dy = dy / dist * effective_radius
      
      end_x = int(JOYSTICK_CENTER[0] + norm_dx)
      end_y = int(JOYSTICK_CENTER[1] + norm_dy)
      
      return (end_x, end_y)
  ```

#### 2.4.2 两段式移动：粗定位 → 精定位

**核心思想**：

- **粗定位阶段**：快速接近到目标附近（如 80px 内）
  - 大力度摇杆
  - 较长的滑动时间
  - 较短的检测间隔
- **精定位阶段**：进入目标附近后切换策略
  - 小力度摇杆（15-30%）
  - 较短的滑动时间
  - 每次滑动后多等一点让角色停稳
  - 更频繁地检查"是否到达"

**实现思路**：

- [ ] **定义两个阶段的参数**：

  ```python
  # 粗定位参数
  COARSE_THRESHOLD = 80      # 进入精定位的距离阈值
  COARSE_STRENGTH = 1.0      # 粗定位力度
  COARSE_DURATION = 500      # 粗定位滑动时长（ms）
  COARSE_INTERVAL = 0.1      # 粗定位检测间隔（s）
  
  # 精定位参数
  FINE_THRESHOLD = 26        # 到达目标的距离阈值（现有的 MOVE_TOLERANCE）
  FINE_STRENGTH = 0.25       # 精定位力度
  FINE_DURATION = 200        # 精定位滑动时长（ms）
  FINE_INTERVAL = 0.15       # 精定位检测间隔（s）
  FINE_SETTLE_TIME = 0.2     # 精定位后等待角色停稳的时间（s）
  ```

#### 2.4.3 防抖规则

**问题分析**：
距离进入阈值后立刻反向拉回，会导致来回振荡。

**解决方案**：

- [ ] **进入近距离后的防抖逻辑**：

  ```python
  if dist <= FINE_THRESHOLD * 1.5:  # 进入"接近到达"区域
      # 1. 先停住，不要立刻反向
      time.sleep(0.3)
      
      # 2. 立刻检查"交互按钮/坑位 UI"是否出现
      if _check_interaction_ui(context):
          # 出现了！直接进入交互流程，不再微调位置
          return True
      
      # 3. 如果没出现，再重新检测位置和距离
      # 4. 只有确认还没到才继续微调
  ```

- [ ] **添加"振荡检测"**：
  - 记录最近 N 次的移动方向
  - 如果方向频繁反转（比如连续 3 次方向相反），说明在振荡
  - 触发振荡时：停止移动，等待一段时间，或直接尝试交互

#### 2.4.4 靠 UI 状态闭环

**核心思想**：
坑位是固定的 16 个中心点（`FARM_PLOT_CENTERS`）。在坑位附近，最可靠的往往不是角色框多精，而是"当前坑位是否高亮/是否弹出信息"、"浇水/修理/捕捉按钮是否出现"。

**最后 1 米靠 UI 状态闭环**：

- [ ] **把"是否出现交互按钮/坑位 UI"作为"到达"的判断条件**：

  ```python
  # 不只依赖距离判断
  if dist <= MOVE_TOLERANCE or _check_interaction_available(context):
      return True  # 到达！
  ```

- [ ] **用"坑位 UI 是否弹出"作为状态信号**：
  - 当靠近坑位弹出"含水量/成熟时间"时，虽然看不清角色，但可以判断：已经在坑位交互范围附近
  - 这时切换策略：少依赖角色检测，多依赖 UI 匹配

---

### 2.5 调试可视化改进

#### 2.5.1 当前问题详解

**瓶颈 1：ADB 截图慢**

- `post_screencap()` 是通过 ADB 从设备端抓取完整画面
- 这个过程涉及：设备端截图 → 图像编码 → ADB 传输 → 解码到内存
- 典型延迟：100-500ms（取决于设备和连接方式）
- 导致帧率上限：2-10fps

**瓶颈 2：显示链路重**

```python
# 当前流程（farm_event.py）
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       # ~5ms
Image.fromarray(image_rgb)                    # ~10ms
Image.new('RGBA', ...)                        # ~2ms
draw.rectangle/text/ellipse(...)              # ~5ms per shape
Image.alpha_composite(pil_image, overlay)     # ~20ms（半透明合成很慢）
ImageTk.PhotoImage(pil_image)                 # ~15ms
canvas.itemconfig(...)                        # ~5ms
# 总计：~60ms+ 每帧显示
```

**为什么示例 YOLO 项目流畅**：

- 画面来源：本地摄像头/视频文件（读取 <10ms）
- 显示方式：OpenCV `cv2.imshow`（直接显示，无转换）
- 绘制方式：`cv2.rectangle/putText`（直接在 numpy 上画，很快）

#### 2.5.2 方案 A：scrcpy 转播 + 透明叠加层（推荐）

**核心思想**：
把"显示画面"和"识别/控制"分离：

- 画面由 scrcpy 负责（高帧率转播，通常 30-60fps）
- 你的脚本只做一个透明置顶窗口（overlay），在上面画框画字
- 即使识别只有 2-5fps，用户看到的画面也能是 60fps

**scrcpy 原理**：

1. **设备端**：
   - 每次启动 scrcpy 时，通过 ADB 往设备里推送一个小的 `scrcpy-server.jar`
   - 设备端用 Android 自带的"屏幕录制接口"抓画面
   - 用硬件编码器（MediaCodec）把画面编码成 H.264/H.265 视频流
   - 通过 ADB 转发的端口把视频流发给电脑
2. **电脑端**：
   - 接收视频流，解码后在窗口里显示
   - 可以把键鼠事件反向发回去（用于控制）

**为什么比 ADB 截图快**：

- 传的是"连续压缩视频流"，不是"每帧一张完整图片"
- 视频编码利用帧间相似性，数据量小很多
- 不需要每帧都完整传输

**实现要点**：

- [ ] **画面由 scrcpy 负责（高帧率转播）**
- [ ] **脚本做一个透明置顶窗口（overlay），画框画字**
- [ ] **需要解决：叠加框坐标与 scrcpy 窗口对齐**
  - 获取 scrcpy 窗口的位置和大小
  - 将检测坐标（1280×720）映射到窗口实际像素坐标
  - 处理窗口缩放/移动时的同步

**scrcpy 引入方式（考虑包体积）**：

| 方式 | 描述 | 包体积影响 | 用户体验 |
|------|------|-----------|----------|
| **方式 1（推荐）** | 作为外部工具调用 | **几乎为 0** | 需要用户自行安装 scrcpy |
| 方式 2 | 打包进发布包 | +10-30MB | 开箱即用 |
| 方式 3 | 运行时按需下载 | 几乎为 0 | 首次使用需下载 |

**方式 1 实现思路**：

```python
import subprocess
import shutil

def start_scrcpy():
    # 检查 scrcpy 是否存在
    scrcpy_path = shutil.which('scrcpy')
    if scrcpy_path is None:
        print("scrcpy 未安装，请从 https://github.com/Genymobile/scrcpy 下载")
        return None
    
    # 启动 scrcpy
    process = subprocess.Popen([
        scrcpy_path,
        '--no-audio',      # 不要音频
        '--max-fps', '30', # 帧率限制
        '--bit-rate', '4M', # 码率
        '--max-size', '1280', # 最大尺寸
        '--window-title', 'MAA Debug View',
    ])
    return process
```

**scrcpy 推荐参数**：

```bash
scrcpy --no-audio --max-fps 30 --bit-rate 4M --max-size 1280
```

#### 2.5.3 方案 B：窗口采集替代 ADB 截图

**核心思想**：
不从 ADB 抓图，而是像录屏一样直接抓模拟器窗口图像（Windows 图形层面抓）。通常比 ADB 截图快。

**优点**：

- 可以做到 20-60fps（取决于窗口采集实现和分辨率）
- 画面和用户看到的一致

**缺点**：

- [ ] **会采集到模拟器半透明按钮/自定义贴图**：
  - 与 ADB 截图内容不一致
  - 可能影响识别准确性
- [ ] **坐标对齐困难**：
  - 窗口可能有标题栏、边框、工具栏
  - 画面可能被缩放
  - 需要处理各种模拟器的差异
- [ ] **某些模拟器用特殊渲染会抓到黑屏**：
  - 使用 OpenGL/DirectX 渲染的窗口
  - 需要用更底层的采集方式（DXGI、D3D）

**不推荐用于识别，但可以用于纯显示**。

#### 2.5.4 方案 C：OpenCV 轻量绘制（最小改动）

**核心思想**：
保留 ADB 截图，但把显示改成最轻量的 OpenCV 绘制。

**实现要点**：

- [ ] **保留 ADB 截图**（瓶颈还在，但至少显示不会更慢）
- [ ] **直接在 numpy 上画框**：

  ```python
  # 比 PIL+alpha_composite 快很多
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.putText(image, f"girl {conf:.2f}", (x, y-10), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
  ```

- [ ] **用 `cv2.imshow` 显示**：

  ```python
  cv2.imshow('Debug', image)
  cv2.waitKey(1)  # 非阻塞
  ```

**预期效果**：

- 帧率上限仍受 ADB 截图限制（2-5fps）
- 但显示链路开销会从 ~60ms 降到 ~10ms
- 画面"没那么卡"（虽然还是低帧率）

**适合场景**：想快速改善、不想引入 scrcpy 依赖。

#### 2.5.5 调试指标显示

**应该显示的指标**：

- [ ] **展示帧率**：调试窗口刷新频率

  ```python
  display_fps = frame_count / (current_time - last_fps_time)
  ```

- [ ] **识别帧率**：实际跑检测的频率

  ```python
  detection_fps = detection_count / (current_time - last_fps_time)
  ```

- [ ] **识别延迟**：截图 → 得到识别结果的耗时

  ```python
  t1 = time.time()
  controller.post_screencap().wait()
  reco_result = context.run_recognition(...)
  detection_latency = time.time() - t1
  ```

- [ ] **控制反馈延迟**：发出滑动命令 → 画面变化的耗时
  - 这个比较难精确测量，可以用"发出命令后多久角色位置变化"来估算
- [ ] **在线代理指标**（不是真正的准确率，但能反映稳定性）：
  - 连续丢失次数（当前 `lost_count`）
  - 误触发次数（点到错误位置的次数）
  - 到达目标耗时
  - 是否超时

**注意**：真正的"准确率"需要有标准答案（ground truth），在线无法自动计算。上面的指标是"代理指标"，用于反映系统运行状态。

---

### 2.6 工程规范

#### 2.6.1 离线评测基准

**为什么要做**：
每次改进模型/参数后，需要有客观的方式比较"到底有没有变好"。光看"跑起来感觉"不靠谱。

**具体实现**：

- [ ] **从采集的图片中分出固定的 val/test 集**：
  - 这部分图片永远不用来训练，只用来测试
  - 按域分桶（不同模拟器、不同场景、不同遮挡程度）
  
- [ ] **写评测脚本，输出以下指标**：
  - per-class PR 曲线（精确率-召回率曲线）
  - per-class AP（平均精确率）
  - FP/frame（每帧误检数）
  - 漏检连续帧统计
  - 在业务 ROI 内的指标（只统计游戏画面中实际用到的区域）
  
- [ ] **每次换模型/超参/增强，都在固定测试集上跑一遍**：

  ```bash
  python evaluate.py --model v1.onnx --test-set test_data/
  python evaluate.py --model v2.onnx --test-set test_data/
  # 对比两个版本的指标
  ```

#### 2.6.2 实验与数据版本管理

**为什么要做**：

- 便于复现结果（自己以后想回到某个版本）
- 便于与他人交流（教授/同学能复现你的实验）
- 在跟教授沟通时显得非常"研究化"

**具体实现**：

- [ ] **数据版本管理**：
  - 使用 DVC（Data Version Control）管理数据集
  - 或者至少用 manifest + hash：

    ```json
    {
      "dataset_version": "v3",
      "train_images": 1234,
      "val_images": 256,
      "test_images": 128,
      "md5": "abc123...",
      "created_at": "2026-01-14"
    }
    ```

- [ ] **训练配置记录**：
  - 记录每次训练的完整配置：

    ```yaml
    experiment: v3_occlusion_aug
    date: 2026-01-14
    seed: 42
    model: yolov8n
    input_size: 640
    epochs: 100
    batch_size: 16
    augmentation:
      - mosaic: 0.5
      - mixup: 0.1
      - jpeg_quality: [70, 95]
      - blur: 0.1
    optimizer:
      type: AdamW
      lr: 0.001
      weight_decay: 0.0005
    scheduler:
      type: cosine
      warmup_epochs: 3
    export:
      format: onnx
      opset: 12
      simplify: true
    ```

- [ ] **结果记录**：
  - 训练曲线（loss、mAP）
  - 最终指标
  - 导出的模型文件及其 hash

---

## 三、作为研究敲门砖的准备

### 3.1 项目定位

> **"在受限场景（游戏自动化）下的小目标检测与稳定性优化"**

这是一个天然的研究场景，包含多个有价值的研究问题：

- 小目标检测（虫子很小）
- 遮挡处理（UI 遮挡、场景遮挡）
- Domain Shift（不同模拟器/渲染差异）
- 时序一致性（在线控制需要稳定）
- 闭环学习（可以自动收集失败样本）

### 3.2 准备材料

#### 3.2.1 一页技术摘要（A4 一页）

**应该包含的内容**：

- [ ] **问题定义**：
  - 场景：游戏自动化中的目标检测与控制
  - 目标：自动识别游戏里的角色（girl）和小虫子（bugs），控制角色去抓虫子
  - 挑战：小目标、遮挡、实时性、稳定性
  
- [ ] **方法**：
  - 模型：YOLOv8（自己训练）
  - 数据：自己收集的数据集（XX 张图片，XX 个标注）
  - 部署：ONNX Runtime，集成到 MaaFramework
  - 工程优化：数据增强、阈值调整、时序平滑等
  
- [ ] **效果**：
  - 定量指标：mAP、FP/frame、漏检率等
  - 定性描述：能稳定运行多久、主要失败场景
  
- [ ] **附图**：
  - 检测效果图（成功案例）
  - 失败案例分析（遮挡、误检）
  - 系统架构图

#### 3.2.2 可复现的 benchmark

- [ ] **固定 test 集 + 一键评测脚本**：

  ```bash
  python evaluate.py --model models/farming.onnx --test-set data/test/
  ```
  
- [ ] **2~3 组消融实验**（展示各个改进的效果）：

  | 实验 | 配置 | mAP | FP/frame | 连续漏检率 |
  |------|------|-----|----------|-----------|
  | baseline | 默认参数 | 0.75 | 0.3 | 15% |
  | +ROI优化 | 缩小ROI | 0.78 | 0.2 | 12% |
  | +阈值分离 | 按类别阈值 | 0.77 | 0.15 | 10% |
  | +时序平滑 | EMA平滑 | 0.76 | 0.1 | 5% |
  
- [ ] **录制演示视频**：
  - `test_yolo_onnx.py` 可用于可视化
  - 展示正常运行、遮挡恢复、失败案例

#### 3.2.3 下一步研究计划（教授最看重）

**主线 A：Active Learning / Hard-case Mining**

- **研究问题**：如何自动识别"模型容易出错的样本"，优先标注这些样本来改进模型？
- **为什么适合这个项目**：
  - 你有完整的闭环：模型部署 → 运行 → 失败 → 收集失败样本 → 重新训练
  - 可以自动判断"失败"（识别不到、超时、走错位置）
  - 天然的 human-in-the-loop 场景
- **可能的研究方向**：
  - 失败样本的自动筛选策略
  - 主动学习的样本选择算法
  - 增量学习（新样本加入后如何高效更新模型）

**主线 B：Domain Adaptation / Test-time Adaptation**

- **研究问题**：模型在 A 模拟器上训练，如何在 B 模拟器上也能稳定工作？
- **为什么适合这个项目**：
  - 用户使用的模拟器多样（MuMu、雷电、夜神等）
  - 不同模拟器的渲染、压缩、颜色可能有差异
  - 不可能为每个模拟器都收集大量数据
- **可能的研究方向**：
  - 无监督域适应（不需要目标域标注）
  - 测试时适应（Test-time Adaptation，TTA）
  - 风格迁移/归一化

**主线 C：Temporal Consistency / Tracking-aware Detection**

- **研究问题**：如何利用"时序信息"（连续多帧）来提高检测稳定性？
- **为什么适合这个项目**：
  - 直接对应当前"遮挡丢失"问题
  - 单帧检测不稳定，但连续多帧应该是一致的
  - 可以利用运动模型预测目标位置
- **可能的研究方向**：
  - 检测+跟踪联合优化
  - 时序一致性约束
  - 遮挡状态建模与恢复

---

## 四、术语参考

| 术语 | 英文 | 通俗解释 |
|------|------|----------|
| mAP | mean Average Precision | 目标检测的"综合评分"。把每个类别的 AP（平均精确率）求平均。越高越好，满分是 1.0 |
| 阈值 | threshold | "门槛"。模型给出的置信度要超过这个值才算"检测到了"。比如阈值 0.3 表示"模型有 30% 以上把握就算检测到" |
| 置信度 | confidence | 模型对"这里有个目标"的把握程度。0~1 之间，越高表示越确定 |
| NMS | Non-Maximum Suppression | 去掉重复框的算法。同一个目标可能检测出多个重叠的框，NMS 只保留置信度最高的那个 |
| ROI | Region of Interest | "感兴趣区域"。只在这个区域内检测，忽略其他地方。可以提高速度和准确性 |
| 数据增强 | Data Augmentation | 人为制造图片变化（亮度、模糊、翻转等），让模型"见多识广"。相当于用一张图生成多张变体 |
| Domain Shift | - | 训练数据和实际使用时的数据"长得不一样"。比如训练用 A 模拟器的图，但用户用 B 模拟器，画面风格有差异 |
| 漏检 | False Negative (FN) | 明明有目标，模型没检测到。"漏"掉了 |
| 误检 | False Positive (FP) | 明明没有目标，模型说有。"误"判了 |
| 校准 | Calibration | 让模型的"置信度"更准确反映"真实概率"。比如模型说 80% 置信度，实际上真的有 80% 概率是对的 |
| EMA | Exponential Moving Average | 指数滑动平均。用"最近几帧的加权平均"来平滑结果，越近的帧权重越高 |
| Tracking | 跟踪 | 不只是每帧单独检测，而是跟踪"同一个目标"在连续多帧里的位置变化 |
| Kalman 滤波 | Kalman Filter | 一种"预测+修正"的平滑算法。先预测目标下一帧会在哪，再用实际检测结果修正。常用于跟踪 |
| 迟滞 | Hysteresis | "状态切换要有惯性"。比如"出现 2 帧才确认存在，消失 3 帧才确认消失"，避免来回跳变 |
| scrcpy | Screen Copy | 开源的 Android 投屏工具。通过视频流的方式把手机/模拟器画面实时显示到电脑上，比截图快很多 |
| ONNX | Open Neural Network Exchange | 一种通用的神经网络模型格式。训练好的模型导出成 ONNX，可以在各种平台上运行 |
| Active Learning | 主动学习 | 一种机器学习策略：让模型主动"挑选"最有价值的样本来标注，而不是随机标注 |
| Test-time Adaptation | 测试时适应 | 在测试/部署时，让模型根据当前输入数据进行微调，适应新的数据分布 |

---

## 五、优先级建议

### 立刻可做的 Top 3

1. **🔴 解决卡死**（最紧迫）
   - 加"短时间丢失用上次位置顶住 + 逐步升级补救"的策略
   - 收集遮挡场景的失败截图，补充训练数据
   - 改用"脚底点"作为位置参考

2. **🟠 解决抖动**（影响体验）
   - 把摇杆力度改成"随距离缩放"
   - 实现"两段式粗到精"移动
   - 加"防抖规则"和"UI 状态闭环"

3. **🟡 解决调试体验**（便于开发）
   - 短期：把显示改成 OpenCV 轻量绘制
   - 中期：引入 scrcpy 转播方案

### 中期改进

1. **数据层面**
   - 实现失败样本自动收集
   - 专门补充遮挡数据
   - 建立分桶评测体系

2. **模型层面**
   - 实现按类别独立阈值
   - 尝试更大的输入尺寸或动态 ROI

### 长期规划

1. **工程规范**
   - 建立离线评测基准
   - 实验与数据版本管理

2. **研究准备**
   - 整理一页技术摘要
   - 准备可复现的 benchmark
   - 确定下一步研究方向

---

*文档生成时间：2026-01-14*
*文档版本：v2 - 详细版*
