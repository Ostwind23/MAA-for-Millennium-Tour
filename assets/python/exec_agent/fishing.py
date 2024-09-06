import time
import cv2
import numpy as np
import pyautogui
from maa.context import SyncContext
from maa.custom_action import CustomAction

LIGHT_SPEED = 5


# 等待并点击钓鱼按钮
class Autofishing(CustomAction):
    def run(self, context, task_name, custom_param, box, rec_detail) -> bool:
        """
        :param context: 运行上下文，提供 具体 方法。
        :param task_name: 任务名称。
        :param custom_param: 自定义参数
        :param box: 识别到的区域。
        :param rec_detail: 识别的详细信息。{
        :return: 滑动是否成功。
        """
        image = context.screencap()
        step_start_fishing = context.run_task("点击钓鱼按钮",
        {"点击钓鱼按钮":
            {
            "recognition": "TemplateMatch",
            "template": "fish/fishing_button.png",
            "roi": [0,0,0,0],
            "action": "Click",
            }
        }
        )
        if step_start_fishing:
            while not self.ifgetfish(context):
                self.wait_for_fish(context)
                self.fishing(context)
                self.detect_and_click_circles(context)
            return True
        return False

    def stop(self) -> None:
        pass

    # 检测是否钓上鱼
    def ifgetfish(self, context:SyncContext) -> bool:
        image = context.screencap()
        res = context.run_recognition(image, "是否钓上鱼",
        {
            "是否钓上鱼":{
            "recognition": "TemplateMatch",
            "template": "fishing/getfish.png",
            "roi":[
                0,
                0,
                0,
                0
            ]
            }
        }
        )
        return res["return"]

    # 等待鱼咬饵并点击钓鱼按钮
    def wait_for_fish(self, context:SyncContext):
        while True:
            res = context.run_task("是否钓上鱼",
            {
                "是否钓上鱼":{
                "inverse": True,
                "recognition": "TemplateMatch",
                "template": "fishing/fishing button.png",
                "roi":[
                    0,
                    0,
                    0,
                    0
                ],
                "action": "Click",
                "post_delay": 100
                },
                "二号任务":{
                    "template": "path/to/new/template.png"
                }
            }
            )
            if res:
                break
            time.sleep(0.5)

    # def compare_histograms(self, imageA, imageB):
    #     histA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    #     histB = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    #     histA = cv2.normalize(histA, histA).flatten()
    #     histB = cv2.normalize(histB, histB).flatten()
    #     similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    #     return similarity


    # 控制摇杆
    def fishing(self, context:SyncContext):
        while True:
            # 截图并识别小三角箭头的位置
            image = context.screencap()
            arrow_position = context.run_recognition(image, "三角箭头位置",
                                                     {
                                                         "三角箭头位置": {
                                                             "recognition": "FeatureMatch",
                                                             "template": "arrow.png",
                                                             "detector": "AKAZE",
                                                             "roi": [
                                                                 0,
                                                                 0,
                                                                 0,
                                                                 0
                                                             ],
                                                         }
                                                     }
                                                     )
            if not arrow_position["return"]:
                break

            # 计算摇杆需要移动的角度
            arrow_x, arrow_y = arrow_position["box"][0] + 0.5*arrow_position["box"][2], arrow_position["box"][1] + 0.5*arrow_position["box"][3]
            center_x, center_y = self.get_center_of_fishing_button(context)
            angle = np.arctan2(arrow_y - center_y, arrow_x - center_x)

            # 移动摇杆
            pyautogui.mouseDown(arrow_x, arrow_y)

            time.sleep(0.1)
            if not arrow_position["return"]:
                break
        print("钓到鱼了！")

    # 获取钓鱼按钮中心位置的函数
    def get_center_of_fishing_button(self, context:SyncContext):
        image = context.screencap()
        button_position = context.run_recognition(image, "钓鱼按钮",
                                                       {
                                                           "钓鱼按钮": {
                                                               "recognition": "TemplateMatch",
                                                               "template": "fishing_button.png",
                                                               "roi": [
                                                                   0,
                                                                   0,
                                                                   0,
                                                                   0
                                                               ],
                                                               "action": "None",
                                                               "post_delay": 100
                                                           }
                                                       }
                                                       )
        if button_position["return"]:
            return button_position["box"][0] + 0.5*button_position["box"][2], button_position["box"][1] + 0.5*button_position["box"][3]
        return None, None

    # # 移动摇杆的函数
    # def move_joystick(self, arrow_x, arrow_y):
    #     joystick_radius = 50  # 假设摇杆的半径为50像素
    #     # target_x = center_x + np.cos(angle) * joystick_radius
    #     # target_y = center_y + np.sin(angle) * joystick_radius
    #     pyautogui.dragTo(arrow_x, arrow_y, duration=0.5)

    # 钓鱼QTE

    def detect_icons(self, context: SyncContext, threshold=0.8):
        # 读取屏幕截图和模板图像
        screenshot = context.screencap()
        template = cv2.imread("path/to/template_path")

        # 转换为灰度图像
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 模板匹配
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # 找到匹配位置
        loc = np.where(result >= threshold)

        # 存储图标的中心坐标和半径
        centers = []
        radius = template.shape[0] // 2

        # 遍历匹配结果
        for pt in zip(*loc[::-1]):
            center_x = pt[0] + radius
            center_y = pt[1] + radius
            centers.append((center_x, center_y))

        return centers, radius

    def detect_circle_radius(self, screenshot_gray, center, icon_radius):
        # 创建一个掩膜，只保留中心点附近的区域
        mask = np.zeros(screenshot_gray.shape, dtype=np.uint8)
        cv2.circle(mask, center, icon_radius + 50, 255, -1)  # 假设光圈半径不会超过图标半径+50像素

        # 使用掩膜提取中心点附近的区域
        masked_img = cv2.bitwise_and(screenshot_gray, screenshot_gray, mask=mask)

        # 检测圆形
        circles = cv2.HoughCircles(masked_img, cv2.HOUGH_GRADIENT, dp=1, minDist=icon_radius, param1=50, param2=30, minRadius=icon_radius, maxRadius=icon_radius + 50)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if (x, y) == center:
                    return r
        return None

    def detect_and_click_circles(self, context: SyncContext, light_speed=LIGHT_SPEED):
        # 检测图标
        centers, icon_radius = self.detect_icons(context)

        # 等待0.5秒
        time.sleep(0.5)

        # 读取新的屏幕截图
        screenshot = context.screencap()
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        for center in centers:
            # 检测光圈的当前半径
            current_radius = self.detect_circle_radius(screenshot_gray, center, icon_radius)
            if current_radius is None:
                print(f"No circle detected around icon at {center}")
                continue

            # 计算光圈收缩到图标半径需要的时间
            shrink_time = ((current_radius - icon_radius) / light_speed)*0.001

            # 等待光圈收缩到图标半径
            time.sleep(shrink_time)

            # 点击图标（这里用打印代替实际点击）
            pyautogui.click(center)
            print(f"Clicking at {center}")




