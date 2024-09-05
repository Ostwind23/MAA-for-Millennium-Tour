from typing import Tuple
import time
import cv2
import numpy as np
import pyautogui
from maa.context import SyncContext

# python -m pip install maafw
from maa.define import RectType
from maa.resource import Resource
from maa.controller import AdbController
from maa.instance import Instance
from maa.toolkit import Toolkit

from maa.custom_recognizer import CustomRecognizer
from maa.custom_action import CustomAction

import asyncio
import sys
import traceback
import os

os.system('chcp 65001')

start_time = time.time()
LIGHT_SPEED = 5


async def main():
    user_path = "./"
    Toolkit.init_option(user_path)

    resource = Resource()
    await resource.load("../resource/base")

    device_list = await Toolkit.adb_devices()
    if not device_list:
        print("未找到任何ADB设备")
        input("按任意键退出")
        exit()

    # for demo, we just use the first device
    print("当前ADB设备有：\n")
    print(device_list)
    devnum = input("请输入你的模拟器ADB设备的序号（从0号开始为第一个）")
    device = device_list[int(devnum)]
    controller = AdbController(
        adb_path=device.adb_path,
        address=device.address,
    )
    await controller.connect()

    maa_inst = Instance()
    maa_inst.bind(resource, controller)

    if not maa_inst.inited:
        print("MAA框架初始化失败,存在资源逻辑异常或环境缺失")
        input("按任意键退出")
        exit()

    maa_inst.register_recognizer("MyRec", my_rec)
    maa_inst.register_action("MyAct", my_act)
    maa_inst.register_action("Autofishing", autofishing)
    print("MAA框架初始化完成,准备开始执行自动钓鱼任务，请务必确保当前界面为钓台界面")

    await maa_inst.run_task("Fully_Automatic_Fishing")


class MyRecognizer(CustomRecognizer):
    def analyze(
        self, context, image, task_name, custom_param
    ) -> Tuple[bool, RectType, str]:
        return True, (0, 0, 100, 100), "Hello World!"


class MyAction(CustomAction):
    def run(self, context, task_name, custom_param, box, rec_detail) -> bool:
        return True

    def stop(self) -> None:
        pass


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
        cv2.imshow("ERROR", image)
        step_start_fishing = context.run_task("点击钓鱼按钮",
            {
                "点击钓鱼按钮": {
                    "recognition": "TemplateMatch",
                    "template": "fishing/control button.png",
                    "roi": [939,
                            377,
                            228,
                            230],
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
        flag,loc,detail = context.run_recognition(image, "是否钓上鱼",
        {
            "是否钓上鱼":{
            "recognition": "TemplateMatch",
            "template": "fishing/getfish.png",
            "roi":[
                939,
                377,
                228,
                230
            ]
            }
        }
        )
        return flag

    # 等待鱼咬饵并点击钓鱼按钮
    def wait_for_fish(self, context:SyncContext):
        while True:
            res = context.run_task("是否钓上鱼",
                                   {
                                       "是否钓上鱼": {
                                           "inverse": True,
                                           "recognition": "TemplateMatch",
                                           "template": "fishing/getfish.png",
                                           "roi": [
                                                939,
                                                377,
                                                228,
                                                230],
                                           "action": "Click",
                                           "post_delay": 100
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
            flag,arrow_position,detail = context.run_recognition(image, "三角箭头位置",
                                                     {
                                                         "三角箭头位置": {
                                                             "recognition": "FeatureMatch",
                                                             "template": "fishing/arrow.png",
                                                             "detector": "AKAZE",
                                                             "roi": [
                                                                 839,
                                                                 277,
                                                                 328,
                                                                 330
                                                             ],
                                                         }
                                                     }
                                                     )
            if not flag:
                break

            # 计算摇杆需要移动的角度
            arrow_x, arrow_y = arrow_position[0] + 0.5*arrow_position[2], arrow_position[1] + 0.5*arrow_position[3]
            center_x, center_y = self.get_center_of_fishing_button(context)
            # if center_x is not None and center_y is not None:
            #     angle = np.arctan2(arrow_y - center_y, arrow_x - center_x)
            # else:
            #     return

            # 移动摇杆
            pyautogui.mouseDown(arrow_x, arrow_y)

            time.sleep(0.1)
            if not flag:
                break


    # 获取钓鱼按钮中心位置的函数
    def get_center_of_fishing_button(self, context:SyncContext):
        image = context.screencap()
        flag,button_position,detail = context.run_recognition(image, "钓鱼按钮",
                                                       {
                                                           "钓鱼按钮": {
                                                               "recognition": "TemplateMatch",
                                                               "template": "fishing/control button.png",
                                                               "roi": [
                                                                   939,
                                                                   377,
                                                                   228,
                                                                   230
                                                               ]
                                                           }
                                                       }
                                                       )
        if flag:
            return button_position[0] + 0.5*button_position[2], button_position[1] + 0.5*button_position[3]
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
        image_path = '../resource/base/image/fishing/icon.png'
        template = cv2.imread(image_path)
        if template is not None:
            print("icon图片读取成功")
        else:
            print("icon图片读取失败")

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
        time.sleep(0.2)

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


my_rec = MyRecognizer()
my_act = MyAction()
autofishing = Autofishing()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        traceback.print_exc()
        input("按回车键退出...")