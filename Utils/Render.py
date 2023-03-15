import random
import sys
import math
import time
import pygame
import collections
import multiprocessing
import numpy as np


# 绘画类
class Render(multiprocessing.Process):
    def __init__(self, map_width, map_height, screen_width, screen_height, queue):
        super(Render, self).__init__()
        # 定义地图与屏幕尺寸, 转换比例
        self.map_width = map_width
        self.map_height = map_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.map_to_screen_width = self.screen_width / self.map_width
        self.map_to_screen_height = self.screen_height / self.map_height
        # 定义进程通信通道
        self.queue = queue

    def run(self):
        super(Render, self).run()
        # 创建窗口
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            self.screen.fill((255, 255, 255))
            self.surface_update()
            pygame.display.flip()

    def surface_update(self):
        system_device_list = self.queue.get(block=True)
        for device_list in system_device_list:
            for _, device in device_list.items():
                color = [0, 0 ,0]
                if device.type == 0:
                    surface = pygame.draw.circle(self.screen, color, (device.xpos, device.ypos), 10)
                else:
                    pygame.draw.rect(self.screen, color, (device.xpos - 10 * self.map_to_screen_width, device.ypos * self.map_to_screen_height - 10, 20, 20), 0)
                for connect_list in device.connect_list:
                    for _, connect in connect_list.items():
                        surface = pygame.draw.line(self.screen, color, (device.xpos, device.ypos), (connect[0].xpos, connect[0].ypos))