import math
import numpy as np

# 定义任务父类
class Base_Task():
    def __init__(self, *args, **kwargs):
        # 定义任务编号
        self.index = kwargs['index']

        # 定义本地设备、执行设备
        self.local_device = kwargs['local_device']
        self.execution_device = None
        self.vm_index = -1

        # 定义起始、最大运行、结束时间
        self.start_time = kwargs['start_time']
        self.max_run_time = kwargs['max_run_time']
        self.end_time = self.start_time + self.max_run_time - 1
        # 定义卸载、上传、计算、下载、结束时间
        self.wait_time = -1
        self.upload_time = -1
        self.compute_time = -1
        self.download_time = -1
        self.finish_time = -1

        # 定义任务数据量(bit bit cycle bit)
        self.upload_data = kwargs['upload_data']
        self.store_data = kwargs['store_data']
        self.compute_data = kwargs['compute_data']
        self.download_data = kwargs['download_data']

        # 定义任务状态
        # 状态(-3 运行失败) (-2 卸载失败) (-1 未开始) (0 上传阶段) (1 计算阶段) (2 下载阶段) (3 任务完成)
        self.state = -1

    # 任务本地执行, 只需要计算阶段
    # 输入: 当前时间, 计算数据量(bit)
    # 输出: (0 任务未完成, 1 任务完成), 浪费资源比例
    def local_run(self, now_time, compute_data):
        # 计算浪费的计算资源量(bit)
        waste_compute_data = max(compute_data - self.compute_data, 0)
        # 计算剩余的计算任务量(bit)
        self.compute_data = max(self.compute_data - compute_data, 0)
        if self.compute_data == 0:
            self.compute_finish(now_time)
            self.download_finish(now_time)
            return 1, waste_compute_data / compute_data
        return 0, waste_compute_data / compute_data

    # 任务远程执行, 需要上传、计算、下载三阶段
    # 输入: 当前时间, 上传数据量(bit), 计算数据量(bit), 下载数据量(bit)
    # 输出: (0 任务未完成, 1 任务完成), 浪费资源比例
    def mec_run(self, now_time, upload_data, compute_data, download_data):
        # 判断任务执行的阶段
        # 上传阶段
        if self.upload_data != 0:
            waste_upload_data = max(upload_data - self.upload_data, 0)
            self.upload_data = max(self.upload_data - upload_data, 0)
            if self.upload_data == 0:
                self.upload_finish(now_time)
            return 0, waste_upload_data / upload_data
        # 计算阶段
        elif self.compute_data != 0:
            waste_compute_data = max(compute_data - self.compute_data, 0)
            self.compute_data = max(self.compute_data - compute_data, 0)
            if self.compute_data == 0:
                self.compute_finish(now_time)
            return 0, waste_compute_data / compute_data
        # 下载阶段
        elif self.download_data != 0:
            waste_download_data = max(download_data - self.download_data, 0)
            self.download_data = max(self.download_data - download_data, 0)
            if self.download_data == 0:
                self.download_finish(now_time)
                return 1, waste_download_data / download_data
            else:
                return 0, waste_download_data / download_data

    # 任务卸载完成
    # 输入: 当前时间(时隙开头), 运行设备, 虚拟机编号
    # 输出: (0 卸载失败, 1 卸载完成)
    def offload_finish(self, now_time, device, vm_index):
        # 若不处于等待阶段
        if self.state != -1:
            return 0
        # 卸载完成, 进入上传阶段
        self.state = 0
        # 计算等待时间, 卸载发生在时隙开头, 无需-1
        self.wait_time = now_time - self.start_time
        # 设置执行的设备
        self.execution_device = device
        # 设置虚拟机编号
        self.vm_index = vm_index
        # 若本地执行, 则直接进入计算阶段
        if self.execution_device == self.local_device:
            self.upload_data = 0
            self.download_data = 0
            # 上传发生在时隙结束, 需要-1
            self.upload_finish(now_time - 1)
        return 1

    # 任务卸载失败
    # 输入: 当前时间(时隙开头)
    def offload_fail(self, now_time):
        self.state = -2
        self.finish_time = now_time

    # 任务上传完成
    # 输入: 当前时间(时隙末尾)
    def upload_finish(self, now_time):
        self.state = 1
        # 任务上传花费的时间 = 总的花费时间 - 任务等待花费的时间
        self.upload_time = (now_time + 1 - self.start_time) - self.wait_time

    # 任务计算完成
    # 输入: 当前时间(时隙末尾)
    def compute_finish(self, now_time):
        self.state = 2
        # 任务计算花费的时间 = 总的花费时间 - 任务等待花费的时间 - 任务上传花费的时间
        self.compute_time = (now_time + 1 - self.start_time) - self.wait_time - self.upload_time

    # 任务下载完成
    # 输入: 当前时间(时隙末尾)
    def download_finish(self, now_time):
        self.state = 3
        self.finish_time = now_time
        # 任务计算花费的时间 = 总的花费时间 - 任务等待花费的时间 - 任务上传花费的时间 - 任务计算花费的时间
        self.download_time = now_time + 1 - self.start_time - self.wait_time - self.upload_time - self.compute_time

    # 任务运行失败
    # 输入: 当前时间(时隙末尾)
    def run_fail(self, now_time):
        self.state = -3
        self.finish_time = now_time

    # 判断任务是否能保留到下个时隙
    # 输入: 当前时间(时隙开头)
    def live(self, now_time):
        return now_time <= self.end_time and now_time >= self.start_time