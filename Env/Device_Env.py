import time
import numpy as np
from Utils.Common import *


# 定义基类设备
class Base_Device():
    def __init__(self, *args, **kwargs):
        # 定义编号、类别
        self.index = kwargs['index']
        self.type = kwargs['type']

        # 定义任务等级
        self.level = kwargs['level']

        # 定义三维位置(m)、速度(m/s)、三维角度
        self.xpos = kwargs['xpos']
        self.ypos = kwargs['ypos']
        self.zpos = kwargs['zpos']
        self.velocity = kwargs['velocity']
        self.solid_angle_1 = kwargs['solid_angle_1']
        self.solid_angle_2 = kwargs['solid_angle_2']

        # 定义资源数量(Hz bit Hz Hz)
        self.upload_resources = kwargs['upload_resources']
        self.store_resources = kwargs['store_resources']
        self.compute_resources = kwargs['compute_resources']
        self.download_resources = kwargs['download_resources']

        # 定义发射、计算、待机功率(mw)
        self.transmission_power = kwargs['transmission_power']
        self.compute_power = kwargs['compute_power']
        self.standby_power = kwargs['standby_power']

        # 定义覆盖范围(m)
        self.coverage = kwargs['coverage']

        # 定义连接设备列表
        # 传入list为各类设备的链接数量上限
        self.max_connect_num_list = args
        self.connect_list = []
        self.access_list = []
        # 定义周围设备列表
        self.near_list = []
        for each in range(len(self.max_connect_num_list)):
            self.connect_list.append({})
            self.access_list.append({})
            self.near_list.append({})

        # 定义任务运行、等待队列
        self.max_task_load_num = kwargs['max_task_load_num']
        self.task_generation_num = kwargs['task_generation_num']
        # 任务等待队列: 由本设备产生、待卸载的任务队列
        self.task_wait_list = []
        # 任务运行队列: 本设备承载的运行中的任务
        self.task_run_list = np.zeros(shape=self.max_task_load_num, dtype=np.int8).tolist()

        # 定义设备能量开销列表(总、传输、计算、待机)
        self.device_energy_cost = np.zeros(shape=(4,))

        # 定义任务能量开销列表(总、传输、计算、待机)、任务服务时延列表(总、等待、上传、计算、下载)
        if self.task_generation_num != 0:
            self.task_energy_cost = np.zeros(shape=(self.task_generation_num, 4))
            self.task_service_delay = np.zeros(shape=(self.task_generation_num, 5))
        else:
            self.task_energy_cost = None
            self.task_service_delay = None

        # 定义任务完成、失败任务列表
        self.task_offload_num = 0
        self.task_finish_num = 0
        self.task_fail_num = 0

        # 定义资源分配状态空间
        self.ra_state_space = (self.max_task_load_num, 6)

    # 重置设备
    def reset(self):
        # 重置连接设备列表
        self.connect_list = []
        self.access_list = []
        # 重置周围设备列表
        self.near_list = []
        for each in range(len(self.max_connect_num_list)):
            self.connect_list.append({})
            self.access_list.append({})
            self.near_list.append({})
        # 重置任务等待、运行队列
        self.task_wait_list = []
        self.task_run_list = np.zeros(shape=self.max_task_load_num, dtype=np.int8).tolist()
        # 重置设备能量开销列表(总、传输、计算、 待机)
        self.device_energy_cost = np.zeros(shape=(4,))
        # 重置任务能量开销列表(总、传输、计算、待机)、任务服务时延列表(总、等待、上传、计算、下载)
        if self.task_generation_num != 0:
            self.task_energy_cost = np.zeros(shape=(self.task_generation_num, 4))
            self.task_service_delay = np.zeros(shape=(self.task_generation_num, 5))
        else:
            self.task_energy_cost = None
            self.task_service_delay = None
        # 定义任务完成、失败任务列表
        self.task_offload_num = 0
        self.task_finish_num = 0
        self.task_fail_num = 0

    # 扫描周围设备
    # 输入: 扫描设备类型, 设备列表
    def near(self, base_device_type, base_device_list: dict):
        # 如果设备类型与本设备不一致
        if base_device_type != self.type:
            for base_device in base_device_list.values():
                distance = get_distance(self.xpos, self.ypos, self.zpos, base_device.xpos, base_device.ypos,
                                        base_device.zpos)
                if distance > self.coverage:
                    continue
                else:
                    self.near_list[base_device_type][base_device.index] = base_device
        # 如果设备类型与本设备一致
        else:
            new_base_device_list = base_device_list.copy()
            # 排除自身
            del new_base_device_list[self.index]
            for base_device in new_base_device_list.values():
                distance = get_distance(self.xpos, self.ypos, self.zpos, base_device.xpos, base_device.ypos,
                                        base_device.zpos)
                if distance > self.coverage:
                    continue
                else:
                    self.near_list[base_device_type][base_device.index] = base_device

    # 发起链接
    # 输入: 链接对象, 自身传输功率(dbm), 对象传输功率(dbm)
    # 输出: (0 链接失败, 1 链接成功)
    def connect(self, base_device, transmission_power1=None, transmission_power2=None):
        # 排除自身
        if base_device.index == self.index and base_device.type == self.type:
            return 0
        # 超出连接数量上限
        if (len(self.connect_list[base_device.type]) + len(self.access_list[base_device.type])) >= \
                self.max_connect_num_list[base_device.type] or \
                (len(base_device.connect_list[self.type]) + len(base_device.access_list[self.type])) >= \
                base_device.max_connect_num_list[self.type]:
            return 0
        distance = get_distance(self.xpos, self.ypos, self.zpos, base_device.xpos, base_device.ypos, base_device.zpos)
        # 排除距离过远
        if distance > self.coverage or distance > base_device.coverage:
            return 0
        # 设置通信双方传输功率
        if transmission_power1 == None:
            transmission_power1 = self.transmission_power
        if transmission_power2 == None:
            transmission_power2 = base_device.transmission_power
        # 设置传输功率上下限
        assert transmission_power1 >= 1 and transmission_power1 <= self.transmission_power
        assert transmission_power2 >= 1 and transmission_power2 <= base_device.transmission_power
        transmission_power1 = max(min(transmission_power1, self.transmission_power), 1)
        transmission_power2 = max(min(transmission_power2, base_device.transmission_power), 1)
        # 链接状态: 链接设备、链接距离、传输功率、接收功率(都是dbm的形式)
        self.connect_list[base_device.type][base_device.index] = [0, 0, 0, 0]
        self.connect_list[base_device.type][base_device.index][0] = base_device
        self.connect_list[base_device.type][base_device.index][1] = distance
        self.connect_list[base_device.type][base_device.index][2] = transmission_power1
        received_power = get_power(transmission_power2, distance)
        self.connect_list[base_device.type][base_device.index][3] = mw_to_dbm(received_power)
        base_device.access(self, distance, transmission_power2, transmission_power1)
        return 1

    # 接入链接
    # 输入: 接入设备, 链接距离(m), 自身传输功率(dbm), 对象传输功率(dbm)
    def access(self, base_device, distance, transmission_power1, transmission_power2):
        self.access_list[base_device.type][base_device.index] = [0, 0, 0, 0]
        self.access_list[base_device.type][base_device.index][0] = base_device
        self.access_list[base_device.type][base_device.index][1] = distance
        self.access_list[base_device.type][base_device.index][2] = transmission_power1
        received_power = get_power(transmission_power2, distance)
        self.access_list[base_device.type][base_device.index][3] = mw_to_dbm(received_power)

    # 断开链接
    # 输入: 断开对象
    # 输出: (0 断开失败, 1 断开成功)
    def disconnect(self, base_device):
        if base_device.index in self.connect_list[base_device.type]:
            del self.connect_list[base_device.type][base_device.index]
            del base_device.access_list[self.type][self.index]
            return 1
        else:
            return 0

    # 任务卸载
    # 输入: 任务, 目标设备(默认自身)
    # 输出: (-1 任务超出时延阈值, 0 无法卸载, 1 卸载成功)
    def task_offload(self, now_time, task, base_device=None):
        if base_device == -1 or now_time < task.start_time:
            return 0
        # 超出时延阈值
        if now_time > task.end_time:
            task.offload_fail(now_time - 1)
            self.task_wait_list.remove(task)
            return -1
        if base_device == None:
            base_device = self
        # 获取空运行位
        free_index = np.where(np.array(base_device.task_run_list) == 0)[0]
        if len(free_index) == 0:
            return 0
        # 卸载任务
        base_device.task_run_list[free_index[0]] = task
        self.task_wait_list.remove(task)
        result = task.offload_finish(now_time, base_device, free_index[0])
        if result:
            base_device.task_offload_num += 1
        return result

    # 任务运行
    # 输入: 时隙长度, 任务, 上行资源分配比例, 计算资源分配比例, 下行资源分配比例
    # 输出: (-1 任务失败, 0 任务未完成, 1 任务完成)
    def task_run(self, now_time, tau, task, upload_resources_rate, compute_resources_rate, download_resources_rate):
        assert task.state == 0 or task.state == 1 or task.state == 2
        # 获取上传、计算、下载的任务数据量
        upload_data = get_speed(dbm_to_mw(self.access_list[task.local_device.type][task.local_device.index][3]),
                                     upload_resources_rate * self.upload_resources) * tau
        compute_data = compute_resources_rate * self.compute_resources * tau
        download_data = get_speed(dbm_to_mw(task.local_device.connect_list[self.type][self.index][3]),
                                       download_resources_rate * self.download_resources) * tau
        # 运行任务
        if task.local_device == self:
            run_state, _ = task.local_device_run(now_time, compute_data)
        else:
            run_state, _ = task.mec_run(now_time, upload_data, compute_data, download_data)
        # 若任务没有完成并且没时间了
        if not task.live(now_time + 1) and run_state != 1:
            task.run_fail(now_time)
            self.task_delect(task)
            return -1
        # 若任务完成
        elif run_state == 1:
            self.task_delect(task)
            return 1
        else:
            return 0

    # 任务删除
    # 输入: 任务
    def task_delect(self, task):
        self.task_run_list[task.vm_index] = 0