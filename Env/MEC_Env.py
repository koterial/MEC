import json
import time
import random
import numpy as np
import tensorflow as tf
from Env.Device_Env import Base_Device
from Env.Task_Env import Base_Task
from Utils.Common import *


# 定义MEC运行环境
class MEC():
    def __init__(self, *args, **kwargs):
        # 定义世界参数
        self.map_width = kwargs["mec_system"]["map_width"]
        self.map_height = kwargs["mec_system"]["map_height"]
        self.screen_width = kwargs["mec_system"]["screen_width"]
        self.screen_height = kwargs["mec_system"]["screen_height"]

        # 定义时间参数
        self.tau = kwargs["mec_system"]["tau"]
        # 起始仿真时间
        self.start_time = 1
        # 定义仿真时间
        self.max_run_time = kwargs["mec_system"]["max_run_time"]
        # 结束仿真时间
        self.end_time = self.max_run_time + self.start_time - 1
        # 当前仿真时间
        self.now_time = self.start_time

        # 定义系统设备信息
        self.system_device_info_list = kwargs["device_info_list"]
        self.system_device_num_list = kwargs["mec_system"]["device_num_list"]
        self.system_device_num = sum(self.system_device_num_list)

        # 定义归一化量
        self.max_task_run_time = max([device_info["max_task_run_time"] for device_info in self.system_device_info_list])
        self.max_task_upload_data = max([device_info["max_task_upload_data"] for device_info in self.system_device_info_list])
        self.max_task_store_data = max([device_info["max_task_store_data"] for device_info in self.system_device_info_list])
        self.max_task_compute_data = max([device_info["max_task_compute_data"] for device_info in self.system_device_info_list])
        self.max_task_download_data = max([device_info["max_task_download_data"] for device_info in self.system_device_info_list])
        self.max_receive_power = get_power(max([device_info["max_transmission_power"] for device_info in self.system_device_info_list]), 25)

    # 重置系统
    def reset(self):
        self.now_time = self.start_time
        self.system_device_list = []
        # 定义系统设备列表
        for _ in self.system_device_info_list:
            self.system_device_list.append({})
        # 定义任务完成/失败数量
        self.finish_num = 0
        self.fail_num = 0

        # 初始化设备
        for device_info, device_num in zip(self.system_device_info_list, self.system_device_num_list):
            for each1 in range(1, device_num + 1):
                # 坐标随机初始化
                xpos = math.floor(np.random.uniform(low=0, high=1, size=1)[0] * self.map_width)
                ypos = math.floor(np.random.uniform(low=0, high=1, size=1)[0] * self.map_height)
                zpos = 0
                velocity = 0
                solid_angle_1 = 0
                solid_angle_2 = 0
                # 资源都按照最大初始化
                upload_resources = math.floor(np.random.uniform(low=1, high=1, size=1)[0] * device_info["max_upload_resources"])
                store_resources = math.floor(np.random.uniform(low=1, high=1, size=1)[0] * device_info["max_store_resources"])
                compute_resources = math.floor(np.random.uniform(low=1, high=1, size=1)[0] * device_info["max_compute_resources"])
                download_resources = math.floor(np.random.uniform(low=1, high=1, size=1)[0] * device_info["max_download_resources"])
                transmission_power = math.floor(np.random.uniform(low=1, high=1, size=1)[0] * device_info["max_transmission_power"])
                compute_power = math.floor(np.random.uniform(low=1, high=1, size=1)[0] * device_info["max_compute_power"])
                standby_power = math.floor(np.random.uniform(low=1, high=1, size=1)[0] * device_info["max_standby_power"])
                coverage = math.floor(np.random.uniform(low=1, high=1, size=1)[0] * device_info["max_coverage"])
                if device_info["max_task_generation_num"] == 0:
                    task_generation_num = 0
                else:
                    task_generation_num = np.random.randint(low=1, high=device_info["max_task_generation_num"] + 1, size=1)[0]
                max_task_load_num = np.random.randint(low=device_info["max_task_load_num"], high=device_info["max_task_load_num"] + 1, size=1)[0]
                max_connect_num = device_info["max_connect_num_list"]
                init = {'index': each1, 'type': device_info["type"],
                        'xpos': xpos, 'ypos': ypos, 'zpos': zpos,
                        'velocity': velocity, 'solid_angle_1': solid_angle_1, 'solid_angle_2': solid_angle_2,
                        'transmission_power': transmission_power, 'compute_power': compute_power, 'standby_power': standby_power,
                        'upload_resources': upload_resources, 'store_resources': store_resources, 'compute_resources': compute_resources, 'download_resources': download_resources,
                        'coverage': coverage, 'max_task_load_num': max_task_load_num, 'task_generation_num': task_generation_num}
                device = Base_Device(*max_connect_num, **init)
                if task_generation_num >= 1:
                    # 初始化任务
                    for each2 in range(1, task_generation_num + 1):
                        upload_data = max(math.floor(np.random.uniform(low=0, high=1, size=1)[0] * device_info["max_task_upload_data"]), 1)
                        store_data = max(math.floor(np.random.uniform(low=0, high=1, size=1)[0] * device_info["max_task_store_data"]), 0)
                        compute_data = max(math.floor(np.random.uniform(low=0, high=1, size=1)[0] * device_info["max_task_compute_data"]), 1)
                        download_data = max(math.floor(np.random.uniform(low=0, high=1, size=1)[0] * device_info["max_task_download_data"]), 1)
                        max_run_time = max(math.floor(np.random.uniform(low=0, high=1, size=1)[0] * device_info["max_task_run_time"]), 1)
                        start_time = 1
                        init = {'index': each2, 'local_device': device, 'start_time': start_time,
                                'max_run_time': max_run_time,
                                'upload_data': upload_data, 'store_data': store_data,
                                'compute_data': compute_data, 'download_data': download_data}
                        task = Base_Task(**init)
                        device.task_wait_list.append(task)
                self.system_device_list[device_info["type"]][each1] = device

        # 将所有设备连接到1号基站
        for device_list in self.system_device_list[1:]:
            for device in device_list.values():
                device.connect(self.system_device_list[0][1])
                for task in device.task_wait_list:
                    device.task_offload(self.now_time, task, self.system_device_list[0][1])
        return self.get_co_state(), self.get_ra_state()

    # 系统运行
    # 输入:任务卸载策略, 资源分配策略
    def system_step(self, co_action_list, ra_action_list):
        # 每个时隙内先进行任务卸载, 再进行资源分配
        # co_reward_list = self.co_step(co_action_list)
        ra_reward_list = self.ra_step(ra_action_list)
        # 下一时隙
        self.now_time += 1
        # time.sleep(0.1)
        done = self.now_time == self.end_time + 1 or self.finish_num + self.fail_num == 5
        co_state_list = self.get_co_state()
        ra_state_list = self.get_ra_state()
        return co_state_list, None, ra_state_list, ra_reward_list, done

    # 设备运行
    # 输入: 任务卸载策略
    def co_step(self, action_list):
        pass

    # 任务运行
    # 输入: 资源分配策略
    def ra_step(self, action_list):
        action_list = self.ra_action_deal(action_list)
        reward_list = np.zeros((self.system_device_num_list[0],)).tolist()
        for _, device in self.system_device_list[0].items():
            reward = 0
            action = action_list[device.index - 1]
            for task_index, task in enumerate(device.task_run_list):
                if task == 0:
                    continue
                if task.upload_data == 0:
                    energy_reward = 0.1
                    device.device_energy_cost[0] += 0.1
                    device.device_energy_cost[1] += 0.1
                else:
                    energy_reward = 0.6
                    device.device_energy_cost[0] += 0.6
                    device.device_energy_cost[1] += 0.6
                run_state = device.task_run(self.now_time, self.tau, task, action[0][task.vm_index], action[1][task.vm_index], action[2][task.vm_index])
                time_reward = (2 - (task.start_time + task.max_run_time - self.now_time) / (self.max_task_run_time))
                data_reward = (task.upload_data / self.max_task_upload_data + task.compute_data / self.max_task_compute_data + task.download_data / self.max_task_download_data)
                reward -= 1 * energy_reward + 1 * time_reward * data_reward
                if run_state == 1:
                    reward += 10
                    device.task_finish_num += 1
                    self.finish_num += 1
                elif run_state == -1:
                    reward -= 10
                    device.task_fail_num += 1
                    self.fail_num += 1
                reward_list[device.index - 1] = reward
        return reward_list

    # 对资源分配策略进行处理
    def ra_action_deal(self, action_list):
        def norm(act):
            act = np.where(act < 1e-4, 0, act)
            if np.sum(act) > 1.001:
                print("error action")
            return act
        new_action_list = []
        for action in action_list:
            action = np.reshape(action, (3, -1))
            upload_act = norm(action[0])
            compute_act = norm(action[1])
            download_act = norm(action[2])
            action[0, :] = upload_act
            action[1, :] = compute_act
            action[2, :] = download_act
            new_action_list.append(action)
        return new_action_list

    def get_co_state(self):
        pass

    # 获取资源分配智能体状态
    def get_ra_state(self):
        state_list = []
        for _, device in self.system_device_list[0].items():
            state = np.zeros(shape=device.ra_state_space)
            for task_index, task in enumerate(device.task_run_list):
                if task == 0:
                    continue
                else:
                    # 时间状态
                    state[task.vm_index][0] = (task.start_time + task.max_run_time - self.now_time) / (self.max_task_run_time)
                    # 数据状态
                    state[task.vm_index][1] = task.upload_data / self.max_task_upload_data
                    state[task.vm_index][2] = task.compute_data / self.max_task_compute_data
                    state[task.vm_index][3] = task.download_data / self.max_task_download_data
                    # 上行链路接收功率状态
                    state[task.vm_index][4] = dbm_to_mw(device.access_list[task.local_device.type][task.local_device.index][3]) / self.max_receive_power
                    # 下行链路接收功率状态
                    state[task.vm_index][5] = dbm_to_mw(task.local_device.connect_list[device.type][device.index][3]) / self.max_receive_power
            if np.max(state[:, 0:4]) > 1 or np.min(state[:, 0:4]) < 0:
                print(state)
            state_list.append(state.flatten())
        return state_list



if __name__ == "__main__":
    with open("system_info.json", "r") as f:
        system_info = json.load(f)
    mec = MEC(**system_info)
    reward_list = []
    for each in range(1000):
        _, ra_state_list = mec.reset()
        total_ra_reward_list = np.zeros((mec.system_device_num_list[0],)).tolist()
        while True:
            ra_action_list = [np.random.uniform(size=(15, )) for _, device in mec.system_device_list[0].items()]
            for action in ra_action_list:
                action = np.reshape(action, (3, -1))
                action[0] = tf.nn.softmax(action[0]).numpy()
                action[1] = tf.nn.softmax(action[1]).numpy()
                action[2] = tf.nn.softmax(action[2]).numpy()
                action = np.reshape(action, (15,))
            _, _, next_ra_state_list, ra_reward_list, done = mec.system_step(None, ra_action_list)
            total_ra_reward_list += ra_reward_list
            if done:
                offload_num = 0
                finish_num = 0
                fail_num = 0
                for _, device in mec.system_device_list[0].items():
                    offload_num += device.task_offload_num
                    finish_num += device.task_finish_num
                    fail_num += device.task_fail_num
                print(offload_num, finish_num, fail_num, sum(total_ra_reward_list))
                reward_list.append(sum(total_ra_reward_list))
                break
    print(np.mean(reward_list))