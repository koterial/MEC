import numpy as np
import math
from Env import egs_env, user_env, Task_Env


class mec():
    def __init__(self):
        # 定义关键参数
        # 用户和边缘服务器数量
        self.user_num = 5
        # 单个用户能产生的最大任务数量
        self.user_max_task_load = 1
        # 单个边缘服务器的虚拟机承载数量
        self.egs_max_task_load = 5

        # 单个任务的上传数据总量bit
        self.task_max_upload_data_size = 3 * 1e6
        # 单个任务的计算总量cycle
        self.task_max_compute_data_size = 2 * 1e8
        # 单个任务的下载数据总量bit
        self.task_max_download_data_size = 1.5 * 1e6

        # 上传量归一化
        self.max_upload_data_size = 3 * 1e6
        # 计算量归一化
        self.max_compute_data_size = 2 * 1e8
        # 下载量归一化
        self.max_download_data_size = 1.5 * 1e6

        # 接收功率归一化
        self.uplink_power = 27
        self.downlink_power = 30
        self.max_uplink_power = self.get_power(self.uplink_power, 25)
        self.max_downlink_power = self.get_power(self.downlink_power, 25)

        # 定义时间参数
        # 单个时隙长度ms
        self.tau = 10
        # 单个任务的最大运行时间(时隙数量)
        self.task_max_run_time = 50
        # 起始仿真时间
        self.start_time = 1
        # 定义仿真时间
        self.max_run_time = 100
        # 当前仿真时间
        self.now_time = self.start_time
        # 结束仿真时间
        self.end_time = self.max_run_time + self.start_time

        # 边缘服务器的上传链路带宽Hz
        self.egs_max_upload_cap = 1 * 1e7
        # 边缘服务器的单位时隙计算能力
        self.egs_max_compute_cap = 1 * 1e7 * self.tau
        # 边缘服务器的下载链路带宽Hz
        self.egs_max_download_cap = 0.5 * 1e7

        # 定义用户、任务、任务等待、任务运行、边缘服务器、虚拟机列表
        self.user_list = {}
        self.task_list = {}
        self.task_wait_queue = {}
        self.task_run_queue = np.zeros(shape=self.egs_max_task_load).tolist()

        # 定义完成、失败任务列表
        self.finish = 0
        self.fail = np.zeros(2)

        # 定义浪费资源列表
        self.waste_upload_cap = 0
        self.waste_compute_cap = 0
        self.waste_download_cap = 0

        # 定义能量开销列表
        self.energy_cost = np.zeros(2)

        # 定义服务时延列表
        self.service_delay = np.zeros(4)

        # 定义单个边缘服务器的观测、动作空间
        self.observation_space = self.egs_max_task_load * 6
        self.action_space = (self.egs_max_task_load + 1) * 3

    def reset(self):
        # 重置系统
        self.now_time = self.start_time
        self.user_list = {}
        self.task_list = {}
        self.task_wait_queue = {}
        self.task_run_queue = np.zeros(shape=(self.egs_max_task_load)).tolist()
        self.finish = 0
        self.fail = np.zeros(2)
        self.waste_upload_cap = 0
        self.waste_compute_cap = 0
        self.waste_download_cap = 0
        self.energy_cost = np.zeros(2)
        self.service_delay = np.zeros(4)

        for each1 in range(1, self.user_num + 1):
            # 初始化位置
            dis = np.random.uniform(low=-1, high=1, size=1)[0] * 225 + 250
            # 创建移动用户，编号从1开始
            user = user_env.user(index=each1, dis=dis, vel=0, angle=0, connect_distance=200)
            # 初始化移动用户承载任务数量
            user.task_num = np.random.randint(low=1 ,high=self.user_max_task_load + 1, size=1)[0]
            # 初始化边缘服务器接收功率
            user.uplink_power = self.get_power(self.uplink_power, user.dis)
            user.downlink_power = self.get_power(self.downlink_power, user.dis)
            # 创建移动用户任务
            for each2 in range(1, user.task_num + 1):
                # # 初始化任务属性
                upload_data_size = max(math.floor(np.random.uniform(low=0, high=1, size=1)[0] * self.task_max_upload_data_size),1)
                compute_size = max(math.floor(np.random.uniform(low=0, high=1, size=1)[0] * self.task_max_compute_data_size),1)
                download_data_size = max(math.floor(np.random.uniform(low=0, high=1, size=1)[0] * self.task_max_download_data_size),1)
                max_run_time = max(math.floor(np.random.uniform(low=0, high=1, size=1)[0] * self.task_max_run_time),1)
                start_time = 1
                # 创建任务
                task = task_env.task(index=each2, user_index=each1, egs_index=0, start_time=start_time, compute_size=compute_size, upload_data_size=upload_data_size, store_data_size=0, download_data_size=download_data_size, max_rum_time=max_run_time)
                user.task_list[each2] = task
            self.user_list[each1] = user
            self.task_list[each1] = user.task_list
        # 初始化边缘服务器属性
        upload_cap =  self.egs_max_upload_cap
        compute_cap = self.egs_max_compute_cap
        download_cap = self.egs_max_download_cap
        egs = egs_env.egs(index=1, compute_cap=compute_cap, upload_cap=upload_cap, download_cap=download_cap, store_cap=self.egs_max_task_load)
        self.egs = egs
        # 初始化任务等待队列
        for time in range(self.start_time, self.end_time):
            task_queue = []
            for tasks in self.task_list:
                for task in self.task_list[tasks]:
                    if self.task_list[tasks][task].start_time == time:
                        task_queue.append(self.task_list[tasks][task])
            self.task_wait_queue[time] = task_queue
        self.task_wait_queue[self.end_time] = []
        self.task_add()
        obs = self.get_obs()
        return obs

    # 将任务等待列表中的任务添加到某个边缘服务器的任务运行列表中
    def task_add(self):
        # 获取本时刻的任务队列
        task_wait_queue = self.task_wait_queue[self.now_time]
        if len(task_wait_queue) == 0:
            return
        task_run_queue = self.task_run_queue
        free_index = np.where(np.array(task_run_queue) == 0)[0]
        free_size = len(free_index)
        free_index = iter(free_index)
        # 添加任务
        for task in task_wait_queue[:min(free_size, len(task_wait_queue))]:
            task_run_queue[next(free_index)] = task
            task.state = 0
            task.egs_index = 1
        # 对无法添加的任务进行处理
        if free_size < len(task_wait_queue):
            for task in task_wait_queue[min(free_size, len(task_wait_queue)):]:
                # 任务执行时间：start_time ~ start_time + run_time -1
                if task.start_time + task.max_run_time > self.now_time + 1:
                    self.task_wait_queue[self.now_time + 1].append(task)
                else:
                    task.state = -2
                    self.fail[-1] += 1
        self.task_run_queue = task_run_queue

    def task_delet(self, task_run_queue):
        task_run_queue=np.array(task_run_queue)#转为数组
        ind=np.nonzero(task_run_queue)#非0数的坐标
        zeros=len(task_run_queue)-np.count_nonzero(task_run_queue) #0的个数
        return list(task_run_queue[ind])+[0]*zeros

    # action形式：(action_dim * egs_max_task_load)
    # action_dim形式：上传，计算，下载
    def step(self, action):
        # 动作预处理，将(action_dim * egs_max_task_load)变成所需要的形状
        action = self.action_deal(np.array(action))[:,:-1]
        # 初始化奖励值
        reward = 0
        # 获取当前时刻所有边缘服务器的运行列表
        task_run_queue = self.task_run_queue
        # 对于运行列表中的任务进行循环
        for task_index, task in enumerate(task_run_queue):
            # 获取分配给这个任务位的资源
            upload_cap = action[0][task_index] * self.egs.upload_cap
            compute_cap = action[1][task_index] * self.egs.compute_cap
            download_cap = action[2][task_index] * self.egs.download_cap
            # 若此任务位没有任务
            if task == 0:
                self.waste_upload_cap += upload_cap
                self.waste_compute_cap += compute_cap
                self.waste_download_cap += download_cap
                # reward -= 1 * (upload_cap/self.egs_max_upload_cap + compute_cap/self.egs_max_compute_cap + download_cap/self.egs_max_download_cap)
                continue
            upload_speed = self.get_speed(self.user_list[task.user_index].uplink_power, upload_cap) * self.tau / 1000
            download_speed = self.get_speed(self.user_list[task.user_index].downlink_power, download_cap) * self.tau / 1000
            # 先进行数据的传输
            if task.upload_data_size != 0:
                waste_upload = max(upload_speed - task.upload_data_size,0)
                task.upload_data_size = max(task.upload_data_size - upload_speed, 0)
                reward -= 1 * 0.6 + 1 * (2 - (task.start_time + task.max_run_time - self.now_time)/(self.task_max_run_time)) * (
                            task.upload_data_size / self.max_upload_data_size + task.compute_size / self.max_compute_data_size + task.download_data_size / self.max_download_data_size) + 0 * (waste_upload / self.egs_max_upload_cap + compute_cap / self.egs_max_compute_cap + download_cap / self.egs_max_download_cap)
                # reward -= 0.6 + 1
                self.waste_upload_cap += waste_upload
                self.waste_compute_cap += compute_cap
                self.waste_download_cap += download_cap
                self.energy_cost += 0.6
            # 在进行数据的处理
            elif task.compute_size != 0:
                waste_compute = max(compute_cap - task.compute_size, 0)
                task.compute_size = max(task.compute_size - compute_cap, 0)
                reward -= 1 * 0.1 + 1 * (2 - (task.start_time + task.max_run_time - self.now_time)/(self.task_max_run_time)) * (
                        task.compute_size / self.max_compute_data_size + task.download_data_size / self.max_download_data_size) + 0 * (upload_cap / self.egs_max_upload_cap + waste_compute / self.egs_max_compute_cap + download_cap / self.egs_max_download_cap)
                # reward -= 0.1 + 1
                self.waste_upload_cap += upload_cap
                self.waste_compute_cap += waste_compute
                self.waste_download_cap += download_cap
                self.energy_cost += 0.1
            # 最后进行数据的处理
            elif task.download_data_size != 0:
                waste_download = max(download_speed - task.download_data_size, 0)
                task.download_data_size = max(task.download_data_size - download_speed, 0)
                reward -= 1 * 0.1 + 1 * (2 - (task.start_time + task.max_run_time - self.now_time)/(self.task_max_run_time)) * (
                        task.download_data_size / self.max_download_data_size) + 0 * (upload_cap / self.egs_max_upload_cap + compute_cap / self.egs_max_compute_cap + waste_download / self.egs_max_download_cap)
                # reward -= 0.1 + 1
                self.waste_upload_cap += upload_cap
                self.waste_compute_cap += compute_cap
                self.waste_download_cap += waste_download
                self.energy_cost += 0.1
            # 判断本时刻任务是否处理完成
            if task.upload_data_size == task.compute_size == task.download_data_size == 0:
                # 处理完成设置完成率为1
                task.finish_rate = 1
                task.end_time = self.now_time
                # 处理完后设置状态为1
                task.state = 1
                # 处理完成后的奖励
                self.finish += 1
                reward += 10
                task_run_queue[task_index] = 0
            # 任务未完成处理
            else:
                # 若下一时刻不满足阈值，则设置状态-2为失败
                if self.now_time + 1 == task.start_time + task.max_run_time:
                    task.state = -2
                    task.end_time = self.now_time
                    reward -= 10
                    self.fail[0] += 1
                    task_run_queue[task_index] = 0
        # task_run_queue = self.task_delet(task_run_queue)
        self.task_run_queue = task_run_queue
        self.now_time += 1
        # self.task_add()
        done = self.now_time == self.end_time or self.finish + sum(self.fail) == self.user_num
        obs = self.get_obs()
        return obs, reward, done, None

    # state形式：(self.egs_max_task_load * 6)
    def get_obs(self):
        obs = np.zeros(shape=(self.egs_max_task_load, 6))
        for task_index, task in enumerate(self.task_run_queue):
            if task == 0:
                continue
            else:
                obs[task_index][0] = (task.start_time + task.max_run_time - self.now_time)/(self.task_max_run_time)
                obs[task_index][1] = (self.user_list[task.user_index].uplink_power)/(self.max_uplink_power)
                obs[task_index][2] = (self.user_list[task.user_index].downlink_power)/(self.max_downlink_power)
                obs[task_index][3] = task.upload_data_size / self.max_upload_data_size
                obs[task_index][4] = task.compute_size / self.max_compute_data_size
                obs[task_index][5] = task.download_data_size / self.max_download_data_size
        obs = obs.flatten()
        if np.max(obs) > 1 or np.min(obs) < 0:
            print(obs)
        return obs

    def action_deal(self, action):
        action = np.reshape(action, [3, self.egs_max_task_load + 1])
        upload_act = action[0, :].reshape((-1)).copy()
        compute_act = action[1, :].reshape((-1)).copy()
        download_act = action[2, :].reshape((-1)).copy()
        upload_act = self.norm(upload_act)
        compute_act = self.norm(compute_act)
        download_act = self.norm(download_act)
        action[0, :] = upload_act
        action[1, :] = compute_act
        action[2, :] = download_act
        return action

    def norm(self, a):
        a = np.where(a<1e-3, 0, a)
        if np.sum(a) > 1.01:
            print("error action")
        return a

# 距离转换成功率
def get_power(self, power, distance):
    log_p = (power-128.1-37.6*math.log10(distance/1000))/10
    power = math.pow(10, log_p)
    return power

# 功率转换成速率
def get_speed(self, power, bandwidth, noise_power=math.pow(10, -17.4)):
    noise_totle_power = max(bandwidth * noise_power,1e-18)
    SNIR = power/noise_totle_power
    speed = bandwidth * math.log2(1+SNIR)
    return speed