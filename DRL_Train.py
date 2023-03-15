import os
import random
import Env.MEC_Env
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from Agent.DDPG.DDPG import DDPG_Agent
from Agent.DDPG.TD3 import TD3_Agent
from Agent.DDPG.MA_TD3 import MA_TD3_Agent

tf.keras.backend.set_floatx('float32')
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

# seed = 3407
# tf.random.set_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# os.environ["PYTHONHASHSEED"] = str(seed)

class Train():
    def __init__(self):
        # 设置模型
        self.model = "DDPG"
        # 设置实验名称
        self.exp_name = "train2_2"

        # 设置读取路径
        # self.load_path = "model/" + self.model + "/" + self.exp_name
        self.load_path = None

        # 设置存储路径
        self.save_path = "model/" + self.model + "/" + self.exp_name
        # self.save_path = None
        # 设置存储频率(单位：episode)
        self.save_rate = 100

        # 设置日志路径
        self.log_dir = "logs/" + self.model + "/" + self.exp_name
        # self.log_dir = None
        if self.log_dir != None:
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        # 创建环境
        self.env = mec_env.mec()

        # 设置训练次数以及最大步长
        self.episode_num = 1000000
        self.max_step = 10

        # 设置Adam学习率
        self.critic_lr = 1e-3
        self.actor_lr = 1e-3

        # 设置gamma值
        self.gamma = 0.95

        # 设置batch和buffer大小
        self.batch_size = 512
        self.buffer_size = 100000

        # 设置网络模型结构
        self.critic_units_num = 256
        self.critic_layers_num = 4
        self.actor_units_num = 64
        self.actor_layers_num = 3

        # 设置使用Target Actor进行预测
        self.target_action = True

        # 设置Critic模型更新频率(单位：step)
        self.update_freq = 10

        # 设置Actor更新频率(相较于Critic)
        self.actor_update_rate = 1
        # 设置模型更新权重
        self.tau = 0.95

        # 设置噪音
        self.add_noise = False

        # 创建智能体
        self.agent = self.agents_create()
        if self.load_path is not None:
            print("读取历史状态")
            self.agent.model_load(self.load_path)

    def train(self):
        sum_step = 0
        rewards_list = []
        for each in range(self.episode_num):
            rewards = 0
            state = self.env.reset()
            step = 0
            while True:
                if self.target_action:
                    action = self.agent.target_action(np.array([state]))[0]
                else:
                    action = self.agent.action(np.array([state]))[0]
                if self.add_noise:
                    action = self.noise(action)
                next_state, reward, done, _ = self.env.step(action)
                step += 1
                sum_step += 1
                rewards += reward
                done = done
                self.agent.remember(state, action, next_state, reward, done)
                state = next_state
                if self.agent.replay_buffer.size() >= self.batch_size * self.max_step:
                    if sum_step % self.update_freq == 0:
                        self.agent.train()
                if done:
                    finish_time = 0
                    offload_num = 0
                    for user_key, user in self.env.user_list.items():
                        for task_key, task in user.task_list.items():
                            if task.end_time != -1:
                                finish_time += task.end_time + 1 - task.start_time
                                offload_num += 1
                            else:
                                pass
                    rewards_list.append(rewards)
                    print("episode:", str(each), "replay_buffer_len:", self.agent.replay_buffer.size(),"Reward:", rewards, "Offload_num:", offload_num, "Finish_num:", self.env.finish, "Finish_time:", finish_time, "Energy_cost:", self.env.energy_cost)
                    if self.log_dir != None:
                        with self.summary_writer.as_default():
                            tf.summary.scalar('Reward', rewards, step=each)
                            tf.summary.scalar('Offload_num', offload_num, step=each)
                            tf.summary.scalar('Finish_num', self.env.finish, step=each)
                            tf.summary.scalar('Finish_time', finish_time, step=each)
                            tf.summary.scalar('Energy_cost', self.env.energy_cost, step=each)
                    break
            if each % self.save_rate == 0 and self.save_path != None:
                self.agent.model_save(self.save_path)

    def agents_create(self):
        if self.model == "DDPG":
            agent = DDPG_Agent(state_shape=self.env.observation_space, action_shape=self.env.action_space,
                               agent_index=1, batch_size=self.batch_size, buffer_size=self.buffer_size,
                               critic_lr=self.critic_lr,
                               actor_lr=self.actor_lr,
                               critic_units_num=self.critic_units_num, critic_layers_num=self.critic_layers_num,
                               actor_units_num=self.actor_units_num, actor_layers_num=self.actor_layers_num,
                               gamma=self.gamma, tau=self.tau
                               )
        elif self.model == "TD3":
            agent = TD3_Agent(state_shape=self.env.observation_space, action_shape=self.env.action_space,
                               agent_index=1, batch_size=self.batch_size, buffer_size=self.buffer_size,
                               critic_lr=self.critic_lr,
                               actor_lr=self.actor_lr,
                               critic_units_num=self.critic_units_num, critic_layers_num=self.critic_layers_num,
                               actor_units_num=self.actor_units_num, actor_layers_num=self.actor_layers_num,
                               gamma=self.gamma, tau=self.tau, actor_update_freq = self.actor_update_rate
                               )
        elif self.model == "MA_TD3":
            agent = MA_TD3_Agent(state_shape=self.env.observation_space,
                                 action_n_shape=[self.env.egs_max_task_load + 1, self.env.egs_max_task_load + 1,
                                                 self.env.egs_max_task_load + 1],
                                 agent_index=1, batch_size=self.batch_size, buffer_size=self.buffer_size,
                                 critic_lr=self.critic_lr,
                                 actor_lr=self.actor_lr,
                                 critic_units_num=self.critic_units_num, critic_layers_num=self.critic_layers_num,
                                 actor_units_num=self.actor_units_num, actor_layers_num=self.actor_layers_num,
                                 gamma=self.gamma, tau=self.tau, actor_update_freq=self.actor_update_rate
                                 )
        return agent

    def noise(self, action):
        def softmax(action):
            action = tf.nn.softmax(action)
            return action.numpy()
        noise = np.random.uniform(low=0, high=0.15, size=(3, self.env.egs_max_task_load + 1))
        action = np.array(action)
        action = action.reshape((3, self.env.egs_max_task_load + 1))
        action[0] = softmax(noise[0] + action[0])
        action[1] = softmax(noise[1] + action[1])
        action[2] = softmax(noise[2] + action[2])
        action = action.flatten()
        return action


if __name__ == "__main__":
    train = Train()
    train.train()