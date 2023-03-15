import os
import gym
import json
import time
import numpy as np
import tensorflow as tf
from Agent.DQN.DQN import DQN_Agent
from Agent.DQN.DDQN import DDQN_Agent
from Agent.DQN.Dueling_DQN import Dueling_DQN_Agent
from Agent.DQN.D3QN import D3QN_Agent
from Agent.DDPG.DDPG import DDPG_Agent
from Agent.DDPG.TD3 import TD3_Agent
from Agent.DDPG.MA_TD3 import MA_TD3_Agent
from Env.MEC_Env import MEC

tf.keras.backend.set_floatx("float32")
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
dqn_agent_list = ["DQN", "DDQN", "Dueling_DQN", "D3QN"]
ddpg_agent_list = ["DDPG", "TD3", "MA_TD3"]


class Train():
    def __init__(self):
        # 智能体类型
        self.agent_class = "TD3"
        # 优先经验回放
        self.prioritized_replay = True
        # 环境名称
        self.env_name = "MEC_RA"
        # 实验名称
        if self.prioritized_replay:
            self.exp_name = self.agent_class + "_PER/" + self.env_name
        else:
            self.exp_name = self.agent_class + "/" + self.env_name

        # 日志路径
        self.log_dir = "Log/" + self.exp_name
        # self.log_dir = None
        # 模型加载路径
        # self.model_load_path = "Model/" + self.exp_name
        self.model_load_path = None
        # 经验加载路径
        # self.buffer_load_path = "Model/" + self.exp_name
        self.buffer_load_path = None
        # 模型存储路径
        self.model_save_path = "Model/" + self.exp_name
        # self.model_save_path = None
        # 经验存储路径
        self.buffer_save_path = "Model/" + self.exp_name
        # self.buffer_save_path = None

        # 训练次数以及最大步长
        self.episode_num = 100000
        self.episode_len = 10

        # batch和buffer大小
        self.batch_size = 4096
        self.buffer_size = 100000

        # 网络模型结构
        if self.agent_class in dqn_agent_list:
            self.units_num = 128
            self.layers_num = 3
        elif self.agent_class in ddpg_agent_list:
            self.critic_units_num = 256
            self.critic_layers_num = 3
            self.actor_units_num = 128
            self.actor_layers_num = 3

        # Adam学习率
        if self.agent_class in dqn_agent_list:
            self.lr = 1e-3
        elif self.agent_class in ddpg_agent_list:
            self.critic_lr = 1e-3
            self.actor_lr = 1e-3

        # gamma值
        self.gamma = 0.95

        # 模型训练频率（单位：step）
        self.train_freq = 100
        # 模型存储频率（单位：episode）
        self.save_rate = 10
        # 模型更新频率（单位：train）
        self.update_freq = 4
        # 模型更新权重
        self.tau = 0.1

        # 添加环境噪音
        self.add_noise = False
        # 环境噪音边界值
        self.noise_bound = 0.1

        # 使用Target进行预测
        self.target_action = True

        # 随机探索权重
        self.epsilon = -1
        self.min_epsilon = -1
        self.epsilon_decay = 1e-3

        # 创建智能体
        self.agent = self.agents_create(1, 30, 18)
        if self.model_load_path != None:
            print("读取模型")
            self.agent.model_load(self.model_load_path)
        if self.buffer_load_path != None:
            print("读取经验")
            self.agent.buffer_load(self.buffer_load_path)
        # 创建日志读写
        if self.log_dir != None:
            print("创建日志")
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        # 创建环境
        with open("Env/system_info.json", "r") as f:
            system_info = json.load(f)
        self.env = MEC(**system_info)

    def dqn_act(self, state):
        if np.random.uniform() <= self.epsilon:
            return np.random.choice([0, 1, 2])
        if self.target_action:
            action = self.agent.target_action(np.array([state]))
        else:
            action = self.agent.action(np.array([state]))
        if self.add_noise:
            action = self.noise(action)
        return np.argmax(action)

    def ddpg_act(self, state):
        if np.random.uniform() <= self.epsilon:
            return np.random.uniform(low=-1, high=1, size=(1,))
        if self.target_action:
            action = self.agent.target_action(np.array([state]))
        else:
            action = self.agent.action(np.array([state]))
        if self.add_noise:
            action = self.noise(action)
        return action

    def noise(self, action, noise_bound=None):
        if noise_bound == None:
            noise = np.random.uniform(low=-1, high=1, size=(action.size,)) * self.noise_bound
        else:
            noise = np.random.uniform(low=-1, high=1, size=(action.size,)) * noise_bound
        noise_action = noise + action
        return noise_action

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def train(self):
        rewards_list = []
        sum_step = 0
        for each in range(self.episode_num):
            rewards = 0
            _, state = self.env.reset()
            state = state[0]
            step = 0
            while True:
                if self.agent_class in dqn_agent_list:
                    action = self.dqn_act(state)
                elif self.agent_class in ddpg_agent_list:
                    action = self.ddpg_act(state)
                _, _, next_state, reward, done = self.env.system_step(None, [action])
                next_state = next_state[0]
                reward = reward[0]
                step += 1
                sum_step += 1
                rewards += reward
                done = done
                self.agent.remember(state, action, next_state, reward, done)
                state = next_state
                if self.agent.replay_buffer.size() >= self.batch_size * self.episode_len:
                    if sum_step % self.train_freq == 0 and self.train_freq != 0:
                        self.agent.train()
                if done:
                    rewards_list.append(rewards)
                    print("episode:", str(each), "replay_buffer_len:", self.agent.replay_buffer.size(), "epsilon",
                          self.epsilon, "reward:", rewards, "step", step, "max_reward", max(rewards_list))
                    self.update_epsilon()
                    if self.log_dir != None:
                        with self.summary_writer.as_default():
                            tf.summary.scalar('Reward', rewards, step=each)
                            tf.summary.scalar('Step', step, step=each)
                            tf.summary.scalar('Finish_Num', self.env.finish_num, step=each)
                            tf.summary.scalar('Fail_Num', self.env.fail_num, step=each)
                    break
            if each % self.save_rate == 0 and self.model_save_path != None:
                self.agent.model_save(self.model_save_path)
            if each % self.save_rate == 0 and self.buffer_save_path != None:
                self.agent.buffer_save(self.buffer_save_path)

    def agents_create(self, index, state_shape, action_shape):
        if self.agent_class == "DQN":
            agent = DQN_Agent(agent_index=index, state_shape=state_shape, action_shape=action_shape,
                              units_num=self.units_num, layers_num=self.layers_num, lr=self.lr,
                              batch_size=self.batch_size, buffer_size=self.buffer_size,
                              gamma=self.gamma, tau=self.tau, update_freq=self.update_freq,
                              prioritized_replay=self.prioritized_replay
                              )
        elif self.agent_class == "DDQN":
            agent = DDQN_Agent(agent_index=index, state_shape=state_shape, action_shape=action_shape,
                              units_num=self.units_num, layers_num=self.layers_num, lr=self.lr,
                              batch_size=self.batch_size, buffer_size=self.buffer_size,
                              gamma=self.gamma, tau=self.tau, update_freq=self.update_freq,
                              prioritized_replay=self.prioritized_replay
                              )
        elif self.agent_class == "Dueling_DQN":
            agent = Dueling_DQN_Agent(agent_index=index, state_shape=state_shape, action_shape=action_shape,
                              units_num=self.units_num, layers_num=self.layers_num, lr=self.lr,
                              batch_size=self.batch_size, buffer_size=self.buffer_size,
                              gamma=self.gamma, tau=self.tau, update_freq=self.update_freq,
                              prioritized_replay=self.prioritized_replay
                              )
        elif self.agent_class == "D3QN":
            agent = D3QN_Agent(agent_index=index, state_shape=state_shape, action_shape=action_shape,
                              units_num=self.units_num, layers_num=self.layers_num, lr=self.lr,
                              batch_size=self.batch_size, buffer_size=self.buffer_size,
                              gamma=self.gamma, tau=self.tau, update_freq=self.update_freq,
                              prioritized_replay=self.prioritized_replay
                              )
        elif self.agent_class == "DDPG":
            agent = DDPG_Agent(agent_index=index, state_shape=state_shape, action_shape=action_shape, critic_units_num=self.critic_units_num,
                               critic_layers_num=self.critic_layers_num, critic_lr=self.critic_lr,
                               actor_units_num=self.actor_units_num, actor_layers_num=self.actor_layers_num,
                               actor_lr=self.actor_lr,
                               batch_size=self.batch_size, buffer_size=self.buffer_size, gamma=self.gamma, tau=self.tau,
                               update_freq=self.update_freq,
                               prioritized_replay=self.prioritized_replay, activation="softmax")
        elif self.agent_class == "TD3":
            agent = TD3_Agent(agent_index=index, state_shape=state_shape, action_shape=action_shape, critic_units_num=self.critic_units_num,
                               critic_layers_num=self.critic_layers_num, critic_lr=self.critic_lr,
                               actor_units_num=self.actor_units_num, actor_layers_num=self.actor_layers_num,
                               actor_lr=self.actor_lr,
                               batch_size=self.batch_size, buffer_size=self.buffer_size, gamma=self.gamma, tau=self.tau,
                               update_freq=self.update_freq,
                               prioritized_replay=self.prioritized_replay, activation="softmax")
        elif self.agent_class == "MA_TD3":
            agent = MA_TD3_Agent(agent_index=index, state_shape=state_shape, action_n_shape=action_shape, critic_units_num=self.critic_units_num,
                               critic_layers_num=self.critic_layers_num, critic_lr=self.critic_lr,
                               actor_units_num=self.actor_units_num, actor_layers_num=self.actor_layers_num,
                               actor_lr=self.actor_lr,
                               batch_size=self.batch_size, buffer_size=self.buffer_size, gamma=self.gamma, tau=self.tau,
                               update_freq=self.update_freq,
                               prioritized_replay=self.prioritized_replay, activation="softmax")
        return agent


if __name__ == "__main__":
    train = Train()
    train.train()