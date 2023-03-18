import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from Agent.Utils.Common import clip_by_local_norm
from Agent.Replay_Buffer.Replay_Buffer import Replay_Buffer, Prioritized_Replay_Buffer

tf.keras.backend.set_floatx('float32')


class DQN_Agent():
    def __init__(self, agent_index, state_shape, action_shape, units_num, layers_num, lr, batch_size, buffer_size,
                 gamma, tau, update_freq=1, activation="linear", prioritized_replay=False,
                 alpha=0.6, beta=0.4, beta_increase=1e-3, max_priority=1, min_priority=0.01, clip_norm=0.5
                 ):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.units_num = units_num
        self.layers_num = layers_num
        self.activation = activation
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq
        self.update_counter = 0
        self.clip_norm = clip_norm

        self.train_q = DQN_Q(agent_index=agent_index, state_shape=state_shape, action_shape=action_shape,
                             units_num=units_num, layers_num=layers_num, lr=lr, activation=self.activation,
                             clip_norm=self.clip_norm)
        self.target_q = DQN_Q(agent_index=agent_index, state_shape=state_shape, action_shape=action_shape,
                              units_num=units_num, layers_num=layers_num, lr=lr, activation=self.activation,
                              clip_norm=self.clip_norm)
        self.target_q.model.set_weights(self.train_q.model.get_weights())

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase
        self.prioritized_replay = prioritized_replay
        self.max_priority = max_priority
        self.min_priority = min_priority
        if self.prioritized_replay:
            self.replay_buffer = Prioritized_Replay_Buffer(buffer_size, self.alpha, self.beta, self.beta_increase,
                                                           self.max_priority, self.min_priority)
        else:
            self.replay_buffer = Replay_Buffer(buffer_size)

    def action(self, state):
        action = self.train_q.get_action(np.array([state]))[0]
        return action.numpy()

    def target_action(self, state):
        action = self.target_q.get_action(np.array([state]))[0]
        return action.numpy()

    def remember(self, state, action, next_state, reward, done):
        self.replay_buffer.remember(state, action, next_state, reward, done)

    def train(self):
        self.update_counter += 1
        if self.prioritized_replay:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch, index_batch, weight_batch = self.replay_buffer.sample(
                self.batch_size)
        else:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.replay_buffer.sample(
                self.batch_size)
            weight_batch = tf.ones(shape=[self.batch_size, ], dtype=tf.float32)
        q_batch = self.train_q.model(state_batch).numpy()
        next_q_batch = self.target_q.model(next_state_batch).numpy()
        next_q_batch = np.amax(next_q_batch, axis=1)
        for each in range(len(state_batch)):
            q_batch[each][int(action_batch[each])] = reward_batch[each] + self.gamma * next_q_batch[each] * (
                    1 - done_batch[each].astype(int))
        td_error_batch = self.train_q.train(state_batch, q_batch, weight_batch)
        if self.prioritized_replay:
            self.replay_buffer.batch_update(index_batch, np.sum(td_error_batch, axis=1))
        if self.update_counter % self.update_freq == 0:
            self.update_target_networks(self.tau)

    def update_target_networks(self, tau):
        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.train_q.model, self.target_q.model)

    def model_save(self, file_path, seed):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        self.target_q.model.save_weights(file_path + "/Agent_{}_Q_model.h5".format(self.agent_index))
        file = open(file_path + "/Agent_{}_train.log".format(self.agent_index), "w")
        file.write(
            "seed:" + str(seed) +
            "\nstate_shape:" + str(self.state_shape) +
            "\naction_shape:" + str(self.action_shape) +
            "\nunits_num:" + str(self.units_num) +
            "\nlayers_num:" + str(self.layers_num) +
            "\nactivation:" + str(self.activation) +
            "\nlr:" + str(self.lr) +
            "\ngamme:" + str(self.gamma) +
            "\ntau:" + str(self.tau) +
            "\nupdate_freq:" + str(self.update_freq) +
            "\nbatch_size:" + str(self.batch_size) +
            "\nbuffer_size:" + str(self.buffer_size) +
            "\nPER:" + str(self.prioritized_replay) +
            "\nalpha:" + str(self.alpha) +
            "\nbeta:" + str(self.beta) +
            "\nbeta_increase:" + str(self.beta_increase) +
            "\nmax_priority:" + str(self.max_priority) +
            "\nmin_priority:" + str(self.min_priority) +
            "\nclip_norm:" + str(self.clip_norm)
        )

    def model_load(self, file_path, agent_index=None):
        if agent_index == None:
            self.train_q.model.load_weights(file_path + "/Agent_{}_Q_model.h5".format(self.agent_index))
            self.target_q.model.load_weights(file_path + "/Agent_{}_Q_model.h5".format(self.agent_index))
        else:
            self.train_q.model.load_weights(file_path + "/Agent_{}_Q_model.h5".format(agent_index))
            self.target_q.model.load_weights(file_path + "/Agent_{}_Q_model.h5".format(agent_index))

    def buffer_save(self, file_path):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        self.replay_buffer.save(self.agent_index, file_path)

    def buffer_load(self, file_path, agent_index=None):
        if agent_index == None:
            self.replay_buffer.load(self.agent_index, file_path)
        else:
            self.replay_buffer.load(agent_index, file_path)


class DQN_Q():
    def __init__(self, agent_index, state_shape, action_shape, units_num, layers_num, lr, activation="linear",
                 clip_norm=0.5):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.units_num = units_num
        self.layers_num = layers_num
        self.activation = activation
        self.lr = lr
        self.clip_norm = clip_norm
        self.opt = keras.optimizers.Adam(self.lr)
        self.model = self.model_create()

    def model_create(self):
        # 创建输入端
        self.state_input_layer = keras.Input(shape=self.state_shape,
                                             name="Agent_{}_state_input".format(self.agent_index))
        # 创建中间层
        self.hidden_layers = []
        for each in range(self.layers_num):
            layer = keras.layers.Dense(self.units_num, activation="relu",
                                       name="Agent_{}_hidden_{}".format(self.agent_index, each))
            self.hidden_layers.append(layer)
        # 创建输出端
        self.action_output_layer = keras.layers.Dense(sum(self.action_shape), activation=self.activation,
                                                      name="Agent_{}_action_output".format(self.agent_index))

        x = self.hidden_layers[0](self.state_input_layer)
        for each in range(1, self.layers_num):
            x = self.hidden_layers[each](x)
        output = self.action_output_layer(x)
        # 创建模型
        model = keras.Model(inputs=self.state_input_layer, outputs=output)
        return model

    def get_action(self, state_batch):
        action_batch = self.model(state_batch)
        return action_batch

    def train(self, state_batch, target_q_batch, weight_batch):
        with tf.GradientTape() as tape:
            q_batch = self.model(state_batch)
            td_error_batch = tf.math.square(target_q_batch - q_batch)
            error = tf.reduce_mean(
                [tf.reduce_sum(td_error * weight) for td_error, weight in zip(td_error_batch, weight_batch)])
        gradients = tape.gradient(error, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return td_error_batch