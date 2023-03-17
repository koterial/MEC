import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from Agent.Utils.Common import clip_by_local_norm
from Agent.Replay_Buffer.Replay_Buffer import Replay_Buffer, Prioritized_Replay_Buffer

tf.keras.backend.set_floatx('float32')


class DDPG_Agent():
    def __init__(self, agent_index, state_shape, action_shape, critic_units_num, critic_layers_num, critic_lr,
                 actor_units_num, actor_layers_num, actor_lr,
                 batch_size, buffer_size, gamma, tau, update_freq=1, activation="linear", prioritized_replay=False,
                 alpha=0.6, beta=0.4, beta_increase=1e-3, max_priority=1, min_priority=0.01, clip_norm=0.5
                 ):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.critic_units_num = critic_units_num
        self.critic_layers_num = critic_layers_num
        self.critic_lr = critic_lr
        self.actor_units_num = actor_units_num
        self.actor_layers_num = actor_layers_num
        self.activation = activation
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq
        self.update_counter = 0
        self.clip_norm = clip_norm

        self.train_critic = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                        action_shape=self.action_shape,
                                        units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                        lr=self.critic_lr,
                                        clip_norm=self.clip_norm)
        self.target_critic = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                         action_shape=self.action_shape,
                                         units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                         lr=self.critic_lr,
                                         clip_norm=self.clip_norm)
        self.train_actor = DDPG_Actor(agent_index=self.agent_index, state_shape=self.state_shape,
                                      action_shape=self.action_shape,
                                      units_num=self.actor_units_num, layers_num=self.actor_layers_num,
                                      lr=self.actor_lr,
                                      critic=self.train_critic, activation=self.activation, clip_norm=self.clip_norm)
        self.target_actor = DDPG_Actor(agent_index=self.agent_index, state_shape=self.state_shape,
                                       action_shape=self.action_shape,
                                       units_num=self.actor_units_num, layers_num=self.actor_layers_num,
                                       lr=self.actor_lr,
                                       critic=self.train_critic, activation=self.activation, clip_norm=self.clip_norm)

        self.target_critic.model.set_weights(self.train_critic.model.get_weights())
        self.target_actor.model.set_weights(self.train_actor.model.get_weights())

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
        action = self.train_actor.get_action(np.array([state]))[0]
        return action.numpy()

    def target_action(self, state):
        action = self.target_actor.get_action(np.array([state]))[0]
        return action.numpy()

    def update_target_networks(self, tau):
        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.train_critic.model, self.target_critic.model)
        update_target_network(self.train_actor.model, self.target_actor.model)

    def train(self):
        self.update_counter += 1
        if self.prioritized_replay:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch, index_batch, weight_batch = self.replay_buffer.sample(
                self.batch_size)
        else:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.replay_buffer.sample(
                self.batch_size)
            weight_batch = tf.ones(shape=(self.batch_size,), dtype=tf.float32)
        next_action_batch = self.target_actor.get_action(next_state_batch)
        next_q_batch = self.target_critic.model([next_state_batch] + [next_action_batch])
        target_q_batch = reward_batch[:, None] + self.gamma * next_q_batch * (1 - done_batch[:, None].astype(int))
        td_error_batch = self.train_critic.train(state_batch, action_batch, target_q_batch, weight_batch)
        if self.prioritized_replay:
            self.replay_buffer.batch_update(index_batch, np.sum(td_error_batch, axis=1))
        if self.update_counter % self.update_freq == 0:
            self.train_actor.train(state_batch)
            self.update_target_networks(self.tau)

    def remember(self, state, action, next_state, reward, done):
        self.replay_buffer.remember(state, action, next_state, reward, done)

    def model_save(self, file_path, seed):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        self.target_critic.model.save_weights(file_path + "/Agent_{}_Critic_model.h5".format(self.agent_index))
        self.target_actor.model.save_weights(file_path + "/Agent_{}_Actor_model.h5".format(self.agent_index))
        file = open(file_path + "/Agent_{}_train.log".format(self.agent_index), "w")
        file.write(
            "seed:" + str(seed) +
            "\nstate_shape:" + str(self.state_shape) +
            "\naction_shape:" + str(self.action_shape) +
            "\ncritic_units_num:" + str(self.critic_units_num) +
            "\ncritic_layers_num:" + str(self.critic_layers_num) +
            "\ncritic_lr:" + str(self.critic_lr) +
            "\nactor_units_num:" + str(self.actor_units_num) +
            "\nactor_layers_num:" + str(self.actor_layers_num) +
            "\nactivation:" + str(self.activation) +
            "\nactor_lr:" + str(self.actor_lr) +
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
            self.target_critic.model.load_weights(file_path + "/Agent_{}_Critic_model.h5".format(self.agent_index))
            self.train_critic.model.load_weights(file_path + "/Agent_{}_Critic_model.h5".format(self.agent_index))
            self.target_actor.model.load_weights(file_path + "/Agent_{}_Actor_model.h5".format(self.agent_index))
            self.train_actor.model.load_weights(file_path + "/Agent_{}_Actor_model.h5".format(self.agent_index))
        else:
            self.target_critic.model.load_weights(file_path + "/Agent_{}_Critic_model.h5".format(agent_index))
            self.train_critic.model.load_weights(file_path + "/Agent_{}_Critic_model.h5".format(agent_index))
            self.target_actor.model.load_weights(file_path + "/Agent_{}_Actor_model.h5".format(agent_index))
            self.train_actor.model.load_weights(file_path + "/Agent_{}_Actor_model.h5".format(agent_index))


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

class DDPG_Critic():
    def __init__(self, agent_index, state_shape, action_shape, units_num, layers_num, lr, clip_norm=0.5):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.units_num = units_num
        self.layers_num = layers_num
        self.lr = lr
        self.clip_norm = clip_norm
        self.opt = keras.optimizers.Adam(self.lr)
        self.model = self.model_create()

    def model_create(self):
        # 创建状态输入端
        self.state_input_layers = [
            keras.Input(shape=self.state_shape, name="Agent_{}_critic_state_input".format(self.agent_index))
        ]
        # 创建动作输入端
        self.action_input_layers = [
            keras.Input(shape=sum(self.action_shape), name="Agent_{}_critic_action_input".format(self.agent_index))
        ]
        # 创建中间层
        self.hidden_layers = [
            keras.layers.Dense(self.units_num, activation="relu", name="Agent_{}_critic_hidden_{}".format(self.agent_index, each)) for each in range(self.layers_num)
        ]
        # 创建输出端
        self.output_layers = keras.layers.Dense(1, activation="linear", name="Agent_{}_critic_output".format(self.agent_index))
        # 创建链接层
        self.input_concat_layers = keras.layers.Concatenate()
        # 链接各层
        x = self.input_concat_layers(self.state_input_layers + self.action_input_layers)
        for each in range(self.layers_num):
            x = self.hidden_layers[each](x)
        output = self.output_layers(x)
        # 创建模型
        model = keras.Model(inputs=self.state_input_layers + self.action_input_layers, outputs=output)
        return model

    def train(self, state_batch, action_batch, target_q_batch, weight_batch):
        with tf.GradientTape() as tape:
            q_batch = self.model([state_batch] + [action_batch])
            td_error_batch = tf.math.square(target_q_batch - q_batch)
            error = tf.reduce_mean(
                [tf.reduce_sum(td_error * weight) for td_error, weight in zip(td_error_batch, weight_batch)])
        gradients = tape.gradient(error, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return td_error_batch


class DDPG_Actor():
    def __init__(self, agent_index, state_shape, action_shape, units_num, layers_num, lr, critic, activation="linear", clip_norm=0.5):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.units_num = units_num
        self.layers_num = layers_num
        self.activation = activation
        self.lr = lr
        self.critic = critic
        self.clip_norm = clip_norm
        self.opt = keras.optimizers.Adam(self.lr)
        self.model = self.model_create()

    def model_create(self):
        # 创建状态输入端
        self.state_input_layers = [
            keras.Input(shape=self.state_shape, name="Agent_{}_actor_input".format(self.agent_index))
        ]
        # 创建中间层
        self.hidden_layers = [
            keras.layers.Dense(self.units_num, activation="relu", name="Agent_{}_actor_hidden_{}".format(self.agent_index, each)) for each in range(self.layers_num)
        ]
        # 创建动作输出端
        self.action_output_layers = [
            keras.layers.Dense(shape, activation=self.activation, name="Agent_{}_actor_output_{}".format(self.agent_index, each)) for each, shape in enumerate(self.action_shape)
        ]
        # 创建链接层
        self.input_concat_layers = keras.layers.Concatenate()
        self.output_concat_layers = keras.layers.Concatenate()
        # 链接各层
        x = self.input_concat_layers(self.state_input_layers)
        for each in range(self.layers_num):
            x = self.hidden_layers[each](x)
        output = []
        for layers in self.action_output_layers:
            output.append(layers(x))
        output = self.output_concat_layers(output)
        # 创建模型
        model = keras.Model(inputs=self.state_input_layers, outputs=output)
        return model

    def get_action(self, state_batch):
        action_batch = self.model(state_batch)
        return action_batch

    def train(self, state_batch):
        with tf.GradientTape() as tape:
            new_action_batch = self.model(state_batch)
            action_batch = new_action_batch
            q_batch = self.critic.model([state_batch] + [action_batch])
            policy_regularization = tf.math.reduce_mean(tf.math.square(new_action_batch))
            loss = -tf.math.reduce_mean(q_batch) + 1e-3 * policy_regularization
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
