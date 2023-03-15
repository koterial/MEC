import os
import numpy as np
import tensorflow as tf
from Agent.Replay_Buffer.Replay_Buffer import Replay_Buffer, Prioritized_Replay_Buffer
from Agent.Utils.Common import clip_by_local_norm
from Agent.DDPG.DDPG import DDPG_Agent, DDPG_Critic, DDPG_Actor

tf.keras.backend.set_floatx('float32')


class MA_TD3_Agent(DDPG_Agent):
    def __init__(self, agent_index, state_shape, action_n_shape, critic_units_num, critic_layers_num, critic_lr,
                 actor_units_num, actor_layers_num, actor_lr, batch_size, buffer_size, gamma, tau, update_freq=2,
                 activation="linear", prioritized_replay=False, alpha=0.6, beta=0.4, beta_increase=1e-3, max_priority=1,
                 min_priority=0.01, clip_norm=0.5
                 ):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_n_shape = action_n_shape
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

        self.train_critic_1 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                          action_shape=sum(self.action_n_shape),
                                          units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                          lr=self.critic_lr,
                                          clip_norm=self.clip_norm)
        self.target_critic_1 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                           action_shape=sum(self.action_n_shape),
                                           units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                           lr=self.critic_lr,
                                           clip_norm=self.clip_norm)
        self.target_critic_1.model.set_weights(self.train_critic_1.model.get_weights())

        self.train_critic_2 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                          action_shape=sum(self.action_n_shape),
                                          units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                          lr=self.critic_lr,
                                          clip_norm=self.clip_norm)
        self.target_critic_2 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                           action_shape=sum(self.action_n_shape),
                                           units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                           lr=self.critic_lr,
                                           clip_norm=self.clip_norm)
        self.target_critic_2.model.set_weights(self.train_critic_2.model.get_weights())

        self.train_actor_list = []
        self.target_actor_list = []
        for actor_index, action_shape in enumerate(self.action_n_shape):
            train_actor = MA_TD3_Actor(agent_index=self.agent_index, actor_index=actor_index, state_shape=self.state_shape,
                                     action_shape=action_shape,
                                     units_num=self.actor_units_num, layers_num=self.actor_layers_num,
                                     lr=self.actor_lr,
                                     critic=self.train_critic_1, activation=self.activation, clip_norm=self.clip_norm)
            target_actor = MA_TD3_Actor(agent_index=self.agent_index, actor_index=actor_index, state_shape=self.state_shape,
                                      action_shape=action_shape,
                                      units_num=self.actor_units_num, layers_num=self.actor_layers_num,
                                      lr=self.actor_lr,
                                      critic=self.train_critic_1, activation=self.activation, clip_norm=self.clip_norm)
            target_actor.model.set_weights(train_actor.model.get_weights())
            self.train_actor_list.append(train_actor)
            self.target_actor_list.append(target_actor)

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

    def action(self, state_batch):
        action = []
        for actor in self.train_actor_list:
            action.append(actor.get_action(state_batch)[0].numpy())
        action = np.concatenate(action, axis=0)
        return action

    def target_action(self, state_batch):
        action = []
        for actor in self.target_actor_list:
            action.append(actor.get_action(state_batch)[0].numpy())
        action = np.concatenate(action, axis=0)
        return action

    def update_target_networks(self, tau):
        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.train_critic_1.model, self.target_critic_1.model)
        update_target_network(self.train_critic_2.model, self.target_critic_2.model)
        for train_actor, target_actor in zip(self.train_actor_list, self.target_actor_list):
            update_target_network(train_actor.model, target_actor.model)

    def train(self):
        self.update_counter += 1
        if self.prioritized_replay:
            state_batch, action_n_batch, next_state_batch, reward_batch, done_batch, index_batch, weight_batch = self.replay_buffer.sample(
                self.batch_size)
        else:
            state_batch, action_n_batch, next_state_batch, reward_batch, done_batch = self.replay_buffer.sample(
                self.batch_size)
            weight_batch = tf.ones(shape=[self.batch_size, ], dtype=tf.float32)
        next_action_n_batch = []
        for actor in self.target_actor_list:
            next_action_n_batch.append(actor.get_action(next_state_batch).numpy())
        action_n_batch = action_n_batch.reshape((len(action_n_batch), len(self.action_n_shape), -1)).transpose(1, 0, 2)
        new_action_n_batch = []
        for each in range(len(self.action_n_shape)):
            new_action_n_batch.append(action_n_batch[each, :, :])
        action_n_batch = new_action_n_batch
        del new_action_n_batch
        next_action_n_batch = np.concatenate(next_action_n_batch, axis=1)
        next_q_batch = np.empty([2, self.batch_size], dtype=np.float32)
        next_q_batch[0] = self.target_critic_1.model([next_state_batch] + [next_action_n_batch]).numpy()[:, 0]
        next_q_batch[1] = self.target_critic_2.model([next_state_batch] + [next_action_n_batch]).numpy()[:, 0]
        next_q_batch = np.min(next_q_batch, 0)[:, None]
        target_q_batch = reward_batch[:, None] + self.gamma * next_q_batch * (1 - done_batch[:, None].astype(int))
        td_error_batch_1 = self.train_critic_1.train(state_batch, np.concatenate(action_n_batch, axis=1), target_q_batch, weight_batch)
        td_error_batch_2 = self.train_critic_2.train(state_batch, np.concatenate(action_n_batch, axis=1), target_q_batch, weight_batch)
        if self.prioritized_replay:
            self.replay_buffer.batch_update(index_batch, np.sum(td_error_batch_1 + td_error_batch_2, axis=1))
        if self.update_counter % self.update_freq == 0:
            new_action_n_batch = []
            for actor in self.target_actor_list:
                new_action_n_batch.append(actor.get_action(state_batch).numpy())
            for actor in self.train_actor_list:
                actor.train(state_batch, new_action_n_batch)
            self.update_target_networks(self.tau)

    def model_save(self, file_path):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        self.target_critic_1.model.save_weights(file_path + "/Agent{}_Critic_1_model.h5".format(self.agent_index))
        self.target_critic_2.model.save_weights(file_path + "/Agent{}_Critic_2_model.h5".format(self.agent_index))
        for index, actor in enumerate(self.target_actor_list):
            actor.model.save_weights(file_path + "/Agent{}_Actor{}_model.h5".format(self.agent_index, actor.actor_index))
        file = open(file_path + "/Agent{}_train.log".format(self.agent_index), "w")
        file.write(
            "state_shape:" + str(self.state_shape) +
            "\naction_n_shape:" + str(self.action_n_shape) +
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
            "\nclip_norm:" + str(self.clip_norm)
        )

    def model_load(self, file_path):
        self.target_critic_1.model.load_weights(file_path + "/Agent{}_Critic_1_model.h5".format(self.agent_index))
        self.train_critic_1.model.load_weights(file_path + "/Agent{}_Critic_1_model.h5".format(self.agent_index))
        self.target_critic_2.model.load_weights(file_path + "/Agent{}_Critic_2_model.h5".format(self.agent_index))
        self.train_critic_2.model.load_weights(file_path + "/Agent{}_Critic_2_model.h5".format(self.agent_index))
        for index, actor in enumerate(self.target_actor_list):
            actor.model.load_weights(file_path + "/Agent{}_Actor{}_model.h5".format(self.agent_index, actor.actor_index))
        for index, actor in enumerate(self.train_actor_list):
            actor.model.load_weights(file_path + "/Agent{}_Actor{}_model.h5".format(self.agent_index, actor.actor_index))

class MA_TD3_Actor(DDPG_Actor):
    def __init__(self, agent_index, actor_index, state_shape, action_shape, units_num, layers_num, lr, critic, activation="linear",
                 clip_norm=0.5):
        super(MA_TD3_Actor, self).__init__(agent_index, state_shape, action_shape, units_num, layers_num, lr, critic, activation, clip_norm)
        self.actor_index = actor_index
    
    def train(self, state_batch, action_n_batch):
        with tf.GradientTape() as tape:
            new_action_batch = self.model(state_batch)
            action_n_batch[self.actor_index] = new_action_batch
            q_batch = self.critic.model([state_batch] + [np.concatenate(action_n_batch, axis=1)])
            policy_regularization = tf.math.reduce_mean(tf.math.square(new_action_batch))
            loss = -tf.math.reduce_mean(q_batch) + 1e-3 * policy_regularization
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))