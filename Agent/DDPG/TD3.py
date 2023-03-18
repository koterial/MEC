import os
import numpy as np
import tensorflow as tf
from Agent.Replay_Buffer.Replay_Buffer import Replay_Buffer, Prioritized_Replay_Buffer
from Agent.DDPG.DDPG import DDPG_Agent, DDPG_Critic, DDPG_Actor

tf.keras.backend.set_floatx('float32')


class TD3_Agent(DDPG_Agent):
    def __init__(self, agent_index, state_shape, action_shape, critic_units_num, critic_layers_num, critic_lr,
                 actor_units_num, actor_layers_num, actor_lr, eval_noise_scale, eval_noise_bound,
                 batch_size, buffer_size, gamma, tau, update_freq=2, activation="linear", prioritized_replay=False,
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
        self.eval_noise_scale = eval_noise_scale
        self.eval_noise_bound = eval_noise_bound
        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq
        self.update_counter = 0
        self.clip_norm = clip_norm

        self.train_critic_1 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                          action_shape=self.action_shape,
                                          units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                          lr=self.critic_lr,
                                          clip_norm=self.clip_norm)
        self.target_critic_1 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                           action_shape=self.action_shape,
                                           units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                           lr=self.critic_lr,
                                           clip_norm=self.clip_norm)
        self.target_critic_1.model.set_weights(self.train_critic_1.model.get_weights())

        self.train_critic_2 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                          action_shape=self.action_shape,
                                          units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                          lr=self.critic_lr,
                                          clip_norm=self.clip_norm)
        self.target_critic_2 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                           action_shape=self.action_shape,
                                           units_num=self.critic_units_num, layers_num=self.critic_layers_num,
                                           lr=self.critic_lr,
                                           clip_norm=self.clip_norm)
        self.target_critic_2.model.set_weights(self.train_critic_2.model.get_weights())

        self.train_actor = DDPG_Actor(agent_index=self.agent_index, state_shape=self.state_shape,
                                      action_shape=self.action_shape,
                                      units_num=self.actor_units_num, layers_num=self.actor_layers_num,
                                      lr=self.actor_lr,
                                      critic=self.train_critic_1, activation=self.activation, clip_norm=self.clip_norm)
        self.target_actor = DDPG_Actor(agent_index=self.agent_index, state_shape=self.state_shape,
                                       action_shape=self.action_shape,
                                       units_num=self.actor_units_num, layers_num=self.actor_layers_num,
                                       lr=self.actor_lr,
                                       critic=self.train_critic_1, activation=self.activation, clip_norm=self.clip_norm)
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

    def update_target_networks(self, tau):
        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.train_critic_1.model, self.target_critic_1.model)
        update_target_network(self.train_critic_2.model, self.target_critic_2.model)
        update_target_network(self.train_actor.model, self.target_actor.model)

    def train(self):
        self.update_counter += 1
        if self.prioritized_replay:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch, index_batch, weight_batch = self.replay_buffer.sample(
                self.batch_size)
        else:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.replay_buffer.sample(
                self.batch_size)
            weight_batch = tf.ones(shape=[self.batch_size, ], dtype=tf.float32)
        next_action_batch = self.target_actor.get_action(next_state_batch).numpy()
        noise = np.random.normal(loc=0.0, scale=self.eval_noise_scale, size=next_action_batch.shape)
        noise = np.clip(noise, -self.eval_noise_bound, self.eval_noise_bound)
        next_action_batch = noise + next_action_batch
        next_q_batch = np.empty([2, self.batch_size], dtype=np.float32)
        next_q_batch[0] = self.target_critic_1.model([next_state_batch] + [next_action_batch]).numpy()[:, 0]
        next_q_batch[1] = self.target_critic_2.model([next_state_batch] + [next_action_batch]).numpy()[:, 0]
        next_q_batch = np.min(next_q_batch, 0)[:, None]
        target_q_batch = reward_batch[:, None] + self.gamma * next_q_batch * (1 - done_batch[:, None].astype(int))
        td_error_batch_1 = self.train_critic_1.train(state_batch, action_batch, target_q_batch, weight_batch)
        td_error_batch_2 = self.train_critic_2.train(state_batch, action_batch, target_q_batch, weight_batch)
        if self.prioritized_replay:
            self.replay_buffer.batch_update(index_batch, np.sum((td_error_batch_1 + td_error_batch_2)/2, axis=1))
        if self.update_counter % self.update_freq == 0:
            self.train_actor.train(state_batch)
            self.update_target_networks(self.tau)

    def model_save(self, file_path, seed):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        self.target_critic_1.model.save_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
        self.target_critic_2.model.save_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(self.agent_index))
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
            "\neval_noise_scale:" + str(self.eval_noise_scale) +
            "\neval_noise_bound:" + str(self.eval_noise_bound) +
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
            self.target_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
            self.train_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
            self.target_critic_2.model.load_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(self.agent_index))
            self.train_critic_2.model.load_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(self.agent_index))
            self.target_actor.model.load_weights(file_path + "/Agent_{}_Actor_model.h5".format(self.agent_index))
            self.train_actor.model.load_weights(file_path + "/Agent_{}_Actor_model.h5".format(self.agent_index))
        else:
            self.target_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(agent_index))
            self.train_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(agent_index))
            self.target_critic_2.model.load_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(agent_index))
            self.train_critic_2.model.load_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(agent_index))
            self.target_actor.model.load_weights(file_path + "/Agent_{}_Actor_model.h5".format(agent_index))
            self.train_actor.model.load_weights(file_path + "/Agent_{}_Actor_model.h5".format(agent_index))
