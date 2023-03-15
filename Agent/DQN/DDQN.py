import numpy as np
import tensorflow as tf
from DQN.DQN import DQN_Agent

tf.keras.backend.set_floatx('float32')


class DDQN_Agent(DQN_Agent):
    def __init__(self, agent_index, state_shape, action_shape, units_num, layers_num, lr, batch_size, buffer_size,
                 gamma, tau, update_freq=1, activation="linear", prioritized_replay=False,
                 alpha=0.6, beta=0.4, beta_increase=1e-3, max_priority=1, min_priority=0.01, clip_norm=0.5
                 ):
        super(DDQN_Agent, self).__init__(agent_index, state_shape, action_shape, units_num, layers_num, lr, batch_size,
                                         buffer_size,
                                         gamma, tau, update_freq, activation, prioritized_replay,
                                         alpha, beta, beta_increase, max_priority, min_priority, clip_norm)

    def train(self):
        self.update_counter += 1
        if self.prioritized_replay:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch, index_batch, weight_batch = self.replay_buffer.sample(self.batch_size)
        else:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.replay_buffer.sample(self.batch_size)
            weight_batch = tf.ones(shape=[self.batch_size, ], dtype=tf.float32)
        q_batch = self.train_q.model(state_batch).numpy()
        next_q_batch = self.train_q.model(next_state_batch).numpy()
        next_action_batch = np.argmax(next_q_batch, axis=1)
        target_next_q_batch = self.target_q.model(next_state_batch).numpy()
        for each in range(len(state_batch)):
            q_batch[each][int(action_batch[each])] = reward_batch[each] + self.gamma * target_next_q_batch[each][next_action_batch[each]] * (1 - done_batch[each].astype(int))
        td_error_batch = self.train_q.train(state_batch, q_batch, weight_batch)
        if self.prioritized_replay:
            self.replay_buffer.batch_update(index_batch, np.sum(td_error_batch, axis=1))
        if self.update_counter % self.update_freq == 0:
            self.update_target_networks(self.tau)
