import tensorflow as tf
from Agent.DQN.DDQN import DDQN_Agent
from Agent.DQN.Dueling_DQN import Dueling_DQN_Q

tf.keras.backend.set_floatx('float32')

class D3QN_Agent(DDQN_Agent):
    def __init__(self, agent_index, state_shape, action_shape, units_num, layers_num, lr, batch_size, buffer_size,
                 gamma, tau, update_freq=1, activation="linear", prioritized_replay=False,
                 alpha=0.6, beta=0.4, beta_increase=1e-3, max_priority=1, min_priority=0.01, clip_norm=0.5
                 ):
        super(DDQN_Agent, self).__init__(agent_index, state_shape, action_shape, units_num, layers_num, lr, batch_size,
                                         buffer_size,
                                         gamma, tau, update_freq, activation, prioritized_replay,
                                         alpha, beta, beta_increase, max_priority, min_priority, clip_norm)

        self.train_q = Dueling_DQN_Q(agent_index=agent_index, state_shape=state_shape, action_shape=action_shape,
                             units_num=units_num, layers_num=layers_num, lr=lr, activation=self.activation,
                             clip_norm=self.clip_norm)
        self.target_q = Dueling_DQN_Q(agent_index=agent_index, state_shape=state_shape, action_shape=action_shape,
                              units_num=units_num, layers_num=layers_num, lr=lr, activation=self.activation,
                              clip_norm=self.clip_norm)
        self.target_q.model.set_weights(self.train_q.model.get_weights())
