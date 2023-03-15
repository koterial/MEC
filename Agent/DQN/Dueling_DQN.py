import tensorflow as tf
import tensorflow.keras as keras
from Agent.DQN.DQN import DQN_Agent, DQN_Q

tf.keras.backend.set_floatx('float32')

class Dueling_DQN_Agent(DQN_Agent):
    def __init__(self, agent_index, state_shape, action_shape, units_num, layers_num, lr, batch_size, buffer_size,
                 gamma, tau, update_freq=1, activation="linear", prioritized_replay=False,
                 alpha=0.6, beta=0.4, beta_increase=1e-3, max_priority=1, min_priority=0.01, clip_norm=0.5
                 ):
        super(Dueling_DQN_Agent, self).__init__(agent_index, state_shape, action_shape, units_num, layers_num, lr, batch_size,
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


class Dueling_DQN_Q(DQN_Q):
    def __init__(self, agent_index, state_shape, action_shape, units_num, layers_num, lr, activation="linear",
                 clip_norm=0.5):
        super(Dueling_DQN_Q, self).__init__(agent_index, state_shape, action_shape, units_num, layers_num, lr, activation,clip_norm)
        self.model = self.model_create()

    def model_create(self):
        # 创建输入端
        self.state_input_layer = keras.Input(shape=self.state_shape,
                                             name="Agent{}_state_input".format(self.agent_index))
        # 创建中间层
        self.hidden_layers = []
        for each in range(self.layers_num):
            layer = keras.layers.Dense(self.units_num, activation="relu",
                                       name="Agent{}_hidden{}".format(self.agent_index, each))
            self.hidden_layers.append(layer)
        # 创建输出端
        self.action_output_layer = keras.layers.Dense(self.action_shape, activation=self.activation,
                                                      name="Agent{}_action_output".format(self.agent_index))
        self.v_output_layer = keras.layers.Dense(1, activation="relu",
                                                      name="Agent{}_v_output".format(self.agent_index))
        self.duel_layer = Dueling_layer()

        x = self.hidden_layers[0](self.state_input_layer)
        for each in range(1, self.layers_num):
            x = self.hidden_layers[each](x)
        action_output = self.action_output_layer(x)
        v_output = self.v_output_layer(x)
        output = self.duel_layer(action_output, v_output)
        # 创建模型
        model = keras.Model(inputs=self.state_input_layer, outputs=output)
        return model

class Dueling_layer(tf.keras.layers.Layer):
    def call(self, action_input, v_input):
        q_output = v_input + (action_input - tf.reduce_mean(action_input, axis=1, keepdims=True))
        return q_output