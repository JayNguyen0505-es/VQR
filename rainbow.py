import tensorflow as tf
import numpy as np
import os
import datetime
import logging
from tensorflow import keras
from tensorflow.keras import layers
from env import Environment, Officers, Events  # <--- Same environment as before

logging.getLogger('matplotlib.font_manager').disabled = True


# ---------------------------------------------------------------------------------
# PRIORITIZED MEMORY
# ---------------------------------------------------------------------------------
class Memory_Prioritized:
    def __init__(self, capacity, state_size, alpha=0.6):
        self.capacity = capacity
        self.state_size = state_size
        self.alpha = alpha

        self.states = np.zeros((capacity, state_size), dtype=float)
        self.actions = np.zeros(capacity, dtype=int)
        self.rewards = np.zeros(capacity, dtype=float)
        self.next_states = np.zeros((capacity, state_size), dtype=float)
        self.terminal = np.zeros(capacity, dtype=int)

        self.priorities = np.zeros(capacity, dtype=float)

        self.start_learning = 15000
        self.batch_size = 128
        self.train_iteration = 1000

        self.insert_at = 0
        self.size = 0

    def add_experience(self, state, action, reward, next_state, terminal, priority=1.0):
        idx = self.insert_at
        self.states[idx, :] = state[:]
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx, :] = next_state[:]
        self.terminal[idx] = terminal
        self.priorities[idx] = priority

        if self.size < self.capacity:
            self.size += 1
        self.insert_at = (self.insert_at + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        # Make sure we have enough samples
        assert self.size >= batch_size
        if self.size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]

        scaled_priorities = priorities ** self.alpha
        sample_prob = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(self.size, batch_size, p=sample_prob)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        terminal = self.terminal[indices]

        total = self.size
        weights = (total * sample_prob[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return states, actions, rewards, next_states, terminal, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


# ---------------------------------------------------------------------------------
# N-STEP BUFFER
# ---------------------------------------------------------------------------------
class NStepBuffer:
    """
    A small buffer to accumulate n-step transitions before storing them in the main replay.
    """

    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []

    def store(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(done)

    def is_ready(self):
        return len(self.states) >= self.n_step

    def pop(self):
        """
        Pop the oldest transition with n-step return.
        Return: (state0, action0, R, next_state_n, done_n)
        """
        G = 0
        discount = 1
        for i in range(self.n_step):
            G += self.rewards[i] * discount
            discount *= self.gamma
            if self.terminals[i]:
                break

        s0 = self.states[0]
        a0 = self.actions[0]
        ns_n = self.next_states[self.n_step - 1]
        done_n = self.terminals[self.n_step - 1]

        # Remove the oldest step
        self.states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.next_states.pop(0)
        self.terminals.pop(0)

        return s0, a0, G, ns_n, done_n

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.terminals.clear()


# ---------------------------------------------------------------------------------
# NOISY DENSE (optional for Rainbow)
# ---------------------------------------------------------------------------------
class NoisyDense(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        limit = 1 / np.sqrt(input_dim)

        self.w_mu = self.add_weight(
            name='w_mu',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(-limit, limit),
            trainable=True
        )
        self.w_sigma = self.add_weight(
            name='w_sigma',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.Constant(0.017),
            trainable=True
        )

        self.b_mu = self.add_weight(
            name='b_mu',
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(-limit, limit),
            trainable=True
        )
        self.b_sigma = self.add_weight(
            name='b_sigma',
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(0.017),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        epsilon_w = tf.random.normal(shape=(inputs.shape[-1], self.units))
        epsilon_b = tf.random.normal(shape=(self.units,))

        w = self.w_mu + self.w_sigma * epsilon_w
        b = self.b_mu + self.b_sigma * epsilon_b

        output = tf.matmul(inputs, w) + b
        if self.activation is not None:
            output = self.activation(output)
        return output


# ---------------------------------------------------------------------------------
# Distributional Head (C51)
# ---------------------------------------------------------------------------------
def create_distributional_head(inputs, num_actions, num_atoms):
    """
    Returns a Tensor of shape (batch_size, num_actions, num_atoms).
    We'll apply a final softmax to get the probability distribution over the atoms.
    """
    # You can insert more layers if you like. For simplicity:
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(num_actions * num_atoms)(x)  # shape (batch_size, num_actions * num_atoms)
    x = layers.Reshape((num_actions, num_atoms))(x)
    dist = layers.Softmax(axis=-1)(x)
    return dist


# ---------------------------------------------------------------------------------
# (Optional) Dueling Architecture
# ---------------------------------------------------------------------------------
def create_dueling_distributional_head(inputs, num_actions, num_atoms):
    """
    Creates a dueling network. We have:
      - Value stream -> distribution of shape (batch, 1, num_atoms)
      - Advantage stream -> distribution of shape (batch, num_actions, num_atoms)
    Then combine them into final distribution (batch, num_actions, num_atoms).
    """
    # Value stream
    val = layers.Dense(512, activation='relu')(inputs)
    val = layers.Dense(num_atoms)(val)  # shape (batch, num_atoms)
    val = layers.Reshape((1, num_atoms))(val)  # shape (batch, 1, num_atoms)

    # Advantage stream
    adv = layers.Dense(512, activation='relu')(inputs)
    adv = layers.Dense(num_actions * num_atoms)(adv)  # shape (batch, num_actions * num_atoms)
    adv = layers.Reshape((num_actions, num_atoms))(adv)

    # Then combine: Q(a) = V + (Adv(a) - mean(Adv(a)))
    # But for distribution, we do this for each atom distribution:
    adv_mean = tf.reduce_mean(adv, axis=1, keepdims=True)  # shape (batch, 1, num_atoms)
    adv_centered = adv - adv_mean

    out = val + adv_centered
    dist = layers.Softmax(axis=-1)(out)
    return dist


# ---------------------------------------------------------------------------------
# Rainbow DQN (no quantum)
# ---------------------------------------------------------------------------------
class RainbowDQN:
    def __init__(
            self,
            officers,
            events,
            gamma=0.99,
            lr=1e-4,
            num_atoms=51,
            v_min=-100.0,
            v_max=0.0,
            n_step=3,
            alpha=0.6,  # PER alpha
            beta_start=0.4,  # PER beta
            beta_frames=100000,
            epsilon=1.0,
            eps_decay=0.998,
            epsilon_min=0.05,
            update_target_interval=1000,
            use_dueling=True
    ):
        self.officers = officers
        self.events = events
        self.env = Environment(officers, events)

        self.state_size = (
                officers.num_officer * events.num_task * events.num_event
                + events.occurence.size
                + events.transition_matrix.size
                + events.num_task * events.num_event
        )

        self.num_actions = officers.num_officer
        self.gamma = gamma
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = tf.cast(tf.linspace(self.v_min, self.v_max, self.num_atoms), dtype=tf.float32)

        self.n_step = n_step
        self.nstep_buffer = NStepBuffer(n_step, gamma)

        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.epsilon_min = epsilon_min

        self.memory = Memory_Prioritized(capacity=1000000, state_size=self.state_size, alpha=alpha)
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames

        self.main_network = self._build_network(lr, use_dueling)
        self.target_network = self._build_network(lr, use_dueling)
        self.update_target()

        self.train_step_count = 0
        self.update_target_interval = update_target_interval
        self.batch_size = 128
        self.losses = []

        self.max_time = np.max(np.array(officers.capability)) * events.num_task * events.num_event + \
                        np.max(np.array(events.transition_matrix)) * events.num_task * events.num_event

    def _build_network(self, lr, use_dueling):
        inputs = layers.Input(shape=(self.state_size,), name="state_input")

        # Deep Classical Front-End
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Add Residual Connections
        residual = layers.Dense(512, activation='relu')(x)
        residual = layers.Add()([x, residual])
        residual = layers.BatchNormalization()(residual)

        # Self-Attention Layer (Fix Shape Issue)
        reshaped_residual = layers.Reshape((1, 512))(residual)
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(reshaped_residual, reshaped_residual)
        attention_output = layers.Flatten()(attention_output)  # Flatten back to 2D

        # Add Residual Connection with Attention Output
        x = layers.Add()([residual, attention_output])
        x = layers.LayerNormalization()(x)

        if use_dueling:
            dist = create_dueling_distributional_head(x, self.num_actions, self.num_atoms)
        else:
            dist = create_distributional_head(x, self.num_actions, self.num_atoms)

        model = keras.Model(inputs=inputs, outputs=dist, name="RainbowDQN")
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss='categorical_crossentropy'
        )
        return model

    def update_target(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def get_q_values_from_distribution(self, dist):
        atoms_expanded = tf.reshape(self.atoms, (1, 1, self.num_atoms))
        q_values = tf.reduce_sum(dist * atoms_expanded, axis=2)
        return q_values

    def play_one_episode_step_by_step(self, obs):
        num_tasks = self.events.num_task * self.events.num_event
        action_table = np.full(num_tasks, -1, dtype=int)
        episode_memory = []

        for i in range(num_tasks):
            current_state = np.concatenate([obs, action_table])
            if np.random.rand() < self.epsilon:
                action_id = np.random.randint(0, self.num_actions)
            else:
                state_tf = tf.convert_to_tensor(current_state[None, :], dtype=tf.float32)
                dist = self.main_network(state_tf, training=False)
                q_values = self.get_q_values_from_distribution(dist).numpy()[0]
                action_id = np.argmax(q_values)

            action_table[i] = action_id
            next_state = np.concatenate([obs, action_table])
            done = 1 if i == (num_tasks - 1) else 0

            episode_memory.append((current_state, action_id, 0.0, next_state, done))

        # Use the old reward logic
        reward = -np.max(self.env.step(action_table.reshape(self.events.num_event, self.events.num_task)))
        # print(f"Reward: {reward:.2f}, Max Time: {self.max_time:.2f}")
        final_reward = reward / self.max_time

        # Update all rewards in episode_memory with the final reward
        for i in range(len(episode_memory)):
            s, a, _, ns, done = episode_memory[i]
            episode_memory[i] = (s, a, final_reward, ns, done)

        # Store in the main memory buffer
        for (s, a, r, ns, done) in episode_memory:
            self.memory.add_experience(
                state=s,
                action=a,
                reward=r,
                next_state=ns,
                terminal=done,
                priority=1.0
            )

        self.epsilon = max(self.epsilon_min, self.epsilon * self.eps_decay)
        return final_reward

    def run_episode(self):
        state = self.env.reset()
        self.max_time = np.max(np.array(self.officers.capability)) * self.events.num_task * self.events.num_event + \
                        np.max(np.array(self.events.transition_matrix)) * self.events.num_task * self.events.num_event
        episode_reward = self.play_one_episode_step_by_step(state)
        self.train()
        return episode_reward

    def train(self):
        if self.memory.size < self.memory.start_learning:
            return

        states, actions, rewards, next_states, terminals, indices, weights = self.memory.sample(self.batch_size,
                                                                                                beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        terminals_tf = tf.convert_to_tensor(terminals, dtype=tf.float32)
        weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)

        with tf.GradientTape() as tape:
            dist_current = self.main_network(states_tf, training=True)
            dist_next_main = self.main_network(next_states_tf, training=False)
            q_next_main = self.get_q_values_from_distribution(dist_next_main)
            next_best_action = tf.argmax(q_next_main, axis=1)

            dist_next_target = self.target_network(next_states_tf, training=False)
            batch_indices = tf.range(self.batch_size, dtype=tf.int64)
            dist_target_chosen = tf.gather_nd(
                dist_next_target,
                tf.stack([batch_indices, next_best_action], axis=1)
            )

            Tz = tf.expand_dims(rewards_tf, 1) + self.gamma * tf.expand_dims((1 - terminals_tf), 1) * tf.reshape(
                self.atoms, (1, self.num_atoms))
            Tz = tf.clip_by_value(Tz, self.v_min, self.v_max)
            b = (Tz - self.v_min) / ((self.v_max - self.v_min) / (self.num_atoms - 1))
            l = tf.cast(tf.floor(b), tf.int64)
            u = tf.cast(tf.math.ceil(b), tf.int64)

            proj_dist = tf.zeros((self.batch_size, self.num_atoms), dtype=tf.float32)
            for j in range(self.num_atoms):
                lj = l[:, j]
                uj = u[:, j]
                pj = dist_target_chosen[:, j]

                w_u = (b[:, j] - tf.cast(lj, tf.float32))
                w_l = (tf.cast(uj, tf.float32) - b[:, j])

                indices_l = tf.stack([batch_indices, tf.cast(lj, tf.int64)], axis=1)
                updates_l = pj * w_l
                proj_dist = tf.tensor_scatter_nd_add(proj_dist, indices_l, updates_l)

                indices_u = tf.stack([batch_indices, tf.cast(uj, tf.int64)], axis=1)
                updates_u = pj * w_u
                proj_dist = tf.tensor_scatter_nd_add(proj_dist, indices_u, updates_u)

            dist_current_chosen = tf.gather_nd(
                dist_current,
                tf.stack([batch_indices, tf.cast(actions_tf, tf.int64)], axis=1)
            )

            cross_entropy = -tf.reduce_sum(
                proj_dist * tf.math.log(dist_current_chosen + 1e-8), axis=1
            )
            loss = tf.reduce_mean(weights_tf * cross_entropy)

        grads = tape.gradient(loss, self.main_network.trainable_variables)
        grads = [tf.clip_by_norm(g, 10.0) for g in grads]
        self.main_network.optimizer.apply_gradients(zip(grads, self.main_network.trainable_variables))

        td_errors = cross_entropy
        new_priorities = td_errors.numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        self.losses.append(loss.numpy())
        self.train_step_count += 1

        if self.train_step_count % self.update_target_interval == 0:
            self.update_target()

    def train_for_episodes(self, num_episodes=1000):
        rewards = []
        for ep in range(num_episodes):
            ep_reward = self.run_episode()
            self.env.reset()
            rewards.append(ep_reward)

            if ep % 10 == 0:
                avg_reward = np.mean(rewards[-50:])
                print(f"Episode {ep}, Reward: {ep_reward:.2f}, Avg: {avg_reward:.2f}, Eps: {self.epsilon:.2f}")
        return rewards


# ---------------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    NUM_OFFICERS = 5
    NUM_EVENTS = 4
    NUM_TASKS = 4
    EPISODES = 50000

    env = Environment(Officers(NUM_OFFICERS, NUM_EVENTS, NUM_TASKS), Events(NUM_EVENTS, NUM_TASKS))
    rainbow_agent = RainbowDQN(
        officers=env.officers,
        events=env.events,
        gamma=0.99,
        lr=5e-5,
        num_atoms=51,
        v_min=-1,
        v_max=0,
        n_step=5,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        epsilon=1.0,
        eps_decay=0.99995,
        epsilon_min=0.05,
        update_target_interval=1000,
        use_dueling=True
    )

    results = rainbow_agent.train_for_episodes(num_episodes=EPISODES)
    # Average reward per 1000 episodes
    avg_rewards = [np.mean(results[i:i + 1000]) for i in range(0, len(results), 1000)]
    print("Average reward per 1000 episodes:", avg_rewards)
    print("Training complete!")
    print("Final returns:", results[-10:])
    print("Average Return of Last 100 Episodes:", np.mean(results[-100:]))
