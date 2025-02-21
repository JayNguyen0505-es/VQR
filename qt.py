import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
from tensorflow import keras
from keras import layers
from cirq.contrib.svg import SVGCircuit
import logging
from env import Environment, Officers, Events


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

        self.start_learning = 1000
        self.batch_size = 1024

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
            priorities = self.priorities[: self.size]

        scaled_priorities = priorities**self.alpha
        sample_prob = scaled_priorities / np.sum(scaled_priorities)

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
# NOISY DENSE LAYER
# ---------------------------------------------------------------------------------
class NoisyDense(layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        kernel_initializer=None,
        kernel_regularizer=None,
        bias_initializer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        limit = 1 / np.sqrt(input_dim)

        kernel_init = (
            self.kernel_initializer
            if self.kernel_initializer is not None
            else tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)
        )
        bias_init = (
            self.bias_initializer
            if self.bias_initializer is not None
            else tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)
        )

        self.w_mu = self.add_weight(
            name="w_mu",
            shape=(input_dim, self.units),
            initializer=kernel_init,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        self.w_sigma = self.add_weight(
            name="w_sigma",
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.Constant(0.017),
            trainable=True,
        )
        self.b_mu = self.add_weight(
            name="b_mu",
            shape=(self.units,),
            initializer=bias_init,
            regularizer=self.bias_regularizer,
            trainable=True,
        )
        self.b_sigma = self.add_weight(
            name="b_sigma",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(0.017),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Generate noise for weights and biases.
        epsilon_w = tf.random.normal(shape=(inputs.shape[-1], self.units))
        epsilon_b = tf.random.normal(shape=(self.units,))
        # Compute noisy weights and biases.
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
    """
    x = NoisyDense(512, activation="relu")(inputs)
    x = NoisyDense(num_actions * num_atoms)(x)
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
    val = NoisyDense(128, activation="relu")(inputs)
    val = NoisyDense(num_atoms)(val)  # shape (batch, num_atoms)
    val = layers.Reshape((1, num_atoms))(val)  # shape (batch, 1, num_atoms)

    # Advantage stream
    adv = NoisyDense(128, activation="relu")(inputs)
    adv = NoisyDense(num_actions * num_atoms)(adv)
    adv = layers.Reshape((num_actions, num_atoms))(adv)

    # Then combine: Q(a) = V + (Adv(a) - mean(Adv(a)))
    # But for distribution, we do this for each atom distribution:
    adv_mean = tf.reduce_mean(adv, axis=1, keepdims=True)  # shape (batch, 1, num_atoms)
    adv_centered = adv - adv_mean

    out = val + adv_centered
    dist = layers.Softmax(axis=-1)(out)
    return dist


# ---------------------------------------------------------------------------------
# QUANTUM LAYER
# ---------------------------------------------------------------------------------
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Here we assume a linear array or 1D grid of qubits
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.symbols = sympy.symbols(f"theta0:{2 * n_qubits * n_layers}")
        self.circuit = self.build_circuit()
        self.circuit_tensor = tfq.convert_to_tensor([self.circuit])
        self.readout_ops = [cirq.Z(q) for q in self.qubits]
        self.expectation_layer = tfq.layers.Expectation()

    def build_circuit(self):
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*self.qubits))
        for layer in range(self.n_layers):
            for i, qubit in enumerate(self.qubits):
                rx_index = layer * (2 * self.n_qubits) + i
                rz_index = layer * (2 * self.n_qubits) + self.n_qubits + i
                circuit.append(cirq.rx(self.symbols[rx_index])(qubit))
                circuit.append(cirq.rz(self.symbols[rz_index])(qubit))

            # Simple ring connectivity for CNOTs
            for i in range(self.n_qubits):
                next_i = (i + 1) % self.n_qubits
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[next_i]))
        return circuit

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Repeat the single reference circuit for each example in the batch
        tiled_circuits = tf.repeat(self.circuit_tensor, repeats=batch_size, axis=0)

        # inputs is shaped (batch_size, 2*n_qubits*n_layers)
        # We pass these values in as symbol_values for parameterized gates
        symbol_values = tf.reshape(
            inputs, (batch_size, 2 * self.n_qubits * self.n_layers)
        )

        # Expectation of Z on each qubit
        return self.expectation_layer(
            tiled_circuits,
            symbol_names=[str(s) for s in self.symbols],
            symbol_values=symbol_values,
            operators=self.readout_ops,
        )

    def visualize_circuit(self):
        return SVGCircuit(self.circuit)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers,
            }
        )
        return config

    def get_circuit_info(self):
        return {
            "number_of_qubits": self.n_qubits,
            "number_of_layers": self.n_layers,
            "number_of_parameters": len(self.symbols),
            "circuit_depth": len(self.circuit),
            "circuit_moments": len(list(self.circuit.moments)),
            "operation_types": set(
                op.gate.__class__.__name__ for op in self.circuit.all_operations()
            ),
        }


# ---------------------------------------------------------------------------------
# Quantum Rainbow DQN
# ---------------------------------------------------------------------------------
class RainbowDQN:
    def __init__(
        self,
        officers,
        events,
        gamma=0.99,
        lr=1e-4,
        num_atoms=51,
        v_min=-1.0,
        v_max=0.0,
        n_step=3,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        epsilon=1.0,
        eps_decay=0.998,
        epsilon_min=0.05,
        update_target_interval=1000,
        use_dueling=True,
        n_qubits=16,
        n_layers_q=3,
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
        self.atoms = tf.cast(
            tf.linspace(self.v_min, self.v_max, self.num_atoms), dtype=tf.float32
        )

        self.n_step = n_step
        self.nstep_buffer = NStepBuffer(n_step, gamma)

        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.epsilon_min = epsilon_min

        self.memory = Memory_Prioritized(
            capacity=100000, state_size=self.state_size, alpha=alpha
        )
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames

        # Store quantum parameters
        self.n_qubits = n_qubits
        self.n_layers_q = n_layers_q

        # Build networks
        self.main_network = self._build_network_with_quantum(lr, use_dueling)
        self.target_network = self._build_network_with_quantum(lr, use_dueling)
        self.update_target()

        self.train_step_count = 0
        self.update_target_interval = update_target_interval
        self.batch_size = 64
        self.losses = []

        self.max_time = (
            np.max(np.array(officers.capability)) * events.num_task * events.num_event
            + np.max(np.array(events.transition_matrix))
            * events.num_task
            * events.num_event
        )

    # ----------------------------------------------------------------
    # Your new network builder with quantum layers
    # ----------------------------------------------------------------
    def _build_network_with_quantum(self, lr, use_dueling):
        """
        Example:
        Build a distributional Q-network (C51 + dueling if desired) with:
          - 2 or 3 noisy dense layers (classical)
          - A quantum circuit (parameterized by n_qubits, n_layers_q)
          - Then final classical layers / dueling distributional head
        """
        inputs = layers.Input(shape=(self.state_size,), name="state_input")

        x = NoisyDense(256, activation="relu")(inputs)
        x = NoisyDense(256, activation="relu")(x)

        # Quantum circuit
        encode = layers.Dense(
            2 * self.n_qubits * self.n_layers_q, activation="tanh", name="encoding"
        )(x)

        # Quantum processing
        q_out = QuantumLayer(self.n_qubits, self.n_layers_q, name="quantum_layer")(
            encode
        )

        # Merge classical + quantum
        merged = layers.Concatenate(name="merged_classical_quantum")([x, q_out])

        post = NoisyDense(256, activation="relu", name="post_merge_dense")(merged)

        if use_dueling:
            dist = create_dueling_distributional_head(
                post, self.num_actions, self.num_atoms
            )
        else:
            dist = create_distributional_head(post, self.num_actions, self.num_atoms)

        model = keras.Model(inputs=inputs, outputs=dist, name="RainbowDQN_Quantum")
        model.compile(
            optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
            loss="categorical_crossentropy",
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

        old_makespan = 0.0  # Initial makespan
        partial_assignment = []

        for i in range(num_tasks):
            current_state = np.concatenate([obs, action_table])

            # Epsilon-greedy action selection
            if np.random.rand() < self.epsilon:
                action_id = np.random.randint(0, self.num_actions)
            else:
                state_tf = tf.convert_to_tensor(
                    current_state[None, :], dtype=tf.float32
                )
                dist = self.main_network(state_tf, training=False)
                q_values = self.get_q_values_from_distribution(dist).numpy()[0]
                action_id = np.argmax(q_values)

            # Assign the chosen officer
            action_table[i] = action_id

            # Build next_state right away
            next_state = np.concatenate([obs, action_table])
            done = 1 if i == (num_tasks - 1) else 0

            # Compute partial makespan with the new assignment
            partial_assignment.append(action_id)
            new_makespan = self.env.partial_step(partial_assignment)
            # The incremental reward is the change in makespan (if it goes up, negative reward,
            # if it stays the same, zero, etc.). For example:
            incremental_reward = -(new_makespan - old_makespan) / self.max_time

            # Update old_makespan
            old_makespan = new_makespan

            # Store it in the n-step buffer
            self.nstep_buffer.store(
                state=current_state,
                action=action_id,
                reward=incremental_reward,
                next_state=next_state,
                done=bool(done),
            )

            # Try to pop from the buffer for training
            if self.nstep_buffer.is_ready():
                s0, a0, R, ns_n, done_n = self.nstep_buffer.pop()
                self.memory.add_experience(
                    state=s0,
                    action=a0,
                    reward=R,
                    next_state=ns_n,
                    terminal=int(done_n),
                    priority=1.0,
                )
                self.train()

        final_time = self.env.step(
            action_table.reshape(self.events.num_event, self.events.num_task)
        )
        final_reward = -np.max(final_time) / self.max_time

        # Now flush out the remaining transitions in the n-step buffer
        while self.nstep_buffer.is_ready():
            s0, a0, R, ns_n, done_n = self.nstep_buffer.pop()
            # Mix in the final reward
            R += final_reward
            self.memory.add_experience(
                state=s0,
                action=a0,
                reward=R,
                next_state=ns_n,
                terminal=int(done_n),
                priority=1.0,
            )
            self.train()

        self.nstep_buffer.clear()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.eps_decay)
        return final_reward

    def run_episode(self):
        state = self.env.reset()
        self.max_time = (
            np.max(np.array(self.officers.capability))
            * self.events.num_task
            * self.events.num_event
            + np.max(np.array(self.events.transition_matrix))
            * self.events.num_task
            * self.events.num_event
        )
        episode_reward = self.play_one_episode_step_by_step(state)
        self.train()
        return episode_reward

    def train(self):
        if self.memory.size < self.memory.start_learning:
            return

        states, actions, rewards, next_states, terminals, indices, weights = (
            self.memory.sample(self.batch_size, beta=self.beta)
        )
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
                dist_next_target, tf.stack([batch_indices, next_best_action], axis=1)
            )

            Tz = tf.expand_dims(rewards_tf, 1) + self.gamma * tf.expand_dims(
                (1 - terminals_tf), 1
            ) * tf.reshape(self.atoms, (1, self.num_atoms))
            Tz = tf.clip_by_value(Tz, self.v_min, self.v_max)
            b = (Tz - self.v_min) / ((self.v_max - self.v_min) / (self.num_atoms - 1))
            l = tf.cast(tf.floor(b), tf.int64)
            u = tf.cast(tf.math.ceil(b), tf.int64)

            proj_dist = tf.zeros((self.batch_size, self.num_atoms), dtype=tf.float32)
            for j in range(self.num_atoms):
                lj = l[:, j]
                uj = u[:, j]
                pj = dist_target_chosen[:, j]

                w_u = b[:, j] - tf.cast(lj, tf.float32)
                w_l = tf.cast(uj, tf.float32) - b[:, j]

                indices_l = tf.stack([batch_indices, tf.cast(lj, tf.int64)], axis=1)
                updates_l = pj * w_l
                proj_dist = tf.tensor_scatter_nd_add(proj_dist, indices_l, updates_l)

                indices_u = tf.stack([batch_indices, tf.cast(uj, tf.int64)], axis=1)
                updates_u = pj * w_u
                proj_dist = tf.tensor_scatter_nd_add(proj_dist, indices_u, updates_u)

            dist_current_chosen = tf.gather_nd(
                dist_current,
                tf.stack([batch_indices, tf.cast(actions_tf, tf.int64)], axis=1),
            )
            # Normalize the distribution
            proj_dist = proj_dist / tf.reduce_sum(proj_dist, axis=1, keepdims=True)
            # proj_dist = tf.clip_by_value(proj_dist, 1e-8, 1.0)
            # proj_dist = proj_dist / tf.reduce_sum(proj_dist, axis=1, keepdims=True)
            cross_entropy = -tf.reduce_sum(
                proj_dist * tf.math.log(dist_current_chosen + 1e-8), axis=1
            )
            loss = tf.reduce_mean(weights_tf * cross_entropy)

        grads = tape.gradient(loss, self.main_network.trainable_variables)
        grads = [tf.clip_by_norm(g, 10.0) for g in grads]
        self.main_network.optimizer.apply_gradients(
            zip(grads, self.main_network.trainable_variables)
        )

        td_errors = cross_entropy
        new_priorities = td_errors.numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        self.losses.append(loss.numpy())
        self.train_step_count += 1

        if self.train_step_count % self.update_target_interval == 0:
            self.update_target()

    def train_for_episodes(self, num_episodes):
        rewards = []
        epsilons = []
        losses = []
        for ep in range(num_episodes):
            ep_reward = self.run_episode()
            self.env.reset()
            rewards.append(ep_reward)
            epsilons.append(self.epsilon)
            losses.append(self.losses)

            if ep % 10 == 0:
                avg_reward = np.mean(rewards[-50:])
                avg_epsilon = np.mean(epsilons[-50:])
                avg_losses = np.mean(losses[-50:])
                print(
                    f"Episode {ep}, Reward: {ep_reward:.4f}, Avg: {avg_reward:.4f}, Eps: {avg_epsilon:.2f}, Loss: {avg_losses:.4f}"
                )
        return rewards, epsilons, avg_losses


if __name__ == "__main__":
    NUM_OFFICERS = 3
    NUM_EVENTS = 2
    NUM_TASKS = 2
    EPISODES = 50000

    env = Environment(
        Officers(NUM_OFFICERS, NUM_EVENTS, NUM_TASKS), Events(NUM_EVENTS, NUM_TASKS)
    )
    rainbow_agent = RainbowDQN(
        officers=env.officers,
        events=env.events,
        gamma=0.99,
        lr=1e-5,
        num_atoms=51,
        v_min=-0.5,
        v_max=0,
        n_step=5,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        epsilon=1.0,
        eps_decay=0.9995,
        epsilon_min=0.1,
        update_target_interval=1000,
        use_dueling=True,
        n_qubits=16,
        n_layers_q=3,
    )

    results, epsilons, losses = rainbow_agent.train_for_episodes(num_episodes=EPISODES)
    # Average reward per 1000 episodes
    avg_rewards = [np.mean(results[i : i + 1000]) for i in range(0, len(results), 1000)]
    avg_epsilons = [
        np.mean(epsilons[i : i + 1000]) for i in range(0, len(epsilons), 1000)
    ]
    avg_losses = [np.mean(losses[i : i + 1000]) for i in range(0, len(losses), 1000)]
    print("Average reward per 1000 episodes:", avg_rewards)
    print("Average epsilon per 1000 episodes:", avg_epsilons)
    print("Average loss per 1000 episodes:", avg_losses)
    print("Training complete!")
    print("Final returns:", results[-10:])
    print("Average Return of Last 100 Episodes:", np.mean(results[-100:]))
