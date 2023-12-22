import os
import random
import time
import numpy as np
from tensorflow.keras import layers, models
from collections import deque
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
from gym_slitherio.envs.slitherio_env import SlitherIOEnv


# Define the DQNNetwork class
@register_keras_serializable()
class DQNNetwork(models.Model):
    def __init__(self, num_actions):
        super(DQNNetwork, self).__init__()
        self.flatten = layers.Flatten(input_shape=(5,))
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(num_actions, activation='linear')
        self.num_actions = num_actions

    def call(self, state):
        x = self.flatten(state)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

    def compile_model(self):
        self.compile(optimizer='adam', loss='mse')

    def get_config(self):
        config = {
            'num_actions': self.num_actions
        }
        base_config = super(DQNNetwork, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Define the DQNAgent class
class DQNAgent:
    def __init__(self, num_actions, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.memory = deque(maxlen=10000)
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNNetwork(num_actions)
        self.target_model = DQNNetwork(num_actions)
        self.target_model.set_weights(self.model.get_weights())

        # Compile the model
        self.model.compile_model()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])

    def update_epsilon(self, episode, num_episodes):
        # Implement a custom decay strategy
        if episode <= 0:
            return  # Avoid division by zero

        decay_factor = (self.epsilon_min / self.epsilon) ** (1.0 / min(episode, num_episodes))
        self.epsilon = max(self.epsilon * decay_factor, self.epsilon_min)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = np.concatenate(states)
        next_states = np.concatenate(next_states)

        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        # Convert 'dones' to a NumPy array
        dones = np.array(dones, dtype=bool)

        targets[range(batch_size), np.array(actions).astype(int)] = rewards + \
                                                                    self.gamma * np.max(next_q_values, axis=1) * ~dones

        self.model.train_on_batch(states, targets)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * 0.01 + target_weights[i] * 0.99
        self.target_model.set_weights(target_weights)

    def save_model(self, filename):
        try:
            self.model.save(filename + '.keras')  # Save with .keras extension
            print("Pre-trained model saved!")
        except Exception as e:
            print("Error saving model:", str(e))

    def load_model(self, filename):
        try:
            self.model = keras.models.load_model(filename + '.keras')
            self.target_model = keras.models.load_model(filename + '.keras')
            print("Pre-trained model loaded!")
        except Exception as e:
            print("Error loading model:", str(e))


# Define the function to preprocess the state
def preprocess_state(state):
    # Extract relevant information from the state
    my_snake = state['my_snake']
    other_snakes = state['other_snakes']
    foods = state['foods']

    # Combine relevant information into a single state representation
    processed_state = {
        'my_snake': my_snake,
        'other_snakes': other_snakes,
        'foods': foods
    }

    return processed_state


def main():
    while True:
        latest_env = None
        try:
            # Create the Gym environment
            env = SlitherIOEnv()
            latest_env = env

            num_actions = env.action_space.n

            # Create the DQNAgent
            agent = DQNAgent(num_actions=num_actions)

            # Check if the model file exists
            model_filename = "slitherio_dqn_model"
            if os.path.exists(model_filename + '.keras'):
                # Load the pre-trained model
                agent.load_model(model_filename)

            # Training parameters
            num_episodes = 10
            batch_size = 32

            for episode in range(num_episodes):
                state = env.reset()
                state = preprocess_state(state)

                # Assuming 'my_snake' is a 1D array
                state_size = len(state['my_snake'])
                state = np.reshape(state['my_snake'], [1, state_size])

                total_reward = 0
                done = False

                # Update epsilon based on the episode number
                agent.update_epsilon(episode, num_episodes)

                while not done:
                    action = agent.act(state)

                    next_state, reward, done, _ = env.step(action)
                    next_state = preprocess_state(next_state)

                    next_state_size = len(next_state['my_snake'])
                    next_state = np.reshape(next_state['my_snake'], [1, next_state_size])

                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    # print("Rewards: {}".format(total_reward))

                    if done:
                        agent.target_train()
                        print("Episode: {}, Total Reward: {}".format(episode + 1, total_reward))

                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)

                # Decay epsilon
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

            # Save the trained model
            agent.save_model("slitherio_dqn_model")

            # Close the environment
            env.close()
        except Exception as e:
            print(f"An error occurred: {e}")
            # Close the latest environment
            latest_env.close()
            print("Restarting training in 30 seconds...")
            time.sleep(30)


if __name__ == "__main__":
    main()
