import gym
import random
import threading
import time
import numpy as np
from keras import backend as K
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class Brain:
    def __init__(self, state_size, action_size):
        self.lock_memory = threading.Lock()
        self.memory = deque(maxlen=10000)
        self.train_start = 1000
        self.batch_size = 32

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.01
        self.loss_value_ratio = 0.5
        self.loss_entropy_ratio = 0.01

        self.actor, self.critic = self.build_model()
        self.actor_optimizer = self.actor_optimizer()

    def build_model(self):
        # actor network
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(24, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        # critic network
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer="he_uniform"))
        critic.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_learning_rate))
        # summary actor and critic networks
        actor.summary()
        critic.summary()
        return actor, critic

    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        good_prob = K.sum(action * self.actor.output, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        optimizer = Adam(lr=self.actor_learning_rate)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        with self.lock_memory:
            mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size))
        update_action = np.zeros((self.batch_size, self.action_size))
        update_target = np.zeros((self.batch_size, 1))
        advantages = np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            value = self.critic.predict(state)[0]

            #
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * \
                                  self.critic.predict(next_state)[0]
            update_input[i] = state
            update_action[i] = action
            update_target[i] = target
            # calculate advantage function(Q function - value function)
            advantages[i] = target - value

        self.critic.fit(update_input, update_target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.actor_optimizer([update_input, update_action, advantages])


class A3CAgent(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.render = False
        self.stop_signal = False

        self.env = gym.make('CartPole-v1')

        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]

    def run_episode(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        score = 0

        while True:
            time.sleep(0.001)
            if self.render:
                self.env.render()

            action = brain.get_action(state)
            next_state, reward, done, info = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])

            self.replay_memory(state, action, reward, next_state, done)
            brain.train_replay()

            if done or self.stop_signal:
                break
            score += reward
            state = next_state

        print("score:", score)

    def replay_memory(self, state, action, reward, next_state, done):
        act = np.zeros(self.action_size)
        act[action] = 1
        brain.memory.append((state, act, reward, next_state, done))

    def run(self):
        while not self.stop_signal:
            self.run_episode()

    def stop(self):
        self.stop_signal = True


if __name__ == "__main__":
    env_test = gym.make('CartPole-v1')
    state_size = env_test.observation_space.shape[0]
    action_size = env_test.action_space.n

    global_step = 0
    brain = Brain(state_size, action_size)

    actors = [A3CAgent() for i in range(1)]

    for actor in actors:
        actor.start()

    time.sleep(30)

    for actor in actors:
        actor.stop()

    for actor in actors:
        actor.join()

    print("Training finished")
    env_test.run()