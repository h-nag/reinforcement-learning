import threading
import numpy as np
import gym
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

global episode


class A3CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.01
        self.hidden1, self.hidden2 = 24, 24

        self.threads = 8

        self.actor, self.critic = self.build_model()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        global episode
        episode = 0

    def build_model(self):
        state = Input(batch_shape=(None, self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform')(
            state)
        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)
        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor.predict(np.array([[0, 1, 0, 1]]))
        critic.predict(np.array([[0, 1, 0, 1]]))

        actor.summary()
        critic.summary()
        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None,))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    def train(self):
        agents = [Agent(self.actor, self.critic, self.optimizer) for _ in range(self.threads)]

        for agent in agents:
            agent.start()


class Agent(threading.Thread):
    def __init__(self, actor, critic, optimizer):
        threading.Thread.__init__(self)

        self.states = []
        self.rewards = []
        self.actions = []

        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.discount_factor = .99

    def run(self):
        global episode
        env = gym.make('CartPole-v1')
        while True:
            state = env.reset()
            score = 0
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                score += reward

                self.memory(state, action, reward)

                state = next_state

                if done:
                    episode += 1
                    print("episode: ", episode, "score : ", score)
                    self.train_episode(score)
                    break

    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, 4)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def memory(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(2)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def train_episode(self, score):
        discounted_rewards = self.discount_rewards(self.rewards, score != 500)

        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[1]([self.states, discounted_rewards])
        self.optimizer[0]([self.states, self.actions, advantages])
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        policy = self.actor.predict(np.reshape(state, [1, 4]))[0]
        return np.random.choice(2, 1, p=policy)[0]


if __name__ == "__main__":
    env = gym.make('CartPole-v1')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    global_agent = A3CAgent(state_size, action_size)
    global_agent.train()
