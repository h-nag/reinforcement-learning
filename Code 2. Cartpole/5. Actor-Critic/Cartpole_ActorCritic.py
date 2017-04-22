import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 300


# This is Actor Critic agent for the Cartpole
# Actor Critic is Policy Gradient + Policy Iteration
# Actor network is policy network, Critic network is Q network
class ACAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.01
        self.batch_size = 32
        self.train_start = 1000
        self.memory = deque(maxlen=10000)
        # create actor and critic models
        self.actor, self.critic = self.build_model()
        # create optimizer for the actor model
        self.actor_optimizer = self.actor_optimizer()

    # actor -> (input : state), (output : probability of each action)
    # critic -> (input : state), (output : value for the state)
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

    # make loss function for actor network
    # [log(action probability) * return] will be input for the back propagation
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

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
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

        # updating critic network is similar with DQN update
        self.critic.fit(update_input, update_target, batch_size=self.batch_size, epochs=1, verbose=0)
        # with advantage, we can evaluate how good the action is
        # with the information of policy evaluation through advantage function, update the actor network
        self.actor_optimizer([update_input, update_action, advantages])

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # save sample <s, a ,r, s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        act = np.zeros(self.action_size)
        act[action] = 1
        self.memory.append((state, act, reward, next_state, done))

    def load_model(self, name):
        self.actor.load_weights(name)
        self.critic.load_weights(name)

    def save_model(self, name1, name2):
        self.actor.save_weights(name1)
        self.critic.save_weights(name2)


if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make Actor Critic agent
    agent = ACAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # agent.load_model("./save_model/Cartpole-Actor.h5", "./save_model/Cartpole-Critic.h5")

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_replay()

            score += reward
            state = next_state

            if done:
                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/Cartpole_ActorCritic.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory))

                # 지난 10 에피소드의 평균이 490 이상이면 학습을 멈춤
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # 50 에피소드마다 학습 모델을 저장
        # if e % 50 == 0:
        #     agent.save_model("./save_model/Cartpole_Actor.h5", "./save_model/Cartpole_Critic.h5")
