import gym
import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D

EPISODES = 50000


class DQNAgent:
    def __init__(self):
        # environment settings
        self.state_size = (84, 84, 4)
        self.action_size = 3
        # parameters about epsilon
        self.epsilon = 0.01
        # parameters about training
        self.discount_factor = 0.99
        # build model and target model. Then update target model with model
        self.model = self.build_model()
        self.model.load_weights("Breakout_DQN.h5")

    # use huber loss for the training. This improves learning process
    def huber_loss(self, target, prediction):
        error = tf.reduce_sum(target - prediction, reduction_indices=1)
        return tf.where(tf.abs(error) < 1, 0.5 * tf.square(error), tf.abs(error) - 0.5)

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        model.compile(loss=self.huber_loss, optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(history)[0])


# 210*160*3(color) --> 84*84(mono) and float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

if __name__ == "__main__":
    # In case of BreakoutDeterministic-v3, always skip 4 frames
    env = gym.make('BreakoutDeterministic-v3')
    agent = DQNAgent()

    for e in range(EPISODES):
        done = False
        observe = env.reset()
        fake_action = 0
        # At start of episode, there is no preceding frame. So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            # get action for the current history and go one step in environment
            action = agent.get_action(history)
            if action == 0: fake_action = 1
            if action == 1: fake_action = 4
            if action == 2: fake_action = 5
            observe, reward, done, info = env.step(fake_action)
            # pre-process the observation --> history
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(history[:, :, :, 1:], next_state, axis=3)
            history = next_history
