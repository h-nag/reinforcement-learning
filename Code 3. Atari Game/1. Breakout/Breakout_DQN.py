import gym
import pylab
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D

EPISODES = 50000


class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        # environment settings
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # parameters about epsilon
        self.train_interval = 4
        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) * self.train_interval / self.epsilon_decay

        self.batch_size = 32
        self.train_start = 20000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # optimizer parameters
        self.learning_rate = 0.00025
        self.momentum = 0.95
        self.min_gradient = 0.01
        # build
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.sess = tf.Session()
        self.avg_q_max = 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/BreakoutDeterministic-v3', self.sess.graph)

    # if the error is in the interval [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def huber_loss(self, target, prediction):
        error = target - prediction
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
        model.compile(loss=self.huber_loss, optimizer=RMSprop(
            lr=self.learning_rate, rho=self.momentum, epsilon=self.min_gradient))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(np.reshape([history], (1, 84, 84, 4)) / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, done):
        self.memory.append([list(history), action, reward, list(next_history), done])

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history, action, reward, next_history, done = [], [], [], [], []
        for data in mini_batch:
            history.append(data[0])
            action.append(data[1])
            reward.append(data[2])
            next_history.append(data[3])
            done.append(data[4])

        done = np.array(done) * 0

        update_target = self.model.predict(history)
        target = self.target_model.predict(next_history)
        target = reward + (1 - done) * self.discount_factor * np.amax(target, axis=1)

        for update in update_target:
            update[action[i]] = target[i]

        self.model.fit(history, update_target, batch_size=self.batch_size, epochs=1, verbose=0)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar('BreakoutDeterministic-v3/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar('BreakoutDeterministic-v3/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar('BreakoutDeterministic-v3/Duration/Episode', episode_duration)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # In case of BreakoutDeterministic-v3, always skip 4 frames
    env = gym.make('BreakoutDeterministic-v3')
    # get size of action from environment
    action_size = env.action_space.n
    agent = DQNAgent(action_size)

    scores, episodes, global_step, step = [], [], 0, 0
    agent.load_model("./save_model/Breakout_DQN.h5")

    for e in range(EPISODES):
        done = False
        dead = False
        # 1 episode = 5 lives
        score, start_life = 0, 5
        observe = env.reset()
        next_observe = observe

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe = next_observe
            next_observe, _, _, _ = env.step(1)

        # At start of episode, there is no preceding frame. So just copy initial states to make history
        state = pre_processing(next_observe, observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape(history, (84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1
            observe = next_observe
            # get action for the current history and go one step in environment
            action = agent.get_action(history)
            next_observe, reward, done, info = env.step(action)
            # pre-process the observation --> history
            next_state = pre_processing(next_observe, observe)
            next_state = np.reshape(next_state, (84, 84, 1))
            next_history = np.append(next_state, history[:, :, :3], axis=2)

            # if the ball is fall, then the agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(history, action, reward, next_history, done)

            # every some time interval, train model and update the target model with model
            if global_step % agent.train_interval == 0:
                agent.train_replay()
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()
            score += reward

            # if agent is dead, then reset the history
            if dead:
                history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape(history, (84, 84, 4))
                dead = False
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                # scores.append(score)
                # episodes.append(e)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./save_graph/Breakout_DQN.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon, "  global_step:", global_step)

                stats = [score, agent.avg_q_max / float(step),
                         step]
                for i in range(len(stats)):
                    agent.sess.run(agent.update_ops[i], feed_dict={
                        agent.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = agent.sess.run(agent.summary_op)
                agent.summary_writer.add_summary(summary_str, e + 1)
                agent.avg_q_max = 0
                step = 0

        if e % 1000 == 0:
            agent.save_model("./save_model/Breakout_DQN.h5")
