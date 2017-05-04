import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D

EPISODES = 50000


class DQNAgent:
    def __init__(self):
        self.render = False
        # environment settings
        self.state_size = (84, 84, 4)
        self.action_size = 3
        # parameters about epsilon
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / 1000000.
        # parameters about training
        self.batch_size = 32
        self.discount_factor = 0.99
        self.train_start = 50000
        self.update_target_interval = 10000
        self.memory = deque(maxlen=400000)
        # build model and target model. Then update target model with model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        # setting tf summary for tensorboard
        self.sess = tf.InteractiveSession()
        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/Breakout_DQN', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

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
        model.compile(loss=self.huber_loss, optimizer=tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(history)[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, alive):
        self.memory.append((history, action, reward, next_history, alive))

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        action, reward, alive = [], [], []
        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            alive.append(mini_batch[i][4])

        target_value = reward + np.array(alive) * self.discount_factor * np.amax(self.target_model.predict(next_history)
                                                                                 , axis=1)
        target = self.model.predict(history)

        for i in range(self.batch_size):
            target[i][action[i]] = target_value[i]

        self.avg_loss += self.model.train_on_batch(np.array(history), np.array(target))

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 210*160*3(color) --> 84*84(mono) and float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # In case of BreakoutDeterministic-v3, always skip 4 frames
    env = gym.make('BreakoutDeterministic-v3')
    agent = DQNAgent()

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        # 1 episode = 5 lives
        step, score, fake_action, start_life = 0, 0, 0, 5
        observe = env.reset()

        # At start of episode, there is no preceding frame. So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            step += 1
            alive = 1

            action = agent.get_action(history)
            if action == 0: fake_action = 1
            if action == 1: fake_action = 4
            if action == 2: fake_action = 5
            observe, reward, done, info = env.step(fake_action)
            score += reward
            # clip reward between -1 and 1 to improve learning performance
            reward = np.clip(reward, -1, 1)

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(history[:, :, :, 1:], next_state, axis=3)
            # get average max q function to see the training performance
            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])

            # if the ball is fall, then the alive of agent is False --> consider when calculating target
            if start_life > info['ale.lives']:
                alive = 0
                start_life = info['ale.lives']

            agent.replay_memory(history, action, reward, next_history, alive)
            history = next_history

            agent.train_replay()

            if global_step % agent.update_target_interval == 0:
                agent.update_target_model()

            # every episode, the result is recorded
            if done:
                stats = [score, agent.avg_q_max / float(step), agent.avg_loss / float(step)]
                for i in range(len(stats)):
                    agent.sess.run(agent.update_ops[i], feed_dict={agent.summary_placeholders[i]: float(stats[i])})
                summary_str = agent.sess.run(agent.summary_op)
                agent.summary_writer.add_summary(summary_str, e + 1)

                global_step = global_step + step
                print("episode:", e, "  score:", score, "  epsilon:", agent.epsilon, "  global_step:", global_step,
                      "  average_max_q:", stats[1], "  average loss:", stats[2], " steps:", step)
                agent.avg_q_max, agent.avg_loss = 0, 0

        if e % 1000 == 0:
            agent.model.save_weights("./save_model/Breakout_DQN.h5")