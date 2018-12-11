from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np

from utils.hyperparameters import Config

cfg = Config()


# 展示收reward，loss和sigma
def plot(frame_idx, rewards, losses, sigma, elapsed_time, nstep=1, name="DQN"):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('name: %s. step: %d. frame %s. reward: %s. time: %s' % (
        name, nstep, frame_idx, np.mean(rewards[-10:]), elapsed_time))
    plt.plot(rewards)
    if losses:
        plt.subplot(132)
        plt.title("loss")
        plt.plot(losses)
    if sigma:
        plt.subplot(133)
        plt.title('noisy param magnitude')
        plt.plot(sigma)
    plt.show()


def save_plot(frame_idx, rewards, losses, sigma, elapsed_time, nstep=1, name="DQN"):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('name: %s. step: %d. frame %s. reward: %s. time: %s' % (
        name, nstep, frame_idx, np.mean(rewards[-10:]), elapsed_time))
    plt.plot(rewards)
    if losses:
        plt.subplot(132)
        plt.title("loss")
        plt.plot(losses)
    if sigma:
        plt.subplot(133)
        plt.title('noisy param magnitude')
        plt.plot(sigma)
    plt.savefig("%sname: %s. step: %d. frame %s." % (cfg.save_plot_path,
        name, nstep, frame_idx))
