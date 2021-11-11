import json

import matplotlib.pyplot as plt

fontsize = 15


def plot_dqn_visualization():
    with open("models/dqn_training_status.json") as f:
        data = json.load(f)

    episode_length = data["episode_length"]
    reward = data["reward"]
    losses = data["losses"]

    plt.figure()
    plt.plot(episode_length)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Length Of Episode", fontsize=fontsize)
    plt.savefig("episode_length_dqn.jpg")

    plt.figure()
    plt.plot(reward)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Total Reward", fontsize=fontsize)
    plt.savefig("reward_dqn.jpg")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Losses", fontsize=fontsize)
    plt.savefig("losses_dqn.jpg")


def plot_ddqn_visualization():
    with open("models/ddqn_training_status.json") as f:
        data = json.load(f)

    episode_length = data["episode_length"]
    reward = data["reward"]
    losses = data["losses"]

    plt.figure()
    plt.plot(episode_length)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Length Of Episode", fontsize=fontsize)
    plt.savefig("episode_length_ddqn.jpg")

    plt.figure()
    plt.plot(reward)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Total Reward", fontsize=fontsize)
    plt.savefig("reward_ddqn.jpg")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Losses", fontsize=fontsize)
    plt.savefig("losses_ddqn.jpg")


def plot_a2c_visualization():
    with open("models/a2c_training_status.json") as f:
        data = json.load(f)

    episode_length = data["episode_len_list"]
    reward = data["reward_list"]
    policy_losses = data["policy_losses_list"]
    value_losses = data["value_losses_list"]
    entropy_losses = data["entropy_losses_list"]

    plt.figure()
    plt.plot(episode_length)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Length Of Episode", fontsize=fontsize)
    plt.savefig("episode_length_a2c.jpg")

    plt.figure()
    plt.plot(reward)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Total Reward", fontsize=fontsize)
    plt.savefig("reward_a2c.jpg")

    plt.figure()
    plt.plot(policy_losses)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Policy Losses", fontsize=fontsize)
    plt.savefig("policy_losses_a2c.jpg")

    plt.figure()
    plt.plot(value_losses)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Value Losses", fontsize=fontsize)
    plt.savefig("value_losses_a2c.jpg")

    plt.figure()
    plt.plot(entropy_losses)
    plt.xlabel("Number Of Episodes", fontsize=fontsize)
    plt.ylabel("Entropy Losses", fontsize=fontsize)
    plt.savefig("entropy_losses_a2c.jpg")


if __name__ == "__main__":
    plot_dqn_visualization()
    plot_ddqn_visualization()
    plot_a2c_visualization()
