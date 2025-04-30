import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ale_py
import gymnasium
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import time
import os
import argparse
from PIL import Image
import wandb
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        state = np.array(state, copy=True)
        next_state = np.array(next_state, copy=True)
        e = self.Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.tensor(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.tensor(np.array([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.tensor(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.tensor(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = (
            torch.tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        )

        return (states, actions.unsqueeze(1), rewards.unsqueeze(1), next_states, dones.unsqueeze(1))

    def __len__(self):
        return len(self.memory)


class DoubleDQNAgent:
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=2.5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.95)

        self.memory = ReplayBuffer(50000)
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.01
        self.update_every = 4
        self.target_update = 1000
        self.t_step = 0
        self.total_steps = 0

        self.loss_list = []
        self.q_values_list = []

        # Log model architecture to wandb
        wandb.watch(self.qnetwork_local, log="all", log_freq=100)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        self.total_steps += 1

        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            return self.learn(experiences)

        if self.total_steps % self.target_update == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            logger.info(f"Target network updated at step {self.total_steps}")

        return None, None

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        max_q_value = action_values.max().item()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), max_q_value
        else:
            return random.choice(np.arange(self.action_size)), max_q_value

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.huber_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10)
        self.optimizer.step()
        self.scheduler.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        loss_value = loss.item()
        self.loss_list.append(loss_value)

        avg_q = Q_expected.mean().item()
        self.q_values_list.append(avg_q)

        wandb.log(
            {
                "batch_loss": loss_value,
                "batch_avg_q": avg_q,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "td_error": torch.abs(Q_targets - Q_expected).mean().item(),
                "q_targets_mean": Q_targets.mean().item(),
                "q_targets_std": Q_targets.std().item(),
                "global_step": self.total_steps,
            }
        )

        return loss_value, avg_q

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def load(self, checkpoint_path):
        self.qnetwork_local.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        logger.info(f"Model loaded from {checkpoint_path}")


def train_agent(
    agent,
    env,
    n_episodes=10000,
    max_t=10000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    save_freq=1000,
    log_freq=100,
    buffer_fill_steps=50000,
    checkpoint_path=None,
    start_episode=1,
    render_training=False,
    render_freq=500,
    save_path="training_frames",
):
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from {checkpoint_path}")
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path, map_location=device))
        agent.qnetwork_target.load_state_dict(agent.qnetwork_local.state_dict())

    if start_episode == 1:
        logger.info("Populating replay buffer...")
        state, _ = env.reset()
        state = np.array(state)
        for i in range(buffer_fill_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated
            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
            if done:
                state, _ = env.reset()
                state = np.array(state)

            if i % 10000 == 0:
                logger.info(f"Replay buffer filling progress: {i}/{buffer_fill_steps} experiences")

        logger.info(f"Replay buffer populated with {len(agent.memory)} experiences")

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    if start_episode > 1:
        eps = max(eps_end, eps_decay ** (start_episode - 1))

    losses = []
    q_values = []
    epsilons = []
    episode_lengths = []

    config = {
        "env_name": env.unwrapped.spec.id,
        "episodes": n_episodes,
        "max_steps_per_episode": max_t,
        "epsilon_start": eps_start,
        "epsilon_end": eps_end,
        "epsilon_decay": eps_decay,
        "buffer_size": agent.memory.memory.maxlen,
        "batch_size": agent.batch_size,
        "gamma": agent.gamma,
        "tau": agent.tau,
        "update_every": agent.update_every,
        "target_update": agent.target_update,
        "learning_rate": agent.optimizer.param_groups[0]["lr"],
        "device": str(device),
    }
    wandb.config.update(config)

    logger.info("Starting training...")
    best_score = -21.0
    for i_episode in range(start_episode, n_episodes + 1):
        episode_start_time = time.time()
        state, _ = env.reset()
        state = np.array(state)
        score = 0
        loss_in_episode = []
        q_value_in_episode = []
        frames = []
        actions_taken = np.zeros(agent.action_size)

        render_this_episode = render_training and (i_episode % render_freq == 0)

        for t in range(max_t):
            action, q_value = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)

            actions_taken[action] += 1

            if render_this_episode:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))

            clipped_reward = np.clip(reward, -1.0, 1.0)
            next_state = np.array(next_state)
            done = terminated or truncated

            loss, avg_q = agent.step(state, action, clipped_reward, next_state, done)
            if loss is not None:
                loss_in_episode.append(loss)
            if avg_q is not None:
                q_value_in_episode.append(avg_q)
            q_value_in_episode.append(q_value)

            state = next_state
            score += reward
            if done:
                break

        episode_duration = time.time() - episode_start_time
        fps = (t + 1) / episode_duration

        scores_window.append(score)
        scores.append(score)
        episode_lengths.append(t + 1)

        if loss_in_episode:
            avg_loss = np.mean(loss_in_episode)
            losses.append(avg_loss)
        else:
            avg_loss = np.nan
            losses.append(np.nan)

        if q_value_in_episode:
            avg_q_val = np.mean(q_value_in_episode)
            q_values.append(avg_q_val)
        else:
            avg_q_val = np.nan
            q_values.append(np.nan)

        epsilons.append(eps)
        eps = max(eps_end, eps_decay * eps)

        action_distribution = (
            actions_taken / np.sum(actions_taken) if np.sum(actions_taken) > 0 else np.zeros(agent.action_size)
        )

        episode_metrics = {
            "episode": i_episode,
            "score": score,
            "avg_score_100": np.mean(scores_window),
            "episode_length": t + 1,
            "epsilon": eps,
            "episode_avg_loss": avg_loss if not np.isnan(avg_loss) else 0,
            "episode_avg_q": avg_q_val if not np.isnan(avg_q_val) else 0,
            "memory_size": len(agent.memory),
            "fps": fps,
            "episode_duration_seconds": episode_duration,
        }

        for i, pct in enumerate(action_distribution):
            episode_metrics[f"action_{i}_pct"] = pct

        wandb.log(episode_metrics)

        if render_this_episode and frames:
            frames_array = np.stack([np.array(frame) for frame in frames])
            wandb.log({"game_play": wandb.Video(frames_array, fps=30, format="gif")})

        if i_episode % log_freq == 0 or i_episode == 1:
            avg_score = np.mean(scores_window)
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            logger.info(
                f"Episode {i_episode}/{n_episodes} | Avg Score: {avg_score:.2f} | Epsilon: {eps:.4f} | Avg Length: {avg_length:.1f} | FPS: {fps:.1f}"
            )

            if not np.isnan(avg_loss):
                logger.info(f"Latest Loss: {avg_loss:.4f} | Latest Q-Value: {avg_q_val:.4f}")

            action_dist_str = " | ".join([f"Action {i}: {pct:.2%}" for i, pct in enumerate(action_distribution)])
            logger.info(f"Action distribution: {action_dist_str}")

            if avg_score > best_score:
                best_score = avg_score
                logger.info(f"New best score: {best_score:.2f}! Saving model...")
                torch.save(agent.qnetwork_local.state_dict(), "best_model.pth")
                wandb.save("best_model.pth")

        if render_this_episode and frames and save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            episode_path = os.path.join(save_path, f"training_episode_{i_episode}")
            if not os.path.exists(episode_path):
                os.makedirs(episode_path)

            for j, frame in enumerate(frames):
                frame.save(os.path.join(episode_path, f"frame_{j:04d}.png"))

            if len(frames) > 0:
                frames[0].save(
                    os.path.join(save_path, f"training_episode_{i_episode}.gif"),
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=40,
                    loop=0,
                )
                logger.info(f"Saved training gif for episode {i_episode}")

        if i_episode % save_freq == 0:
            logger.info(
                f"Checkpoint reached at Episode {i_episode}/{n_episodes}. Average Score: {np.mean(scores_window):.2f}. Saving model..."
            )
            checkpoint_file = f"checkpoint_{i_episode}.pth"
            torch.save(agent.qnetwork_local.state_dict(), checkpoint_file)
            wandb.save(checkpoint_file)

            if not os.path.exists("metrics"):
                os.makedirs("metrics")
            metrics = {
                "scores": scores,
                "losses": losses,
                "q_values": q_values,
                "epsilons": epsilons,
                "episode_lengths": episode_lengths,
            }
            metrics_file = f"metrics/metrics_{i_episode}.npy"
            np.save(metrics_file, metrics)
            wandb.save(metrics_file)

            # Create plots and save to wandb
            plot_metrics(
                scores[:i_episode],
                losses[:i_episode],
                q_values[:i_episode],
                epsilons[:i_episode],
                episode_lengths[:i_episode],
                save_dir=f"plots_checkpoint_{i_episode}",
            )

            for plot_file in os.listdir(f"plots_checkpoint_{i_episode}"):
                wandb.log(
                    {
                        f"checkpoint_plot_{plot_file}": wandb.Image(
                            os.path.join(f"plots_checkpoint_{i_episode}", plot_file)
                        )
                    }
                )

    return scores, losses, q_values, epsilons, episode_lengths


def plot_metrics(scores, losses, q_values, epsilons, episode_lengths, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot scores
    plt.figure(figsize=(12, 8))
    window_size = 100
    plt.plot(np.arange(len(scores)), scores, alpha=0.3, label="Episode Score")
    if len(scores) >= window_size:
        rolling_mean = np.convolve(scores, np.ones(window_size) / window_size, mode="valid")
        plt.plot(
            np.arange(window_size - 1, len(scores)), rolling_mean, linewidth=2, label="Rolling Mean (100 episodes)"
        )
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.title("Scores per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "scores_plot.png"))
    plt.close()

    # Plot losses
    valid_losses = [x for x in losses if not np.isnan(x)]
    if valid_losses:
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(len(losses)), losses, alpha=0.3, label="Loss per Episode")
        if len(losses) >= window_size:
            valid_indices = np.where(~np.isnan(losses))[0]
            if len(valid_indices) >= window_size:
                valid_losses = np.array(losses)[valid_indices]
                rolling_mean = np.convolve(valid_losses, np.ones(window_size) / window_size, mode="valid")
                plt.plot(
                    valid_indices[window_size - 1 :], rolling_mean, linewidth=2, label="Rolling Mean (100 episodes)"
                )
        plt.ylabel("Loss")
        plt.xlabel("Episode #")
        plt.title("Huber Loss per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "loss_plot.png"))
        plt.close()

    valid_q_values = [x for x in q_values if not np.isnan(x)]
    if valid_q_values:
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(len(q_values)), q_values, alpha=0.3, label="Q-Value per Episode")
        if len(q_values) >= window_size:
            valid_indices = np.where(~np.isnan(q_values))[0]
            if len(valid_indices) >= window_size:
                valid_q = np.array(q_values)[valid_indices]
                rolling_mean = np.convolve(valid_q, np.ones(window_size) / window_size, mode="valid")
                plt.plot(
                    valid_indices[window_size - 1 :], rolling_mean, linewidth=2, label="Rolling Mean (100 episodes)"
                )
        plt.ylabel("Average Q-Value")
        plt.xlabel("Episode #")
        plt.title("Q-Values per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "q_value_plot.png"))
        plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(epsilons)), epsilons, label="Epsilon Value")
    plt.ylabel("Epsilon")
    plt.xlabel("Episode #")
    plt.title("Exploration Rate (Epsilon) Decay")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "epsilon_plot.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(episode_lengths)), episode_lengths, alpha=0.3, label="Episode Length")
    if len(episode_lengths) >= window_size:
        rolling_mean = np.convolve(episode_lengths, np.ones(window_size) / window_size, mode="valid")
        plt.plot(
            np.arange(window_size - 1, len(episode_lengths)),
            rolling_mean,
            linewidth=2,
            label="Rolling Mean (100 episodes)",
        )
    plt.ylabel("Steps")
    plt.xlabel("Episode #")
    plt.title("Episode Length")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "episode_length_plot.png"))
    plt.close()

    logger.info(f"All plots saved to {save_dir}/")


def evaluate_agent(agent, env, n_episodes=10, render=False, save_path=None):
    scores = []
    episode_lengths = []
    logger.info("\nStarting evaluation...")

    if render and save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    max_score = float("-inf")
    max_score_episode = 0
    min_score = float("inf")
    min_score_episode = 0
    score_distribution = []
    all_action_distributions = []

    for i in range(1, n_episodes + 1):
        state, _ = env.reset()
        state = np.array(state)
        score = 0
        frames = []
        actions_taken = np.zeros(agent.action_size)

        step = 0
        while True:
            action, q_value = agent.act(state, eps=0.01)
            actions_taken[action] += 1

            next_state, reward, terminated, truncated, info = env.step(action)

            if render and step % 4 == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))

            done = terminated or truncated
            next_state = np.array(next_state)
            state = next_state
            score += reward
            step += 1

            if done:
                break

        action_distribution = (
            actions_taken / np.sum(actions_taken) if np.sum(actions_taken) > 0 else np.zeros(agent.action_size)
        )
        all_action_distributions.append(action_distribution)

        scores.append(score)
        episode_lengths.append(step)
        score_distribution.append(score)

        if score > max_score:
            max_score = score
            max_score_episode = i

        if score < min_score:
            min_score = score
            min_score_episode = i

        eval_metrics = {
            "eval_episode": i,
            "eval_score": score,
            "eval_steps": step,
        }
        for j, pct in enumerate(action_distribution):
            eval_metrics[f"eval_action_{j}_pct"] = pct
        wandb.log(eval_metrics)

        # Log action distribution in text
        action_dist_str = " | ".join([f"Action {j}: {pct:.2%}" for j, pct in enumerate(action_distribution)])
        logger.info(
            f"Evaluation Episode {i}/{n_episodes}: Score: {score} | Steps: {step} | Actions: {action_dist_str}"
        )

        if render and save_path and frames:
            episode_path = os.path.join(save_path, f"episode_{i}")
            if not os.path.exists(episode_path):
                os.makedirs(episode_path)

            for j, frame in enumerate(frames):
                frame.save(os.path.join(episode_path, f"frame_{j:04d}.png"))

            if len(frames) > 0:
                gif_path = os.path.join(save_path, f"episode_{i}.gif")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=40,
                    loop=0,
                )
                logger.info(f"Saved gif for episode {i} to {gif_path}")
                wandb.log(
                    {
                        f"eval_episode_{i}_video": wandb.Video(
                            np.stack([np.array(frame) for frame in frames]), fps=30, format="gif"
                        )
                    }
                )

    avg_score = np.mean(scores)
    avg_length = np.mean(episode_lengths)
    score_std = np.std(scores)

    avg_action_dist = np.mean(all_action_distributions, axis=0)
    action_dist_str = " | ".join([f"Action {i}: {pct:.2%}" for i, pct in enumerate(avg_action_dist)])

    logger.info("\nEvaluation Results:")
    logger.info(f"Average Score: {avg_score:.2f} ± {score_std:.2f}")
    logger.info(f"Average Episode Length: {avg_length:.2f}")
    logger.info(f"Max Score: {max_score} (Episode {max_score_episode})")
    logger.info(f"Min Score: {min_score} (Episode {min_score_episode})")
    logger.info(f"Average Action Distribution: {action_dist_str}")

    if save_path:
        plt.figure(figsize=(10, 6))
        plt.hist(score_distribution, bins=10, alpha=0.7)
        plt.axvline(avg_score, color="r", linestyle="dashed", linewidth=2)
        plt.title(f"Score Distribution (Avg: {avg_score:.2f})")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        score_dist_path = os.path.join(save_path, "score_distribution.png")
        plt.savefig(score_dist_path)
        plt.close()
        logger.info(f"Score distribution plot saved to {score_dist_path}")
        wandb.log({"eval_score_distribution": wandb.Image(score_dist_path)})

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(avg_action_dist)), avg_action_dist)
        plt.title("Average Action Distribution During Evaluation")
        plt.xlabel("Action")
        plt.ylabel("Frequency")
        plt.xticks(range(len(avg_action_dist)))
        plt.grid(True, alpha=0.3)
        action_dist_path = os.path.join(save_path, "action_distribution.png")
        plt.savefig(action_dist_path)
        plt.close()
        logger.info(f"Action distribution plot saved to {action_dist_path}")
        wandb.log({"eval_action_distribution": wandb.Image(action_dist_path)})

    wandb.log(
        {
            "final_eval_avg_score": avg_score,
            "final_eval_score_std": score_std,
            "final_eval_max_score": max_score,
            "final_eval_min_score": min_score,
            "final_eval_avg_episode_length": avg_length,
        }
    )

    return {
        "avg_score": avg_score,
        "avg_length": avg_length,
        "max_score": max_score,
        "min_score": min_score,
        "score_std": score_std,
        "scores": scores,
        "episode_lengths": episode_lengths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Double DQN for Atari Games")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "resume"],
        help="Mode: train, evaluate, or resume training",
    )
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4", help="Atari environment name")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--save_freq", type=int, default=100, help="Frequency of saving checkpoints")
    parser.add_argument("--log_freq", type=int, default=10, help="Frequency of logging detailed information")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--eps-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation or resuming training"
    )
    parser.add_argument("--start-episode", type=int, default=1, help="Episode to start from when resuming training")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes")
    parser.add_argument("--save-path", type=str, default="eval_frames", help="Path to save rendered frames and gifs")
    parser.add_argument("--render-training", action="store_true", help="Render training episodes")
    parser.add_argument("--render-freq", type=int, default=500, help="Frequency of rendering training episodes")
    parser.add_argument("--wandb-project", type=str, default="atari-dqn", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    run_name = (
        args.wandb_run_name if args.wandb_run_name else f"{args.env}_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "env": args.env,
            "mode": args.mode,
            "episodes": args.episodes,
            "save_freq": args.save_freq,
            "buffer_size": args.buffer_size,
            "eps_decay": args.eps_decay,
            "checkpoint": args.checkpoint,
            "start_episode": args.start_episode,
            "seed": args.seed,
        },
    )

    logger.info(f"Starting run: {run_name}")
    logger.info(f"Args: {args}")

    env_name = args.env

    if args.mode == "train" or args.mode == "resume":
        env = gymnasium.make(env_name, render_mode="rgb_array" if args.render_training else None)
        env = AtariPreprocessing(env, terminal_on_life_loss=True, grayscale_obs=True, frame_skip=4, screen_size=84)
        env = FrameStackObservation(env, 4)

        state_size = (4, 84, 84)
        action_size = env.action_space.n
        agent = DoubleDQNAgent(state_size, action_size, seed=args.seed)

        start_time = time.time()

        if args.mode == "train":
            logger.info("Starting fresh training run")
            metrics = train_agent(
                agent,
                env,
                n_episodes=args.episodes,
                max_t=10000,
                save_freq=args.save_freq,
                log_freq=args.log_freq,
                eps_decay=args.eps_decay,
                buffer_fill_steps=args.buffer_size,
                render_training=args.render_training,
                render_freq=args.render_freq,
                save_path=args.save_path,
            )
        else:
            if not args.checkpoint:
                logger.error("Checkpoint path must be provided when resuming training")
                exit(1)

            logger.info(f"Resuming training from checkpoint: {args.checkpoint}")
            metrics = train_agent(
                agent,
                env,
                n_episodes=args.episodes,
                max_t=10000,
                save_freq=args.save_freq,
                log_freq=args.log_freq,
                eps_decay=args.eps_decay,
                buffer_fill_steps=args.buffer_size,
                checkpoint_path=args.checkpoint,
                start_episode=args.start_episode,
                render_training=args.render_training,
                render_freq=args.render_freq,
                save_path=args.save_path,
            )

        end_time = time.time()
        training_time_minutes = (end_time - start_time) / 60
        logger.info(f"Training completed in {training_time_minutes:.2f} minutes")
        wandb.log({"total_training_time_minutes": training_time_minutes})

        scores, losses, q_values, epsilons, episode_lengths = metrics
        plot_metrics(scores, losses, q_values, epsilons, episode_lengths)

        if not os.path.exists("results"):
            os.makedirs("results")

        final_model_path = "results/final_model.pth"
        metrics_path = "results/final_metrics.npz"
        torch.save(agent.qnetwork_local.state_dict(), final_model_path)
        np.savez(
            metrics_path,
            scores=np.array(scores),
            losses=np.array(losses),
            q_values=np.array(q_values),
            epsilons=np.array(epsilons),
            episode_lengths=np.array(episode_lengths),
        )

        wandb.save(final_model_path)
        wandb.save(metrics_path)

        logger.info(f"Final model saved to {final_model_path}")
        logger.info(f"Metrics saved to {metrics_path}")

        env.close()

        logger.info("\nRunning evaluation on trained model...")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        eval_env = gymnasium.make(env_name, render_mode="rgb_array")
        eval_env = AtariPreprocessing(
            eval_env, terminal_on_life_loss=True, grayscale_obs=True, frame_skip=4, screen_size=84
        )
        eval_env = FrameStackObservation(eval_env, 4)
        agent.qnetwork_local.load_state_dict(torch.load(final_model_path, map_location=device))
        eval_results = evaluate_agent(
            agent, eval_env, n_episodes=args.eval_episodes, render=args.render, save_path=args.save_path
        )

        eval_results_path = os.path.join("results", "eval_results.txt")
        with open(eval_results_path, "w") as f:
            for key, value in eval_results.items():
                if key not in ["scores", "episode_lengths"]:
                    f.write(f"{key}: {value}\n")

        wandb.save(eval_results_path)

        plt.figure(figsize=(10, 6))
        plt.hist(eval_results["scores"], bins=10, alpha=0.7)
        plt.axvline(eval_results["avg_score"], color="r", linestyle="dashed", linewidth=2)
        plt.title(f"Evaluation Score Distribution (Avg: {eval_results['avg_score']:.2f})")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        final_score_dist_path = "results/final_eval_score_distribution.png"
        plt.savefig(final_score_dist_path)
        plt.close()

        wandb.log({"final_evaluation": wandb.Image(final_score_dist_path)})

        eval_env.close()
        logger.info("Evaluation finished.")

    elif args.mode == "evaluate":
        if not args.checkpoint:
            logger.error("Checkpoint path must be provided for evaluation")
            exit(1)

        logger.info("\nSetting up evaluation environment...")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        eval_env = gymnasium.make(env_name, render_mode="rgb_array")
        eval_env = AtariPreprocessing(
            eval_env, terminal_on_life_loss=True, grayscale_obs=True, frame_skip=4, screen_size=84
        )
        eval_env = FrameStackObservation(eval_env, 4)

        state_size = (4, 84, 84)
        action_size = eval_env.action_space.n
        agent = DoubleDQNAgent(state_size, action_size, seed=args.seed)
        agent.load(args.checkpoint)

        eval_results = evaluate_agent(
            agent, eval_env, n_episodes=args.eval_episodes, render=args.render, save_path=args.save_path
        )

        eval_dir = os.path.dirname(args.checkpoint) if os.path.dirname(args.checkpoint) else "."
        eval_results_path = os.path.join(eval_dir, "eval_results.txt")
        with open(eval_results_path, "w") as f:
            for key, value in eval_results.items():
                if key not in ["scores", "episode_lengths"]:
                    f.write(f"{key}: {value}\n")

        wandb.save(eval_results_path)

        eval_env.close()
        logger.info("Evaluation finished.")

    wandb.finish()