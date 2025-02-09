import numpy as np

from env import Environment, Officers, Events
from tqdm import tqdm

def random_algorithm(env, num_episodes=100):
    """
    Randomly assigns officers to tasks and computes total reward for each episode.

    Args:
        env: An instance of the Environment class.
        num_episodes: Number of episodes to run the random algorithm.

    Returns:
        List of rewards for each episode.
    """
    rewards = []
    for episode in tqdm(range(num_episodes)):
        env.reset()
        done = False
        total_reward = 0

        while not done:
            # Generate a random assignment
            random_assignment = np.random.randint(0, env.officers.num_officer,
                                                  (env.events.num_event, env.events.num_task))
            max_time =  np.max(np.array(env.officers.capability)) * env.events.num_task * env.events.num_event + \
                        np.max(np.array(env.events.transition_matrix)) * env.events.num_task * env.events.num_event
            # Step through the environment
            event_times = env.step(random_assignment)

            # Calculate reward (negative of max completion time)
            reward = -np.max(event_times) / max_time
            total_reward += reward

            # In a single-pass environment, consider the episode done after one step
            done = True

        rewards.append(total_reward)
        # print(f"Episode {episode + 1}/{num_episodes}: Reward = {total_reward}")

    return rewards


# Example usage
if __name__ == "__main__":
    NUM_OFFICERS = 4
    NUM_EVENTS = 3
    NUM_TASKS = 2
    EPISODES = 50000

    env = Environment(Officers(NUM_OFFICERS, NUM_EVENTS, NUM_TASKS), Events(NUM_EVENTS, NUM_TASKS))
    random_rewards = random_algorithm(env, num_episodes=EPISODES)
    avg_rewards = [np.mean(random_rewards[i:i + 1000]) for i in range(0, len(random_rewards), 1000)]
    print("Average reward per 1000 episodes:", avg_rewards)
    print("Average Reward over Random Algorithm:", np.mean(random_rewards))
