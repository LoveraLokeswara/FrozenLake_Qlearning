import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
import time

def run(episodes, istraining = True, render = False):
    env = gym.make("FrozenLake-v1", map_name = "4x4", is_slippery = False, render_mode = "human" if render else None)

    if istraining:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        try:
            f = open('frozenlake_4x4.pkl', 'rb')
            q_table = pickle.load(f)
            f.close()
            # print("Loaded trained Q-table from frozenlake_4x4.pkl")
        except FileNotFoundError:
            # print("Error: frozenlake_4x4.pkl not found! Train the model first with istraining=True")
            return

    learning_rate = 0.9
    discount_factor = 0.9
    max_steps = 100

    if istraining:
        epsilon = 1.0  # Start with full exploration for training
    else:
        epsilon = 0.0  # No exploration 
    epsilon_decay = 0.0001
    rng = np.random.default_rng()

    total_rewards = np.zeros(episodes)
    
    start_time = time.perf_counter()  # Initialize timing for progress tracking

    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        steps = 0

        while(not terminated and not truncated and steps < max_steps):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state,:])

            new_state, reward, terminated, truncated, info = env.step(action)
            if istraining:
                q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state,:]) - q_table[state, action])
            state = new_state
            steps += 1

        if istraining:
            epsilon = max(epsilon - epsilon_decay, 0)
            
            if epsilon == 0:
                learning_rate = 0.0001

        if reward == 1:
            total_rewards[episode] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for i in range(episodes):
        sum_rewards[i] = np.sum(total_rewards[max(0, i-100):i+1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(sum_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Successes in Last 100 Episodes')
    plt.title('Q-Learning Progress: FrozenLake 4x4')
    plt.grid(True, alpha=0.3)
    plt.savefig('frozenlake_4x4.png', dpi=300, bbox_inches='tight')
    # plt.show()

    if istraining:
        # Save Q-table as pickle file (for loading later)
        with open('frozenlake_4x4.pkl', 'wb') as f:
            pickle.dump(q_table, f)
        # print(f"Q-table saved to: frozenlake_4x4.pkl")
        
        # Save Q-table as readable CSV file
        np.savetxt('frozenlake_4x4_qtable.csv', q_table, delimiter=',', fmt='%.6f')
        # print(f"Q-table saved to: frozenlake_4x4_qtable.csv")


    # return sum_rewards

    # Debugging
    # if (episode + 1) % 100 == 0:
    #     elapsed = time.perf_counter() - start_time
    #     eps_per_s = (episode + 1) / elapsed
    #     remaining = (episodes - (episode + 1)) / eps_per_s
    #     print(f"[{episode+1}/{episodes}] ~{eps_per_s:.1f} eps/s, ETA {remaining:.1f}s")

if __name__ == "__main__":
    # Training mode
    run(15000, istraining = True, render = False)

    # Testing mode
    run(10, istraining = False, render = True)