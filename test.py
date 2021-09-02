from Environment.base_env import Environment
from utilize.settings import settings
from utils import get_state_from_obs

def run_task(my_agent):
    max_episode = 10
    max_timestep = 10000
    episode_reward = [0 for _ in range(max_episode)]
    for episode in range(max_episode):
        print('------ episode ', episode)
        env = Environment(settings, "EPRIReward")
        print('------ reset ')
        obs = env.reset()

        reward = 0.0
        done = False

        # while not done:
        for timestep in range(max_timestep):
            ids = [i for i, x in enumerate(obs.rho) if x > 1.0]
            print("overflow rho: ", [obs.rho[i] for i in ids])
            print('------ step ', timestep)
            state = get_state_from_obs(obs, settings)
            action = my_agent.act(state, obs)
            obs, reward, done, info = env.step(action)
            episode_reward[episode] += reward
            if done:
                print('info:', info)
                print(f'episode cumulative reward={episode_reward[episode]}')
                break

        return sum(episode_reward) / len(episode_reward)