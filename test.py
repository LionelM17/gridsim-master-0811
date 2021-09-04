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

        done = False
        timestep = 0
        while not done:
        # while timestep < max_timestep:
        # for timestep in range(max_timestep):
        #     ids = [i for i, x in enumerate(obs.rho) if x > 1.0]
            # print("overflow rho: ", [obs.rho[i] for i in ids])
            print('------ step ', timestep)
            state = get_state_from_obs(obs, settings)
            action = my_agent.act(state, obs)
            obs, reward, done, info = env.step(action)
            episode_reward[episode] += reward
            timestep += 1
            if done:
                obs = env.reset()
                print('info:', info)
                print(f'episode cumulative reward={episode_reward[episode]}')

    return sum(episode_reward) / len(episode_reward)