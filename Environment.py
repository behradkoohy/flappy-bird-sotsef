import time
import flappy_bird_gymnasium as flappy_bird_gym
from Models.DQN import DQNWrapper
from torchvision import transforms
from tqdm import tqdm

from Logger import Logger

log = Logger()
env = flappy_bird_gym.make("FlappyBird-rgb-v0")
# shape = env.observation_space.shape
transform = transforms.Compose([transforms.ToTensor()])

obs = transform(env.reset())
shape = obs.shape

agent = DQNWrapper(shape, env, 100)
# breakpoint()
epoch = 0
max_epoch = 10000
t_reward = 0
steps = 0
t = tqdm(total=max_epoch)
while epoch < max_epoch:
    # Next action:
    # (feed the observation to your agent here)
    # print(type(obs))
    action = agent.act(obs)
    # Processing:
    obs, reward, done, info = env.step(action)
    t_reward += reward
    obs = transform(obs)
    agent.observe(obs, reward, done, info)
    # Rendering the game:
    # (remove this two lines during training)
    #
    # env.render()
    # time.sleep(1 / 30)  # FPS
    steps += 1
    # Checking if the player is still alive
    if done:
        obs = transform(env.reset())
        # print(epoch, t_reward, steps)
        t_reward = 0
        epoch += 1
        t.update(n=1)
        log.record_run(epoch, reward, steps, agent)

log.commit()
env.close()