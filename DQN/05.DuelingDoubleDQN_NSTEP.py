import numpy as np
from timeit import default_timer as timer

from datetime import timedelta
import math

from agents.DuelingDoubleDQN_NSTEP import Model
from utils.hyperparameters import Config
from utils.plot import plot, save_plot
from utils.wrappers import wrap_pytorch, make_atari, wrap_deepmind

# 获取配置文件
config = Config()
# 记录开始时间
start = timer()
# 声明环境为乒乓球
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env, frame_stack=False)
env = wrap_pytorch(env)
# 构建模型
model = Model(env=env, config=config)
# 场景的收获
episode_reward = 0
# 获取场景初始状态
observation = env.reset()
# max_frames = int(config.MAX_FRAMES / 50)
# 最大frame数量
max_frames = config.MAX_FRAMES
# process_count = int(max_frames / 40)
# 输出进度的频率值，达到数量即输出
process_count = int(max_frames / 2000)
# 上次输出处理时间
process_time = 0
for frame_idx in range(1, max_frames + 1):
    epsilon = config.epsilon_by_frame(frame_idx)
    # print(type(epsilon))
    action = model.get_action(observation, epsilon)
    prev_observation = observation
    observation, reward, done, _ = env.step(action)
    observation = None if done else observation
    model.update(prev_observation, action, reward, observation, frame_idx)
    episode_reward += reward

    if done:
        model.finish_nstep()
        observation = env.reset()
        model.save_reward(episode_reward)
        episode_reward = 0
        if np.mean(model.rewards[-10:]) > 20:
            plot(frame_idx, model.rewards, model.losses, model.sigma_parameter_mag,
                 timedelta(seconds=int(timer() - start)), model.nsteps, name='DuelingDDQN')
            print("达到20提前结束，结束轮次,", frame_idx)
            break

    # 每一万次迭代绘制训练信息
    if frame_idx % config.polt_num == 0:
        plot(frame_idx, model.rewards, model.losses, model.sigma_parameter_mag, timedelta(seconds=int(timer() - start)),
             model.nsteps, name='DuelingDDQN')

    # 每save_polt_num保存图像结果
    if frame_idx % config.save_plot_num == 0:
        save_plot(frame_idx, model.rewards, model.losses, model.sigma_parameter_mag,
                  timedelta(seconds=int(timer() - start)),
                  nstep=model.nsteps, name="DuelingDDQN")

    # 输出进度与剩余时间
    if frame_idx % process_count == 0:
        finish_rate = frame_idx / max_frames
        remain_time = int((timer() - process_time) * (2000 - int(frame_idx / process_count)))
        hour = int(remain_time / 3600)
        minute = int((remain_time - hour * 3600) / 60)
        second = remain_time - hour * 3600 - minute * 60
        avg_reward = np.mean(model.rewards[-10:])
        print("step=%d 第%d轮 训练完成%.2f%%, avg_reward= %.1f, 剩余 %d小时 %d分 %d秒" % (model.nsteps,frame_idx,finish_rate * 100, avg_reward, hour, minute, second))
        process_time = timer()
model.save_weight(model_name="DuelingDoubleDQN_" + str(model.nsteps) + "STEP")
env.close()
print("输出完毕")
