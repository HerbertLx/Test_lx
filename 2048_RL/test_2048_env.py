# %%
from env_2048 import Game2048Env
import numpy as np
import time
import curses
import argparse


def default_agent(env: Game2048Env, obs: np.ndarray) -> int:
    """
    一个简单的默认 agent，这里示例为“总是向左”（动作 2）。
    你可以之后替换成自己的策略（比如规则/网络输出）。
    """
    return 2  # 2=左


def random_agent(env: Game2048Env, obs: np.ndarray) -> int:
    """随机 agent。"""
    return env.action_space.sample()


def run_episode(agent_fn, max_steps: int = 100, render: bool = True, fps: float = 2.0):
    """
    通用的环境运行函数：
    - agent_fn: 给定 (env, obs) 输出一个 action。
    - render=True 时，在终端中以固定频率 fps 输出棋盘。
    """
    env = Game2048Env()
    obs, info = env.reset()
    total_reward = 0.0

    dt = 1.0 / fps if fps > 0 else 0.0

    for t in range(max_steps):
        action = agent_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render:
            print(f"Step {t}, action={action}, reward={reward}, terminated={terminated}")
            env.render()
            print("-" * 30)
            if dt > 0:
                time.sleep(dt)

        if terminated or truncated:
            break

    print("Episode finished, total reward:", total_reward)
    env.close()


def human_agent_curses(max_steps: int = 1000):
    """
    使用方向键进行人工控制的 agent。
    注意：需要在真正的终端中运行，IDE 的集成终端有时对 curses 支持不好。
    """

    def _loop(stdscr):
        curses.curs_set(0)  # 隐藏光标
        stdscr.nodelay(False)  # 阻塞等待按键
        stdscr.keypad(True)  # 支持方向键

        env = Game2048Env()
        obs, info = env.reset()
        total_reward = 0.0
        step = 0

        key_to_action = {
            curses.KEY_UP: 0,
            curses.KEY_DOWN: 1,
            curses.KEY_LEFT: 2,
            curses.KEY_RIGHT: 3,
        }

        while True and step < max_steps:
            stdscr.clear()
            stdscr.addstr(0, 0, f"Step: {step}  Total reward: {total_reward}\n")
            stdscr.addstr(1, 0, "Use arrow keys to move, 'q' to quit.\n\n")

            # 打印棋盘
            board_str = env._board_to_string()
            stdscr.addstr(3, 0, board_str)
            stdscr.refresh()

            key = stdscr.getch()
            if key == ord("q"):
                break

            if key not in key_to_action:
                continue

            action = key_to_action[key]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if terminated or truncated:
                stdscr.clear()
                stdscr.addstr(0, 0, "Game over!\n")
                stdscr.addstr(1, 0, f"Final step: {step}  Total reward: {total_reward}\n\n")
                stdscr.addstr(3, 0, env._board_to_string())
                stdscr.addstr(3 + env.board_size + 2, 0, "Press any key to exit.")
                stdscr.refresh()
                stdscr.getch()
                break

    curses.wrapper(_loop)


# 以下仍然保留你之前写的规则测试函数，方便用 pytest 等测试逻辑
def test_merge_simple_row():
    env = Game2048Env()
    env.board = np.array([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    # 向左
    obs, reward, terminated, truncated, info = env.step(2)  # 2=左

    # 合并成 [4,0,0,0]，本行奖励为 4
    assert env.board[0, 0] == 4
    assert env.board[0, 1] == 0
    assert reward >= 4  # 可能还会新生成 tile 的奖励不影响 >=4 的判断


def test_no_merge_with_gap():
    env = Game2048Env()
    env.board = np.array([
        [2, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    # 向左，应先压缩为 [2,2,0,0] 再合并 => [4,0,0,0]
    obs, reward, terminated, truncated, info = env.step(2)
    assert env.board[0, 0] == 4
    assert env.board[0, 1] == 0
    assert reward >= 4


def test_cannot_move_end():
    env = Game2048Env()
    # 人为设置一个已满且无相邻可合并的棋盘
    env.board = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ], dtype=np.int32)
    # 任意动作之后应直接结束（因为没有空格也无可合并）
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="2048 Gym Env demo with different agents.")
    # 添加 agent 类型参数，可选 default（贪心）、random（随机）、human（手动控制）
    parser.add_argument(
        "--agent",
        type=str,
        choices=["default", "random", "human"],
        default="default",
        help="选择 agent 类型：default / random / human",
    )
    # 添加最大步数参数，仅对非 human agent 有效，默认 200
    parser.add_argument("--max_steps", type=int, default=200, help="最大步数（对非 human agent 有效）")
    # 添加帧率参数，对应每秒渲染步数，非 human agent 有效，默认 2.0
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="渲染帧率（每秒步数，对非 human agent 有效）",
    )
    # 添加 no_render 参数，指定非 human agent 时是否关闭渲染，默认渲染
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="非 human agent 模式下关闭渲染。",
    )

    # 解析命令行参数
    args = parser.parse_args()

    if args.agent == "human":
        # 如果是 human agent，则用 curses 实现的手动游玩界面
        human_agent_curses(max_steps=args.max_steps)
    else:
        # 选择 agent (default: 贪心，random: 随机)
        if args.agent == "default":
            agent = default_agent
        else:
            agent = random_agent
        # 运行一次游戏 episode，根据传入的参数进行渲染与控制
        run_episode(agent, max_steps=args.max_steps, render=not args.no_render, fps=args.fps)