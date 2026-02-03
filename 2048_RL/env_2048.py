import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional


class Game2048Env(gym.Env):
    """
    一个用于强化学习训练的 2048 游戏环境。
    观测空间：4x4 棋盘的 tile 值（int），也可以自行改成 log2 表示。
    动作空间：0=上, 1=下, 2=左, 3=右
    奖励：当前一步中所有合并产生的 tile 和（标准做法）。
    终止：无合法动作时（所有方向都无法移动）。
    """
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, board_size: int = 4, seed: Optional[int] = None):
        super().__init__()
        self.board_size = board_size

        # 4 个离散动作：上、下、左、右
        self.action_space = spaces.Discrete(4)

        # 观测：board_size x board_size 的非负整数
        # 这里用 0 ~ 2**16 作为上界，基本够用。也可以用 Box(low=0, high=np.inf, ...)
        self.observation_space = spaces.Box(
            low=0,
            high=2 ** 16,
            shape=(self.board_size, self.board_size),
            dtype=np.int32,
        )

        self.rng = np.random.RandomState(seed)
        self.board: np.ndarray = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self._done: bool = False

    # ---- Gym 标准接口 ----

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng.seed(seed)
        self.board[:] = 0
        self._done = False
        # 初始生成两个 tile
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        if self._done:
            # 若已经结束，通常 gym 做法是再 reset；这里返回原样
            return self.board.copy(), 0.0, True, False, {}

        old_board = self.board.copy()
        reward = self._move(action)

        # 如果移动后棋盘没有变化，则视为非法/无效动作，可设置小负奖励（可选）
        if np.array_equal(old_board, self.board):
            # 这里不给惩罚也可以，根据你的算法需求调整
            reward = 0.0

        # 每一步后尝试添加新 tile（如果有空格且动作有效）
        if not np.array_equal(old_board, self.board):
            self._add_random_tile()

        self._done = not self._can_move()

        terminated = self._done
        truncated = False  # 可以根据步数限制等设置

        observation = self.board.copy()
        info: Dict[str, Any] = {}
        return observation, float(reward), terminated, truncated, info

    def render(self, mode: str = "human"):
        if mode == "ansi":
            return self._board_to_string()
        else:
            print(self._board_to_string())

    def close(self):
        pass

    # ---- 2048 逻辑实现 ----

    def _add_random_tile(self):
        """在一个随机空格中添加 2 或 4（90% 概率 2，10% 概率 4）"""
        empty_positions = np.argwhere(self.board == 0)
        if empty_positions.size == 0:
            return
        idx = self.rng.randint(len(empty_positions))
        r, c = empty_positions[idx]
        # 90% 2, 10% 4
        self.board[r, c] = 4 if self.rng.rand() < 0.1 else 2

    def _can_move(self) -> bool:
        """检查是否还有合法动作。"""
        # 有空格就可以移动
        if np.any(self.board == 0):
            return True

        # 棋盘满时看是否有相邻相等的
        for r in range(self.board_size):
            for c in range(self.board_size - 1):
                if self.board[r, c] == self.board[r, c + 1]:
                    return True
        for c in range(self.board_size):
            for r in range(self.board_size - 1):
                if self.board[r, c] == self.board[r + 1, c]:
                    return True
        return False

    def _move(self, action: int) -> int:
        """
        执行动作并返回本步奖励。
        动作含义：0=上, 1=下, 2=左, 3=右
        """
        reward = 0

        if action == 0:  # 上
            for c in range(self.board_size):
                col = self.board[:, c]
                new_col, gained = self._compress_and_merge(col)
                self.board[:, c] = new_col
                reward += gained

        elif action == 1:  # 下
            for c in range(self.board_size):
                col = self.board[::-1, c]
                new_col, gained = self._compress_and_merge(col)
                self.board[::-1, c] = new_col
                reward += gained

        elif action == 2:  # 左
            for r in range(self.board_size):
                row = self.board[r, :]
                new_row, gained = self._compress_and_merge(row)
                self.board[r, :] = new_row
                reward += gained

        elif action == 3:  # 右
            for r in range(self.board_size):
                row = self.board[r, ::-1]
                new_row, gained = self._compress_and_merge(row)
                self.board[r, ::-1] = new_row
                reward += gained

        return reward

    def _compress_and_merge(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        先压缩非零到一侧，再从头到尾合并相同的块（合并一次），最后再压缩一次。
        返回 (新行, 本行总奖励)。
        """
        size = len(line)
        new_line = line[line != 0]  # 去掉 0
        gained = 0

        # 如果全是 0
        if len(new_line) == 0:
            return np.zeros_like(line), 0

        merged = []
        skip = False
        for i in range(len(new_line)):
            if skip:
                skip = False
                continue
            if i + 1 < len(new_line) and new_line[i] == new_line[i + 1]:
                merged_value = new_line[i] * 2
                gained += merged_value
                merged.append(merged_value)
                skip = True
            else:
                merged.append(new_line[i])

        merged_arr = np.array(merged, dtype=np.int32)
        # 再次压缩到左侧，右侧补 0
        result = np.zeros(size, dtype=np.int32)
        result[: len(merged_arr)] = merged_arr
        return result, gained

    def _board_to_string(self) -> str:
        return "\n".join(
            ["\t".join(f"{v:4d}" for v in row) for row in self.board]
        )