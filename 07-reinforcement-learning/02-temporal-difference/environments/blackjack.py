"""
二十一点环境 (Blackjack Environment)
===================================

核心思想 (Core Idea):
--------------------
二十一点是研究Monte Carlo方法和TD学习的经典环境。
玩家需要决定是否继续要牌(hit)或停牌(stick)，
目标是手牌点数尽可能接近21点但不超过。

数学原理 (Mathematical Theory):
------------------------------
状态空间: S = {(player_sum, dealer_showing, usable_ace)}
    - player_sum ∈ {12, 13, ..., 21} (小于12时必须hit)
    - dealer_showing ∈ {1, 2, ..., 10}
    - usable_ace ∈ {True, False}
    总状态数: 10 × 10 × 2 = 200

动作空间: A = {HIT, STICK}

奖励结构:
    R = +1  玩家赢
    R = -1  玩家输
    R = 0   平局

庄家策略（固定）:
    - 点数 < 17: 必须hit
    - 点数 ≥ 17: 必须stick

最优策略要点:
    - 有可用A时，在18+停牌
    - 无可用A时，策略取决于庄家明牌
    - 庄家明牌大时需要更激进

问题背景 (Problem Statement):
----------------------------
Blackjack是测试RL算法的理想环境:
1. 状态空间适中（200个状态）
2. 有明确的最优策略可供对比
3. 随机性来自发牌，体现期望值学习
4. 经典MC控制和TD控制的测试床

复杂度 (Complexity):
-------------------
- 状态数: 200
- 动作数: 2
- 平均回合长度: O(1) ~ O(10)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from enum import IntEnum

from .base import DiscreteSpace


class BlackjackAction(IntEnum):
    """Blackjack动作枚举。"""
    STICK = 0  # 停牌
    HIT = 1    # 要牌


class Blackjack:
    """
    简化版二十一点环境实现。

    核心思想 (Core Idea):
    --------------------
    实现标准Blackjack规则，用于测试TD和MC算法。
    采用简化规则：无分牌、无双倍下注、无保险。

    数学原理 (Mathematical Theory):
    ------------------------------
    牌值计算:
        - 2-10: 面值
        - J, Q, K: 10
        - A: 1 或 11 (可用A)

    可用A (Usable Ace):
        当手中有A且将其算作11时点数不超过21，
        则称有"可用A"。这是状态的重要组成部分。

    状态表示:
        state = (player_sum, dealer_showing, usable_ace)

        编码为整数:
        state_index = (player_sum - 12) * 20 + (dealer_showing - 1) * 2 + usable_ace

    算法对比 (Comparison):
    ---------------------
    在Blackjack上的算法表现:

    ┌─────────────────┬──────────────┬─────────────┐
    │      算法       │   收敛速度   │  最终性能   │
    ├─────────────────┼──────────────┼─────────────┤
    │   MC预测        │     慢       │    精确     │
    │   TD(0)预测     │     快       │   略有偏差  │
    │   SARSA         │     中等     │    良好     │
    │   Q-Learning    │     中等     │    良好     │
    └─────────────────┴──────────────┴─────────────┘

    由于回合较短，MC和TD在此环境差异不大。

    Example:
        >>> env = Blackjack()
        >>> state, _ = env.reset()
        >>> player_sum, dealer_showing, usable_ace = env.decode_state(state)
        >>> print(f"玩家: {player_sum}, 庄家: {dealer_showing}, 可用A: {usable_ace}")
        >>> # 决定要牌
        >>> next_state, reward, done, _, _ = env.step(BlackjackAction.HIT)
    """

    def __init__(self, natural_bonus: bool = False) -> None:
        """
        初始化Blackjack环境。

        Args:
            natural_bonus: 是否对自然21点(Natural Blackjack)给予额外奖励
                          True: Natural赢得1.5倍奖励
                          False: 统一+1奖励
        """
        self.natural_bonus = natural_bonus

        # 状态空间: 10(玩家点数) × 10(庄家明牌) × 2(可用A) = 200
        self.observation_space = DiscreteSpace(200)
        self.action_space = DiscreteSpace(2)

        # 牌组（无限牌组，有放回抽取）
        # 牌值: 1=A, 2-10=面值, 10=J/Q/K (简化为都是10)
        self._deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

        # 游戏状态
        self._player_hand: List[int] = []
        self._dealer_hand: List[int] = []
        self._done: bool = False

    def _draw_card(self) -> int:
        """从牌组抽一张牌（有放回）。"""
        return np.random.choice(self._deck)

    def _hand_value(self, hand: List[int]) -> Tuple[int, bool]:
        """
        计算手牌点数。

        数学原理:
        --------
        1. 先将所有A算作1点，计算基础点数
        2. 如果有A且加10后不超过21，则使用可用A

        Args:
            hand: 手牌列表

        Returns:
            (点数, 是否有可用A)
        """
        total = sum(hand)
        usable_ace = False

        # 检查是否有可用A
        if 1 in hand and total + 10 <= 21:
            total += 10
            usable_ace = True

        return total, usable_ace

    def _is_bust(self, hand: List[int]) -> bool:
        """检查是否爆牌（超过21点）。"""
        return self._hand_value(hand)[0] > 21

    def _is_natural(self, hand: List[int]) -> bool:
        """
        检查是否为自然21点（Natural Blackjack）。

        Natural: 初始两张牌就是21点（A + 10/J/Q/K）
        """
        if len(hand) != 2:
            return False
        return set(hand) == {1, 10}

    def _encode_state(
        self,
        player_sum: int,
        dealer_showing: int,
        usable_ace: bool
    ) -> int:
        """
        将状态元组编码为整数索引。

        编码方式:
            index = (player_sum - 12) * 20 + (dealer_showing - 1) * 2 + usable_ace

        Args:
            player_sum: 玩家点数 (12-21)
            dealer_showing: 庄家明牌 (1-10)
            usable_ace: 是否有可用A

        Returns:
            状态索引 (0-199)
        """
        # 裁剪到有效范围
        player_sum = max(12, min(21, player_sum))
        dealer_showing = max(1, min(10, dealer_showing))

        return (player_sum - 12) * 20 + (dealer_showing - 1) * 2 + int(usable_ace)

    def decode_state(self, state: int) -> Tuple[int, int, bool]:
        """
        将状态索引解码为状态元组。

        Args:
            state: 状态索引 (0-199)

        Returns:
            (player_sum, dealer_showing, usable_ace)
        """
        usable_ace = state % 2
        state //= 2
        dealer_showing = state % 10 + 1
        player_sum = state // 10 + 12

        return player_sum, dealer_showing, bool(usable_ace)

    def _get_obs(self) -> int:
        """获取当前观测（状态索引）。"""
        player_sum, usable_ace = self._hand_value(self._player_hand)

        # 如果点数小于12，还不需要决策（自动hit）
        # 但为了返回有效状态，我们裁剪到12
        player_sum = max(12, player_sum)

        dealer_showing = self._dealer_hand[0]

        return self._encode_state(player_sum, dealer_showing, usable_ace)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        重置游戏，发初始手牌。

        发牌顺序:
            1. 玩家获得两张牌
            2. 庄家获得两张牌（一张明牌，一张暗牌）

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            (初始状态, 信息字典)
        """
        if seed is not None:
            np.random.seed(seed)

        # 发牌
        self._player_hand = [self._draw_card(), self._draw_card()]
        self._dealer_hand = [self._draw_card(), self._draw_card()]
        self._done = False

        # 如果玩家点数小于12，自动补牌直到>=12
        while self._hand_value(self._player_hand)[0] < 12:
            self._player_hand.append(self._draw_card())

        info = {
            "player_hand": self._player_hand.copy(),
            "dealer_showing": self._dealer_hand[0],
        }

        return self._get_obs(), info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        执行动作。

        游戏流程:
        --------
        1. HIT: 玩家要牌
           - 爆牌则输(-1)
           - 否则继续
        2. STICK: 玩家停牌
           - 庄家按规则补牌
           - 比较点数决定胜负

        Args:
            action: 0=STICK, 1=HIT

        Returns:
            (下一状态, 奖励, 是否终止, 是否截断, 信息字典)
        """
        if self._done:
            # 游戏已结束，返回当前状态
            return self._get_obs(), 0.0, True, False, {}

        if action == BlackjackAction.HIT:
            # 要牌
            self._player_hand.append(self._draw_card())

            if self._is_bust(self._player_hand):
                # 爆牌，玩家输
                self._done = True
                return (
                    self._get_obs(),
                    -1.0,
                    True,
                    False,
                    {"result": "player_bust", "player_hand": self._player_hand.copy()}
                )

            # 未爆牌，继续游戏
            return self._get_obs(), 0.0, False, False, {}

        else:  # STICK
            # 停牌，庄家按规则补牌
            self._done = True

            # 庄家策略: 小于17必须hit
            while self._hand_value(self._dealer_hand)[0] < 17:
                self._dealer_hand.append(self._draw_card())

            player_sum = self._hand_value(self._player_hand)[0]
            dealer_sum = self._hand_value(self._dealer_hand)[0]

            info = {
                "player_hand": self._player_hand.copy(),
                "dealer_hand": self._dealer_hand.copy(),
                "player_sum": player_sum,
                "dealer_sum": dealer_sum,
            }

            # 判定结果
            if self._is_bust(self._dealer_hand):
                # 庄家爆牌，玩家赢
                info["result"] = "dealer_bust"
                reward = 1.0
            elif player_sum > dealer_sum:
                # 玩家点数更高
                info["result"] = "player_win"
                reward = 1.0
                # Natural bonus
                if self.natural_bonus and self._is_natural(self._player_hand):
                    reward = 1.5
            elif player_sum < dealer_sum:
                # 庄家点数更高
                info["result"] = "dealer_win"
                reward = -1.0
            else:
                # 平局
                info["result"] = "draw"
                reward = 0.0

            return self._get_obs(), reward, True, False, info

    def render(self, mode: str = "human") -> Optional[str]:
        """
        渲染当前游戏状态。

        显示玩家手牌、点数和庄家明牌。
        """
        player_sum, usable_ace = self._hand_value(self._player_hand)
        dealer_showing = self._dealer_hand[0]

        result = f"玩家手牌: {self._player_hand} (点数: {player_sum}"
        if usable_ace:
            result += ", 可用A"
        result += ")\n"
        result += f"庄家明牌: {dealer_showing}\n"

        if self._done:
            dealer_sum = self._hand_value(self._dealer_hand)[0]
            result += f"庄家手牌: {self._dealer_hand} (点数: {dealer_sum})\n"

        if mode == "human":
            print(result)
            return None
        return result

    def get_optimal_policy(self) -> Dict[Tuple[int, int, bool], int]:
        """
        获取近似最优策略（基于基本策略表）。

        数学原理:
        --------
        最优策略是通过大量模拟或动态规划得出的。
        以下是简化版的基本策略:

        有可用A时:
            - 点数 <= 17: HIT
            - 点数 >= 18: STICK

        无可用A时:
            - 点数 <= 11: HIT (实际上不会出现，因为状态从12开始)
            - 点数 == 12: 庄家明牌4-6时STICK，否则HIT
            - 点数 13-16: 庄家明牌2-6时STICK，否则HIT
            - 点数 >= 17: STICK

        Returns:
            (player_sum, dealer_showing, usable_ace) -> action 的映射
        """
        policy = {}

        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                for usable_ace in [False, True]:
                    if usable_ace:
                        # 有可用A: 18+停牌
                        action = BlackjackAction.STICK if player_sum >= 18 else BlackjackAction.HIT
                    else:
                        # 无可用A
                        if player_sum >= 17:
                            action = BlackjackAction.STICK
                        elif player_sum <= 11:
                            action = BlackjackAction.HIT
                        elif player_sum == 12:
                            action = BlackjackAction.STICK if 4 <= dealer_showing <= 6 else BlackjackAction.HIT
                        else:  # 13-16
                            action = BlackjackAction.STICK if 2 <= dealer_showing <= 6 else BlackjackAction.HIT

                    policy[(player_sum, dealer_showing, usable_ace)] = action

        return policy

    @staticmethod
    def get_state_space_info() -> Dict[str, Any]:
        """
        获取状态空间信息。

        Returns:
            状态空间的详细描述
        """
        return {
            "player_sum_range": (12, 21),
            "dealer_showing_range": (1, 10),
            "usable_ace_values": [False, True],
            "total_states": 200,
            "encoding": "index = (player_sum - 12) * 20 + (dealer_showing - 1) * 2 + usable_ace"
        }
