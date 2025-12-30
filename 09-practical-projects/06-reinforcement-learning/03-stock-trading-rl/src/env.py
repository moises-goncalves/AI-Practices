"""
股票交易环境

基于Gym接口的股票交易环境，支持多种强化学习算法。
"""

import numpy as np

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


class StockTradingEnv:
    """
    股票交易强化学习环境
    
    状态空间：
        - 账户余额
        - 持仓数量
        - 股票价格
        - 技术指标（可选）
    
    动作空间：
        - 连续动作：[-1, 1]，负数卖出，正数买入
        - 或离散动作：0=持有, 1=买入, 2=卖出
    
    奖励设计：
        - 基于资产变化的奖励
    """
    
    def __init__(
        self,
        df,
        initial_balance=100000,
        max_shares=100,
        transaction_fee=0.001,
        reward_scaling=1e-4
    ):
        """
        Args:
            df: 股票数据DataFrame，需包含'close'列
            initial_balance: 初始资金
            max_shares: 最大持仓数量
            transaction_fee: 交易手续费率
            reward_scaling: 奖励缩放因子
        """
        self.df = df
        self.initial_balance = initial_balance
        self.max_shares = max_shares
        self.transaction_fee = transaction_fee
        self.reward_scaling = reward_scaling
        
        self.prices = df['close'].values
        self.n_steps = len(self.prices)
        
        # 技术指标列（如果有）
        self.feature_columns = [col for col in df.columns 
                               if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_asset = self.initial_balance
        self.prev_total_asset = self.initial_balance
        
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        price = self.prices[self.current_step]
        
        state = [
            self.balance / self.initial_balance,
            self.shares_held / self.max_shares,
            price / self.prices[0],
        ]
        
        if self.feature_columns:
            for col in self.feature_columns:
                state.append(self.df[col].iloc[self.current_step])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        执行交易动作
        
        Args:
            action: 交易动作
                - 连续：[-1, 1]，负数卖出比例，正数买入比例
                - 离散：0=持有, 1=买入, 2=卖出
        
        Returns:
            state, reward, done, info
        """
        current_price = self.prices[self.current_step]
        
        if isinstance(action, (int, np.integer)):
            if action == 1:
                action = 1.0
            elif action == 2:
                action = -1.0
            else:
                action = 0.0
        
        if action > 0:
            max_buy = self.balance / (current_price * (1 + self.transaction_fee))
            shares_to_buy = min(int(max_buy * action), self.max_shares - self.shares_held)
            cost = shares_to_buy * current_price * (1 + self.transaction_fee)
            self.balance -= cost
            self.shares_held += shares_to_buy
        elif action < 0:
            shares_to_sell = min(int(self.shares_held * abs(action)), self.shares_held)
            revenue = shares_to_sell * current_price * (1 - self.transaction_fee)
            self.balance += revenue
            self.shares_held -= shares_to_sell
        
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        self.total_asset = self.balance + self.shares_held * self.prices[self.current_step]
        reward = (self.total_asset - self.prev_total_asset) * self.reward_scaling
        self.prev_total_asset = self.total_asset
        
        info = {
            'total_asset': self.total_asset,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'price': self.prices[self.current_step]
        }
        
        return self._get_state(), reward, done, info
    
    def render(self):
        """打印当前状态"""
        print(f"Step: {self.current_step}, "
              f"Balance: {self.balance:.2f}, "
              f"Shares: {self.shares_held}, "
              f"Total: {self.total_asset:.2f}")
