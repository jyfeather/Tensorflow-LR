import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # 选择 action
        if np.random.uniform() < self.epsilon: # 选择 Q value 最高的 action
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # 同一个 state，可能会有多个相同的 Q action value，所以先乱序一下
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else: # 随机选择 action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """
        学习率 * (真实值 - 预测值). 将判断误差传递回去, 有着和神经网络更新的异曲同工之处。
        """
        self.check_state_exist(s_) # 检测 q_table 中是否存在 s_
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # 下个 state 不是 终止符
        else:
            q_target = r  # 下个 state 是 终止符
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新对应的 state-action 值

    def check_state_exist(self, state):
        """
        检测 q_table 中有没有当前 state 的步骤了, 如果还没有当前 state, 
        那我我们就插入一组全 0 数据, 当做这个 state 的所有 action 初始 values。
        """
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )