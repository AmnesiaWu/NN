import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATE = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9 # 贪婪率
ALPHA = 0.1 # 学习率
LAMBDA = 0.9 # Q(next)折扣率
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_Q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(State, Action):
    if Action =='right':
        if State == N_STATE - 2:
            S_re = 'terminal'
            R = 1
        else:
            S_re = State + 1
            R = 0
    else:
        R = 0
        if State == 0:
            S_re = State
        else:
            S_re = State - 1
    return S_re, R

def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATE - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode:{}, total_step:{}'.format(episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print("\r{}".format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    Q_table = build_Q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, Q_table)
            S_re, R = get_env_feedback(S, A)
            q_predict = Q_table.loc[S, A]
            if S_re != 'terminal':
                q_target = R + LAMBDA * Q_table.iloc[S_re, :].max()
            else:
                q_target = R
                is_terminated = True
            Q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_re
            step_counter += 1
            update_env(S, episode, step_counter)
    return Q_table

if __name__ == '__main__':
    q_table = rl()
    print(q_table)