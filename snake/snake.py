import pdb
from time import sleep
import gym
import sneks
import numpy as np
import math

WIDTH = 16
HEIGHT = 16

def get_head_pos(raw_state):
    head_row, head_col = np.where(raw_state == 101)
    if len(head_row) == 0:
        return -1, -1
    return head_col[0], head_row[0]

def get_apple_pos(raw_state):
    apple_row, apple_col = np.where(raw_state == 64)
    return apple_col[0], apple_row[0]

def extract_features(raw_state, raw_next_state):
    old_head_x, old_head_y = get_head_pos(raw_state)
    head_x, head_y = get_head_pos(raw_next_state)
    if head_x == -1:
        return extract_features(raw_state, raw_state)
    apple_x, apple_y = get_apple_pos(raw_next_state)

    curr_dir = (head_x - old_head_x, head_y - old_head_y)

    dist_left = 0
    for x in range(head_x-1, -1, -1):
        dist_left += 1
        if raw_next_state[head_y][x] % 64 != 0:
            break
    
    dist_right = 0
    for x in range(head_x+1, WIDTH):
        dist_right += 1
        if raw_next_state[head_y][x] % 64 != 0:
            break

    dist_up = 0
    for y in range(head_y-1, -1, -1):
        dist_up += 1
        if raw_next_state[:, head_x][y] % 64 != 0:
            break

    dist_down = 0
    for y in range(head_y+1, HEIGHT):
        dist_down += 1
        if raw_next_state[:, head_x][y] % 64 != 0:
            break
    
    norm = lambda v : v / np.linalg.norm(v)

    v1_u = np.array([curr_dir[0], -curr_dir[1]])
    v2_u = np.array([apple_x - head_x, - (apple_y - head_y)])
    angle = np.arctan2(v2_u[1], v2_u[0]) - np.arctan2(v1_u[1], v1_u[0])
    if (angle > math.pi):
        angle -= 2 * math.pi
    elif (angle <= -math.pi):
        angle += 2 * math.pi

    # print(f"CURR_DIR: {v1_u}")
    # print(f"APPLE VECTOR: {v2_u}")
    # print(f"ANGLE: {angle} -> {angle * 180 / math.pi}")

    return np.array([dist_left / WIDTH, dist_right / WIDTH, dist_up / HEIGHT, dist_down / HEIGHT, angle / math.pi])
    # return np.array([dist_left, dist_right, dist_up, dist_down, angle])

if __name__ == "__main__":
    env = gym.make("babysnek-raw-16-v1")
    total_rewards = []

    raw_state = env.reset()
    total_reward = 0
    done = False

    head_x, head_y = get_head_pos(raw_state)
    
    while not done:
        action = np.random.choice([0, 1])
        raw_next_state, reward, done, _ = env.step(action)

        next_head_x, next_head_y = get_head_pos(raw_next_state)
        next_state = extract_features(raw_state, raw_next_state)

        env.render()
        sleep(.1)
        print(next_state)
        pdb.set_trace()

        head_x, head_y = next_head_x, next_head_y
        state = next_state
        raw_state = raw_next_state
        total_reward += reward

    total_rewards.append(total_reward)
    
    env.close()
    
    average_reward = np.mean(total_rewards)
    print(f"Average reward for this model is {'{:.3f}'.format(average_reward)} ± {'{:.3f}'.format(np.std(total_rewards))}.")

