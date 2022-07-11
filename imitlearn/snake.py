import gym
import gym_snake
import pdb

import math
import numpy as np

def decode_state(state):
    if state is None:
        return []
    
    N = state.shape[0]
    output = np.ones((N+2, N+2)) * 4
    for i in range(N):
        for j in range(N):
            same_color = (state[i][j][0] == state[i][j][1] == state[i][j][2])

            is_apple = state[i][j][0] == 255
            is_body = (state[i][j][1] == 255) or (same_color and state[i][j][0] == 64)
            is_head = (state[i][j][2] == 255) or (same_color and state[i][j][0] == 128)
            # is_space = not (is_apple or is_body or is_head)
            
            code = 1 if is_body else (2 if is_head else (3 if is_apple else 0))
            output[i+1][j+1] = code
    return output.T

def construct_features(env, old_state, state):
    N = state.shape[0]

    head_i, head_j = np.where(state == 2)

    if old_state != []:
        head_i1, head_j1 = np.where(old_state == 2)
        
        snake_vec = (head_i - head_i1, head_j - head_j1)
        snake_dir = 0 if snake_vec[0] == -1 else (1 if snake_vec[1] == 1 else (2 if snake_vec[0] == 1 else 3))
    else:
        snake_vec = (1, 0)
        snake_dir = 2

    apple_i, apple_j = np.where(state == 3)
    apple_dir = (apple_i - head_i, apple_j - head_j)

    rotate_dir = lambda dir, a : [(-1, 0), (0, 1), (1, 0), (0, -1)][min(max(snake_dir + a, 0), 3)]

    # try:
    #     if state[head_i, head_j-1] == 1:
    #         snake_dir = [0, 1]
    #     elif state[head_i-1, head_j] == 1:
    #         snake_dir = [1, 0]
    #     elif state[head_i, head_j+1] == 1:
    #         snake_dir = [0, -1]
    #     elif state[head_i+1, head_j] == 1:
    #         snake_dir = [-1, 0]
    # except:
    #     pdb.set_trace()

    imminent_colision = True
    collision_front, obj_in_front = 0, 0
    collision_left, obj_to_left = 0, 0
    collision_right, obj_to_right = 0, 0
    angle_to_apple = 0

    if snake_vec != (0, 0):
        imminent_collision = False
        if state[head_i + snake_vec[0], head_j + snake_vec[1]] in [1, 4]:
            imminent_collision = True

        scanned_i, scanned_j = head_i[0], head_j[0]
        while state[scanned_i, scanned_j] != 4:
            scanned_i += snake_vec[0]
            scanned_j += snake_vec[1]
            # if state[scanned_i, scanned_j]
        obj_in_front = state[scanned_i, scanned_j]
        collision_front = 1 if state[head_i[0] + snake_vec[0], head_j[0] + snake_vec[1]] else 0

        scanned_i, scanned_j = head_i[0], head_j[0]
        while state[scanned_i, scanned_j] not in [1, 3, 4]:
            scanned_i += rotate_dir(snake_vec, -1)[0]
            scanned_j += rotate_dir(snake_vec, -1)[1]
        obj_to_left = state[scanned_i, scanned_j]
        collision_left = 1 if state[head_i[0] + rotate_dir(snake_vec, -1)[0], head_j[0] + rotate_dir(snake_vec, -1)[1]] else 0

        dist_wall_left = 0
        scanned_i, scanned_j = head_i[0], head_j[0]
        while state[scanned_i, scanned_j] != 4:
            dist_wall_left += 1
            scanned_i += rotate_dir(snake_vec, -1)[0]
            scanned_j += rotate_dir(snake_vec, -1)[1]
        
        scanned_i, scanned_j = head_i[0], head_j[0]
        while state[scanned_i, scanned_j] not in [1, 3, 4]:
            scanned_i += rotate_dir(snake_vec, 1)[0]
            scanned_j += rotate_dir(snake_vec, 1)[1]
        obj_to_right = state[scanned_i, scanned_j]
        collision_right = 1 if state[head_i[0] + rotate_dir(snake_vec, 1)[0], head_j[0] + rotate_dir(snake_vec, 1)[1]] else 0

        dist_wall_right = 0
        scanned_i, scanned_j = head_i[0], head_j[0]
        while state[scanned_i, scanned_j] != 4:
            dist_wall_right += 1
            scanned_i += rotate_dir(snake_vec, 1)[0]
            scanned_j += rotate_dir(snake_vec, 1)[1]
        
        # return [head_i, head_j, 1 if imminent_collision else 0, obj_in_view]
        a = snake_vec / np.linalg.norm(snake_vec)
        b = apple_dir / np.linalg.norm(apple_dir)
        angle_to_apple = math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    features = [
        collision_front, 
        collision_left,
        collision_right, 
        1 if obj_in_front == 3 else 0,
        1 if obj_to_left == 3 else 0, 
        1 if obj_to_right == 3 else 0,
        - angle_to_apple]

    # if np.random.uniform(0, 1) <= 0.2:
    #     pdb.set_trace()
    
    return features

if __name__ == "__main__":
    env = gym.make("Snake-4x4-v0")
    total_rewards = []

    for episode in range(1):
        state = env.reset()
        total_reward = 0
        done = False
        
        k = 0
        while not done:
            decoded_state = decode_state(state)
            if k == 2:
                features = construct_features(decoded_state)

            action = [2, 2][k]
            k += 1
            next_state, reward, done, _ = env.step(action)
            print(f"Action taken {action}, reward {reward}")

            state = next_state
            total_reward += reward

        print(f"Episode #{episode} finished with total reward {total_reward}")
        total_rewards.append(total_reward)
    
    env.close()
    print(f"Average reward for this model is {np.mean(total_rewards)}.")