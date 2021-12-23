def mdp1_step(state, action, timestep):
	next_state, reward = state, 0
	done = False

	if action == 0 and state > 0:
		next_state -= 1
	elif action == 1 and state <= 3:
		next_state += 1
	
	if (state == 3 and next_state == 2) or (state == 2 and next_state == 3):
		reward = 1
	if state == 4 and action == 1:
		done = True

	if timestep > 100:
		done = True
	
	return next_state, reward, done

def mdp2_step(state, action, timestep):
	next_state, reward = state, 0
	done = False

	if action == 0 and state > 0:
		next_state -= 1
	elif action == 1 and state <= 3:
		next_state += 1
	
	if (state == 0 and next_state == 1):
		reward = 0.5
	if (state == 3 and next_state == 4):
		reward = 1
	if state == 4 and action == 1:
		done = True

	if timestep > 100:
		done = True
	
	return next_state, reward, done

def mdp3_step(state, action, timestep):
	next_state, reward = state, 0
	done = False

	if action == 0 and state > 0:
		next_state -= 1
	elif action == 1 and state <= 3:
		next_state += 1
	
	if (state == 0 and next_state == 1):
		reward = 1
	if (state == 1 and next_state == 2):
		reward = 1
	if (state == 2 and next_state == 1):
		reward = 0.1
	if state == 4 and action == 1:
		done = True

	if timestep > 100:
		done = True
	
	return next_state, reward, done

def mdp4_step(state, action, timestep):
	next_state, reward = state, 0
	done = False

	if action == 0 and state > 0:
		next_state -= 1
	elif action == 1 and state <= 3:
		next_state += 1
	
	if (state == 0 and next_state == 1):
		reward = 1
	if (state == 2 and next_state == 3):
		reward = 1
	if (state == 3 and next_state == 2):
		reward = 1
	if state == 4 and action == 1:
		done = True
	if state == 0 and action == 0:
		done = True
	if timestep > 100:
		done = True
	
	return next_state, reward, done