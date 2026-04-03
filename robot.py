import numpy as np
from collections import deque


class Robot:
    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y
        self.x = None
        self.y = None
        self.total_moves = 0
        self.points = 0
        self.collisions = 0
        self.moves_since_last_goal = 0
        self.position_history = deque(maxlen=8)
        self.stuck_counter = 0
        self.last_actions = deque(maxlen=4)
        self.movement_actions = {
            0: (0, 1),   # Up
            1: (0, -1),  # Down
            2: (-1, 0),  # Left
            3: (1, 0)    # Right
        }

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.position_history.clear()
        self.stuck_counter = 0
        self.last_actions.clear()

    def _detect_loop(self):
        if len(self.position_history) < 4:
            return False
        positions = list(self.position_history)
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        if x_range <= 2 and y_range <= 2:
            self.stuck_counter += 1
            return self.stuck_counter > 2
        else:
            self.stuck_counter = 0
            return False

    def _detect_action_pattern(self):
        if len(self.last_actions) < 4:
            return False
        actions = list(self.last_actions)
        # Check for back-and-forth pattern (e.g., up-down-up-down)
        if (actions[0] == actions[2] and actions[1] == actions[3] and actions[0] != actions[1]):
            return True
        # Check for same action repeated
        if len(set(actions)) == 1:
            return True
        return False

    def get_state(self, obstacles=None, all_goals=None):
        state = [
            self.x / (self.max_x - 1),   # Normalized X position
            self.y / (self.max_y - 1)    # Normalized Y position
        ]

        # Info about adjacent cells (obstacles and borders)
        for action in [0, 1, 2, 3]:
            dx, dy = self.movement_actions[action]
            new_x, new_y = self.x + dx, self.y + dy
            if new_x < 0 or new_x >= self.max_x or new_y < 0 or new_y >= self.max_y:
                state.append(1.0)
            elif obstacles and obstacles[new_x][new_y]:
                state.append(1.0)
            else:
                state.append(0.0)

        # Collisions and moves since last goal
        state.append(min(self.collisions / 10.0, 1.0))
        free_cells = (self.max_x - 2) * (self.max_y - 2)
        state.append(len(all_goals) / free_cells if all_goals and free_cells > 0 else 0.0)
        state.append(min(self.moves_since_last_goal / 50.0, 1.0))
        state.append(float(self._detect_loop()))
        state.append(float(self._detect_action_pattern()))

        # Direction to nearest goal
        if all_goals and len(all_goals) > 0:
            goal_x, goal_y = min(all_goals, key=lambda p: abs(p[0] - self.x) + abs(p[1] - self.y))
            dx_goal = (goal_x - self.x) / (self.max_x - 1)
            dy_goal = (goal_y - self.y) / (self.max_y - 1)
            state.append(dx_goal)
            state.append(dy_goal)
        else:
            state.extend([0.0, 0.0])

        # Normalized distance to goal
        if all_goals and len(all_goals) > 0:
            distance = abs(goal_x - self.x) + abs(goal_y - self.y)
            normalized_distance = distance / (self.max_x + self.max_y)
            state.append(normalized_distance)
        else:
            state.append(0.0)

        # Minimum distance from obstacles
        min_distance = self.max_x + self.max_y
        for action in [0, 1, 2, 3]:
            distance = 0
            dx, dy = self.movement_actions[action]
            test_x, test_y = self.x, self.y
            while 0 <= test_x + dx < self.max_x and 0 <= test_y + dy < self.max_y:
                test_x += dx
                test_y += dy
                distance += 1
                if obstacles and obstacles[test_x][test_y]:
                    break
            min_distance = min(min_distance, distance)
        state.append(min(min_distance / max(self.max_x, self.max_y), 1.0))

        return np.array(state, dtype=np.float32)

    def move_with_action(self, action, obstacles=None, all_goals=None):
        if action not in self.movement_actions:
            return -10, True, {"error": "Invalid action"}

        self.position_history.append((self.x, self.y))
        self.last_actions.append(action)

        dx, dy = self.movement_actions[action]
        new_x = self.x + dx
        new_y = self.y + dy

        reward = -0.1
        done = False
        info = {}

        if self._detect_loop():
            reward -= 5
            info["loop_detected"] = True

        if self._detect_action_pattern():
            reward -= 3
            info["pattern_detected"] = True

        # Check grid boundaries
        if new_x < 0 or new_x >= self.max_x or new_y < 0 or new_y >= self.max_y:
            return reward, done, info

        # Check obstacle collision
        if obstacles and obstacles[new_x][new_y]:
            collision_penalty = 30 + (self.collisions * 2)
            reward -= min(collision_penalty, 50)
            self.collisions += 1
            info["collision"] = True
            return reward, done, info

        # Move to new position
        self.x = new_x
        self.y = new_y
        self.total_moves += 1
        self.moves_since_last_goal += 1

        # Check if goal reached
        if all_goals and (self.x, self.y) in all_goals:
            base_reward = 50
            if self.moves_since_last_goal < 20:
                base_reward += (20 - self.moves_since_last_goal) * 2
            reward += base_reward
            self.moves_since_last_goal = 0
            self.points += 10
            info["goal_reached"] = True

            all_goals.remove((self.x, self.y))

            if not all_goals or len(all_goals) == 0:
                done = True
                info["all_goals_completed"] = True

        # Penalty for revisiting recent positions
        if (self.x, self.y) in list(self.position_history)[-4:]:
            reward -= 40

        # Reward for exploring unique positions
        if len(self.position_history) >= 5:
            unique_positions = len(set(self.position_history))
            reward += unique_positions * 0.5

        return reward, done, info
