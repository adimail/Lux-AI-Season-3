import numpy as np
from lux.utils import direction_to


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        np.random.seed(0)
        self.env_cfg = env_cfg

        # Game parameters
        self.unit_sap_cost = env_cfg["unit_sap_cost"]
        self.unit_move_cost = env_cfg["unit_move_cost"]
        self.unit_sap_range = env_cfg["unit_sap_range"]

        # Exploration tracking
        self.explored_map = None
        self.unit_assignments = {}  # {unit_id: target_position}
        self.prev_team_points = 0

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        if not obs["obs"]:
            return actions

        # Initialize explored map
        sensor_mask = np.array(obs["obs"]["sensor_mask"])
        if self.explored_map is None:
            self.explored_map = np.zeros_like(sensor_mask, dtype=bool)
        self.explored_map |= sensor_mask

        # Extract observations
        unit_mask = np.array(obs["obs"]["units_mask"][self.team_id])
        unit_positions = np.array(obs["obs"]["units"]["position"][self.team_id])
        unit_energys = np.array(obs["obs"]["units"]["energy"][self.team_id])
        map_features = obs["obs"]["map_features"]
        available_unit_ids = np.where(unit_mask)[0]

        # Update relic information
        visible_relic_mask = np.array(obs["obs"]["relic_nodes_mask"])
        visible_relic_positions = np.array(obs["obs"]["relic_nodes"])[
            visible_relic_mask
        ]

        # Clear invalid assignments
        for uid in list(self.unit_assignments.keys()):
            if uid not in available_unit_ids or not any(
                np.array_equal(self.unit_assignments[uid], pos)
                for pos in visible_relic_positions
            ):
                del self.unit_assignments[uid]

        # Assign units to relics
        unassigned_units = [
            uid for uid in available_unit_ids if uid not in self.unit_assignments
        ]
        for relic_pos in visible_relic_positions:
            if not unassigned_units:
                break
            distances = [
                np.abs(unit_positions[uid] - relic_pos).sum()
                for uid in unassigned_units
            ]
            closest_uid = unassigned_units[np.argmin(distances)]
            self.unit_assignments[closest_uid] = relic_pos
            unassigned_units.remove(closest_uid)

        # Process each unit
        for unit_id in available_unit_ids:
            current_pos = unit_positions[unit_id]
            current_energy = unit_energys[unit_id][0]
            action_taken = False

            # 1. Energy management and sapping
            if current_energy < self.unit_move_cost * 2:
                best_sap_gain = 0
                best_delta = (0, 0)
                for dx in range(-self.unit_sap_range, self.unit_sap_range + 1):
                    for dy in range(-self.unit_sap_range, self.unit_sap_range + 1):
                        if abs(dx) + abs(dy) > self.unit_sap_range:
                            continue
                        x = current_pos[0] + dx
                        y = current_pos[1] + dy
                        if (
                            0 <= x < self.env_cfg["map_width"]
                            and 0 <= y < self.env_cfg["map_height"]
                        ):
                            if (
                                sensor_mask[x][y]
                                and map_features["energy"][x][y] > self.unit_sap_cost
                            ):
                                sap_gain = (
                                    map_features["energy"][x][y] - self.unit_sap_cost
                                )
                                if sap_gain > best_sap_gain:
                                    best_sap_gain = sap_gain
                                    best_delta = (dx, dy)
                if best_sap_gain > 0:
                    actions[unit_id] = [5, best_delta[0], best_delta[1]]
                    continue

            # 2. Enemy avoidance
            enemy_positions = np.array(
                obs["obs"]["units"]["position"][self.opp_team_id]
            )
            enemy_mask = np.array(obs["obs"]["units_mask"][self.opp_team_id])
            if enemy_mask.any():
                closest_enemy_dist = np.inf
                for epos in enemy_positions[enemy_mask]:
                    dist = np.abs(current_pos - epos).sum()
                    if dist < closest_enemy_dist:
                        closest_enemy_dist = dist
                if closest_enemy_dist <= 2:
                    flee_dir = self.get_flee_direction(
                        current_pos, enemy_positions[enemy_mask]
                    )
                    actions[unit_id] = [flee_dir, 0, 0]
                    continue

            # 3. Relic collection
            if unit_id in self.unit_assignments:
                target_pos = self.unit_assignments[unit_id]
                move_dir = self.get_safe_direction(
                    current_pos, target_pos, map_features, sensor_mask
                )
                actions[unit_id] = [move_dir, 0, 0]
                continue

            # 4. Systematic exploration
            unexplored = np.argwhere(~self.explored_map)
            if len(unexplored) > 0:
                distances = np.abs(unexplored - current_pos).sum(axis=1)
                closest_idx = np.argmin(distances)
                target = unexplored[closest_idx]
                move_dir = self.get_safe_direction(
                    current_pos, target, map_features, sensor_mask
                )
                actions[unit_id] = [move_dir, 0, 0]
            else:
                actions[unit_id] = [0, 0, 0]

        return actions

    def get_flee_direction(self, current_pos, enemy_positions):
        avg_enemy = np.mean(enemy_positions, axis=0)
        dx = current_pos[0] - avg_enemy[0]
        dy = current_pos[1] - avg_enemy[1]
        if abs(dx) > abs(dy):
            return 2 if dx < 0 else 4
        else:
            return 3 if dy < 0 else 1

    def get_safe_direction(self, current_pos, target_pos, map_features, sensor_mask):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        if dx == 0 and dy == 0:
            return 0

        # Prefer primary direction
        if abs(dx) > abs(dy):
            primary_dir = 2 if dx > 0 else 4
        else:
            primary_dir = 3 if dy > 0 else 1

        # Check for obstacles
        new_x = current_pos[0] + (
            1 if primary_dir == 2 else -1 if primary_dir == 4 else 0
        )
        new_y = current_pos[1] + (
            1 if primary_dir == 3 else -1 if primary_dir == 1 else 0
        )
        if self.is_passable(new_x, new_y, map_features, sensor_mask):
            return primary_dir

        # Try alternative directions
        for alt_dir in [1, 3, 2, 4]:
            if alt_dir == primary_dir:
                continue
            new_x = current_pos[0] + (1 if alt_dir == 2 else -1 if alt_dir == 4 else 0)
            new_y = current_pos[1] + (1 if alt_dir == 3 else -1 if alt_dir == 1 else 0)
            if self.is_passable(new_x, new_y, map_features, sensor_mask):
                return alt_dir
        return 0

    def is_passable(self, x, y, map_features, sensor_mask):
        if not (
            0 <= x < self.env_cfg["map_width"] and 0 <= y < self.env_cfg["map_height"]
        ):
            return False
        if sensor_mask[x][y] and map_features["tile_type"][x][y] == 2:
            return False
        return True
