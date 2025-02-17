from lux.utils import direction_to
import numpy as np


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        # Tracking exploration targets and discovered relic nodes.
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Decide what actions to send to each available unit.
        This agent prioritizes:
         1. Evading enemy units that are too close.
         2. Moving toward the closest discovered relic node.
         3. Exploring randomly if no relic nodes are found.
        """
        # Own team information.
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])

        # Relic node info.
        observed_relic_node_positions = np.array(obs["relic_nodes"])
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])
        team_points = np.array(obs["team_points"])

        # Enemy team info (if available).
        enemy_unit_mask = None
        enemy_positions = None
        if "units_mask" in obs and len(obs["units_mask"]) > self.opp_team_id:
            enemy_unit_mask = np.array(obs["units_mask"][self.opp_team_id])
            enemy_positions = np.array(obs["units"]["position"][self.opp_team_id])

        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # Update relic node information.
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        for rid in visible_relic_node_ids:
            if rid not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(rid)
                self.relic_node_positions.append(observed_relic_node_positions[rid])

        # Decide actions for each unit.
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]

            # 1. Enemy avoidance: if an enemy unit is extremely close, move away.
            if enemy_unit_mask is not None and enemy_positions is not None:
                visible_enemy_ids = np.where(enemy_unit_mask)[0]
                enemy_close = False
                for e_id in visible_enemy_ids:
                    enemy_pos = enemy_positions[e_id]
                    distance = abs(unit_pos[0] - enemy_pos[0]) + abs(
                        unit_pos[1] - enemy_pos[1]
                    )
                    if distance < 2:
                        enemy_close = True
                        # Move in the opposite direction.
                        dx = unit_pos[0] - enemy_pos[0]
                        dy = unit_pos[1] - enemy_pos[1]
                        if abs(dx) >= abs(dy):
                            move_dir = 2 if dx >= 0 else 4  # right or left.
                        else:
                            move_dir = 3 if dy >= 0 else 1  # down or up.
                        actions[unit_id] = [move_dir, 0, 0]
                        break
                if enemy_close:
                    continue  # Skip further decision-making for this unit.

            # 2. If relic nodes have been discovered, choose the closest.
            if len(self.relic_node_positions) > 0:
                # Compute Manhattan distances to all discovered relic nodes.
                distances = [
                    abs(unit_pos[0] - r[0]) + abs(unit_pos[1] - r[1])
                    for r in self.relic_node_positions
                ]
                nearest_idx = np.argmin(distances)
                nearest_relic_node_position = self.relic_node_positions[nearest_idx]
                manhattan_distance = distances[nearest_idx]
                if manhattan_distance <= 4:
                    # If near, add randomness to cover the relic node's area.
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    actions[unit_id] = [
                        direction_to(unit_pos, nearest_relic_node_position),
                        0,
                        0,
                    ]
            else:
                # 3. No relic nodes discovered: perform random exploration.
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (
                        np.random.randint(0, self.env_cfg["map_width"]),
                        np.random.randint(0, self.env_cfg["map_height"]),
                    )
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [
                    direction_to(unit_pos, self.unit_explore_locations[unit_id]),
                    0,
                    0,
                ]

        return actions
