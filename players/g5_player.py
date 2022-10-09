import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
import statistics.mean as mean


class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, total_days: int, spawn_days: int,
                 player_idx: int, spawn_point: sympy.geometry.Point2D, min_dim: int, max_dim: int, precomp_dir: str) \
            -> None:
        """Initialise the player with given skill.

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                total_days (int): total number of days, the game is played
                spawn_days (int): number of days after which new units spawn
                player_idx (int): index used to identify the player among the four possible players
                spawn_point (sympy.geometry.Point2D): Homebase of the player
                min_dim (int): Minimum boundary of the square map
                max_dim (int): Maximum boundary of the square map
                precomp_dir (str): Directory path to store/load pre-computation
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.player_idx = player_idx


    def force_vec(self, p1, p2):
        v = p1 - p2
        mag = np.linalg.norm(v)
        unit = v / mag
        return unit, mag

    def to_polar(self, p):
        x, y = p
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)

    def normalize(self, v):
        return v / np.linalg.norm(v)

    def repelling_force(self, p1, p2):
        dir, mag = self.force_vec(p1, p2)
        # Inverse magnitude: closer things apply greater force
        return dir * 1 / (mag)


    def play(self, unit_id, unit_pos, map_states, current_scores, total_scores) -> [tuple[float, float]]:
        """Function which based on current game state returns the distance and angle of each unit active on the board

                Args:
                    unit_id (list(list(str))): contains the ids of each player's units (unit_id[player_idx][x])
                    unit_pos (list(list(float))): contains the position of each unit currently present on the map
                                                    (unit_pos[player_idx][x])
                    map_states (list(list(int)): contains the state of each cell, using the x, y coordinate system
                                                    (map_states[x][y])
                    current_scores (list(int)): contains the number of cells currently occupied by each player
                                                    (current_scores[player_idx])
                    total_scores (list(int)): contains the cumulative scores up until the current day
                                                    (total_scores[player_idx]

                Returns:
                    List[Tuple[float, float]]: Return a list of tuples consisting of distance and angle in radians to
                                                move each unit of the player
                """

        own_units = list(
            zip(
                unit_id[self.player_idx],
                [sympy_p_float(pos) for pos in unit_pos[self.player_idx]],
            )
        )
        enemy_units_locations = [
            sympy_p_float(unit_pos[player][i])
            for player in range(len(unit_pos))
            for i in range(len(unit_pos[player]))
            if player != self.player_idx
        ]

        ENEMY_INFLUENCE = 1
        HOME_INFLUENCE = 20
        ALLY_INFLUENCE = 0.5



        moves = []
        num_enemy=0

        safe_units=[]
        dangerous_units=[]

        if self.player_idx == 0:
            #check if unit is in danger
            for unit_id, unit_pos in own_units:
                # ally_direction_x=[]
                # ally_direction_y=[]

                nearby_cells_x = list(range(min(1, int(unit_pos.x) - 5), min(int(unit_pos.x) + 5, 100)))
                nearby_cells_y = list(range(min(1, int(unit_pos.y) - 5), min(int(unit_pos.y) + 5, 100)))
                for x in nearby_cells_x:
                    for y in nearby_cells_y:
                        if map_states[x][y] != self.player_idx:
                            num_enemy += 1
                        #else:
                            #ally_direction_x.append(x)
                            #ally_direction_y.append(y)

                danger_level = num_enemy / (len(nearby_cells_x) * len(nearby_cells_y))
                if danger_level>=0.75:
                    dangerous_units.append([unit_id,unit_pos])
                else:
                    safe_units.append([unit_id,unit_pos])


            for unit_id, unit_pos in own_units:
                self.debug(f"Unit {unit_id}", unit_pos)

                #TODO: determine if enemy force is needed
                enemy_unit_forces = [
                    self.repelling_force(unit_pos, enemy_pos)
                    for enemy_pos in enemy_units_locations
                ]
                enemy_force = np.add.reduce(enemy_unit_forces)

                #TODO: separate safe ally and dangerous ally
                safe_ally_force = [
                    self.repelling_force(unit_pos, ally_pos)
                    for ally_id,ally_pos in safe_units
                    if ally_id != unit_id
                ]
                safe_ally_force = np.add.reduce(safe_ally_force)

                danger_ally_force = [
                    -self.repelling_force(unit_pos, ally_pos)
                    for ally_id, ally_pos in dangerous_units
                    if ally_id != unit_id
                ]
                danger_ally_force = np.add.reduce(danger_ally_force)

                #TODO: boundary
                boundary_force=[0,0]
                if unit_pos.x <= 5:
                    if unit_pos.y <=5:
                        boundary_force = self.repelling_force(unit_pos, [0, 0])
                    elif unit_pos.y >=95:
                        boundary_force = self.repelling_force(unit_pos, [0, 100])
                    else:
                        boundary_force = self.repelling_force(unit_pos, [0, unit_pos.y])
                elif unit_pos.x >= 95:
                    if unit_pos.y <=5:
                        boundary_force = self.repelling_force(unit_pos, [100, 0])
                    elif unit_pos.y >=95:
                        boundary_force = self.repelling_force(unit_pos, [100, 100])
                    else:
                        boundary_force = self.repelling_force(unit_pos, [100, unit_pos.y])


                home_force = self.repelling_force(unit_pos, self.homebase)
                self.debug("\tEnemy force:", enemy_force)
                self.debug("\tHome force:", home_force)

                total_force = self.normalize(
                    (enemy_force * ENEMY_INFLUENCE)
                    + (home_force * HOME_INFLUENCE)
                    + (safe_ally_force * ALLY_INFLUENCE)
                    + (danger_ally_force * ALLY_INFLUENCE)
                    + (boundary_force)
                )
                #TODO: what to do if the current unit is in danger, retreat?
                if danger_level > 0.75:
                    # safe_x=int(mean(ally_direction_x))
                    # safe_y=int(mean(ally_direction_y))
                    # safe_retreat_force = self.repelling_force(unit_pos[self.player_idx][i], [safe_x, safe_y])
                    # moves.append(self.to_polar(safe_retreat_force))

                # if the current unit is safe
                else:
                    moves.append(self.to_polar(total_force))



        elif self.player_idx == 1:
            angles = [- np.pi / 4, - np.pi / 2, 0]
            for i in range(len(unit_id[self.player_idx])):
                distance = 1
                angle = angles[i % 3]
                moves.append((distance, angle))
        elif self.player_idx == 2:
            anglesEvenLayer = [- np.pi / 2, -3 * np.pi / 4, - np.pi]
            anglesOddLayer = [ -5 * np.pi / 8, -7 * np.pi / 8]
            n = len(unit_id[self.player_idx])
            r = n % 5
            for i in range(n-r):
                if i % 5 <= 2:
                    distance = 1
                    angle = anglesEvenLayer[i % 5]
                    moves.append((distance, angle))
                else:
                    # i % 5 > 2
                    distance = 1
                    angle = anglesOddLayer[(i % 5) - 3]
                    moves.append((distance, angle))
            if r == 1:
                moves.append((0, 0))
            elif r == 2:
                moves.append((0, 0))
                moves.append((0, 0))
            elif r == 3:
                distance = 1
                for j in range(3):
                    angle = anglesEvenLayer[j]
                    moves.append((distance, angle))
            elif r == 4:
                distance = 1
                for j in range(3):
                    angle = anglesEvenLayer[j]
                    moves.append((distance, angle))
                moves.append((0, 0))
        else:
            for i in range(len(unit_id[self.player_idx])):
                moves.append((0, 0))

        return moves
