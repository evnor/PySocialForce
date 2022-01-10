# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""
from pysocialforce.utils import DefaultConfig
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces
import numpy as np


class Simulator:
    """Simulate social force model.

    ...

    Attributes
    ----------
    state : np.ndarray [n, 6] or [n, 7]
        Each entry represents a pedestrian state, (x, y, v_x, v_y, d_x, d_y, [tau])
    goals : np.ndarray [n, 2]
        Each pair of entries represents a waypoint after the first, given in state
    obstacles : np.ndarray
        Environmental obstacles
    groups : List of Lists
        Group members are denoted by their indices in the state
    config : Dict
        Loaded from a toml config file
    max_speeds : np.ndarray
        Maximum speed of pedestrians
    forces : List
        Forces to factor in during navigation

    Methods
    ---------
    capped_velocity(desired_velcity)
        Scale down a desired velocity to its capped speed
    step()
        Make one step
    """

    def __init__(self, state, goals=None, groups=None, obstacles=None, config_file=None):
        self.config = DefaultConfig()
        if config_file:
            self.config.load_config(config_file)
        # TODO: load obstacles from config
        self.scene_config = self.config.sub_config("scene")
        # initiate obstacles
        self.env = EnvState(obstacles, self.config("resolution", 10.0))

        # TODO: support more than one waypoint
        # initiate agents
        self.peds = PedState(state, goals, groups, self.config)

        # construct forces
        self.forces = self.make_forces(self.config)

        self.autonomy = []

    def make_forces(self, force_configs):
        """Construct forces"""
        force_list = [
            forces.DesiredForce(),
            forces.SocialForce(),
            forces.ObstacleForce(),
            # forces.PedRepulsiveForce(),
            # forces.SpaceRepulsiveForce(),
        ]
        group_forces = [
            forces.GroupCoherenceForceAlt(),
            forces.GroupRepulsiveForce(),
            forces.GroupGazeForceAlt(),
        ]
        if self.scene_config("enable_group"):
            force_list += group_forces
        
        self.autonomous_forces = [
            forces.DesiredForce, 
            forces.GroupCoherenceForceAlt,
            forces.GroupRepulsiveForce,
            forces.GroupGazeForceAlt
        ]

        # initiate forces
        for force in force_list:
            force.init(self, force_configs)

        return force_list

    def compute_forces(self):
        """compute forces"""
        result = None
        total_force = None
        total_aut_force = None
        mask = self.get_non_boarded_mask()
        for force in self.forces:
            f = force.get_force()
            result = f if result is None else f+result
            mag = np.linalg.norm(f[mask], axis=1)
            total_force = mag if total_force is None else mag+total_force
            if any(isinstance(force, aut_force) for aut_force in self.autonomous_forces):
                total_aut_force = mag if total_aut_force is None else mag+total_aut_force
        # self.autonomy.append(np.linalg.norm(total_aut_force[mask], axis=1)/np.linalg.norm(total_force[mask], axis=1))
        self.autonomy.append(total_aut_force/total_force)
        return result
        # return sum(map(lambda x: x.get_force(), self.forces))

    def get_states(self):
        """Expose whole state"""
        return self.peds.get_states()

    def get_length(self):
        """Get simulation length"""
        return len(self.get_states()[0])

    def get_obstacles(self):
        return self.env.obstacles

    def step_once(self):
        """step once"""
        self.peds.step(self.compute_forces())

    def get_non_boarded_mask(self):
        return np.squeeze(np.concatenate((self.get_states()[0][-1, self.go_out_train:, 1:2] > -1, self.get_states()[0][-1, :self.go_out_train, 1:2] < 4), axis=0))
        
    def simulate_boarding(self, go_out_train):
        N_steps = 0
        self.go_out_train = go_out_train
        while np.any(self.get_non_boarded_mask()):
            self.step()
            N_steps += 1
        return N_steps

    def step(self, n=1):
        """Step n time"""
        for _ in range(n):
            self.step_once()
        return self
