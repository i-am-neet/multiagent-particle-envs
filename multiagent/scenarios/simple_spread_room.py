import numpy as np
from multiagent.core import World, Agent, Landmark, Wall, Background
from multiagent.scenario import BaseScenario
from multiagent.scenarios.room_arguments import RoomArgs
import yaml
import os
import math

# room_args = get_room_args()
room_args = RoomArgs()

cwd = os.path.dirname(__file__)
with open(cwd+'/color_coded/colors-glasbey.yaml', 'r') as f:
    color_args = yaml.full_load(f)

class Scenario(BaseScenario):
    def make_world(self, amount):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = amount
        num_landmarks = amount
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # add walls
        world.walls = [Wall() for i in range(room_args.wall_num)]
        for i, wall in enumerate(world.walls):
            wall.name = 'wall %d' % i
            wall.collide = True
            wall.movable = False
        # # add background
        # world.backgrounds = [Background()]
        # for i, bg in enumerate(world.backgrounds):
        #     bg.name = 'background %d' % i
        #     bg.collide = False
        #     bg.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array(color_args[f'color_{i}'])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.color = np.array(color_args[f'color_{i}'])
        # random properties for walls
        for i, wall in enumerate(world.walls):
            wall.color = np.array([0, 0.7, 0.0])
        # TODO change room
        # set random initial states
        for i, wall in enumerate(world.walls):
            # wall.state.p_pos = np.array(room_args.wall_centers[i]) + world.landmarks[0].state.p_pos
            wall.state.p_pos = np.array(room_args.wall_info['wall_centers'][i])
            wall.state.p_vel = np.zeros(world.dim_p)
            wall.W = room_args.wall_info['wall_shapes'][i][0]
            wall.L = room_args.wall_info['wall_shapes'][i][1]
        for i, agent in enumerate(world.agents):
            # p = np.array([[0.8, 0.8], [0.8, -0.8], [-0.8, 0.8], [-0.8, -0.8]])
            # agent.state.p_pos = p[i]
            valid_pos = False
            while (not valid_pos):
                p = np.random.uniform(-0.8, +0.8, world.dim_p)
                tmpA = Agent()
                tmpA.state.p_pos = p
                tmpA.size = 0.07
                dd_walls = [ self.get_dist_min_to_wall(wall, tmpA) for wall in world.walls ]
                if not any(dd_walls):
                    break
            # agent.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
            agent.state.p_pos = p
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            # p = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]])
            # landmark.state.p_pos = p[i]
            valid_pos = False
            while (not valid_pos):
                p = np.random.uniform(-0.8, +0.8, world.dim_p)
                tmpL = Landmark()
                tmpL.state.p_pos = p
                tmpL.size = 0.07
                dd_walls = [ self.get_dist_min_to_wall(wall, tmpL) for wall in world.walls ]
                if not any(dd_walls):
                    break
            landmark.state.p_pos = p
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def get_dist_min_to_wall(self, wall, agent):
        vec = np.append(agent.state.p_pos - wall.state.p_pos, 0)
        corner1 = np.array([wall.W / 2, wall.L / 2, 0])
        corner2 = np.array([-wall.W / 2, wall.L / 2, 0])
        flag1 = np.dot(np.cross(corner1, vec), np.cross(corner1, corner2))
        flag2 = np.dot(np.cross(corner2, vec), np.cross(corner2, corner1))
        if ((flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0)) and abs(vec[0]) <= wall.W / 2:
            delta_pos = np.array([0, vec[1]])
            dist = np.linalg.norm(delta_pos)
            dist_min = agent.size + wall.L / 2
        elif ((flag1 > 0 and flag2 < 0) or (flag1 < 0 and flag2 > 0)) and abs(vec[1]) <= wall.L / 2:
            delta_pos = np.array([vec[0], 0])
            dist = np.linalg.norm(delta_pos)
            dist_min = agent.size + wall.W / 2
        else:
            nearest_corner = np.array([wall.W / 2 * ((vec[0] > 0) * 2 - 1),
                                       wall.L / 2* ((vec[1] > 0) * 2 - 1)])
            delta_pos = vec[:2] - nearest_corner
            dist = np.linalg.norm(delta_pos)
            dist_min = agent.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        l = world.landmarks[agent.id]
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        rew -= dist

        collision = False
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and agent.name != a.name:
                    collision = True
            for w in world.walls:
                if self.get_dist_min_to_wall(w, agent):
                    collision = True
        if collision:
            rew -= 10
        return rew

    def observation(self, agent, world):
        # get lidar scanner data
        ranges = [self.lidar(agent, world, 12)]

        entity_pos = []
        # for entity in world.landmarks:  # world.entities: # get positions of all entities in this agent's reference frame
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        entity = world.landmarks[agent.id]
        entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + ranges)

    def lidar(self, agent, world, num_scan):
        """
        Return lidar scan's ranges [] by agent's position & scan_num
        degree gap is 360/scan_num
        """
        ranges = []
        deg_gap = 2*np.pi / num_scan
        for s in range(num_scan):
            deg = s*deg_gap
            lidar_dir = np.array([np.cos(deg), np.sin(deg)])
            agent_pos = agent.state.p_pos
            p_ranges = []
            for wall in world.walls:
                wall_pos = wall.state.p_pos
                wall_dir = np.array([1, 0] if wall.W > wall.L else [0, 1])
                inter_pos = self.intersection(agent_pos, lidar_dir, wall_pos, wall_dir)
                if inter_pos is None:
                    d = math.inf
                else:
                    # Check inter_pos is on the wall
                    check = abs(inter_pos - wall_pos) <= np.array([wall.W/2, wall.L/2])
                    d = np.linalg.norm(inter_pos - agent_pos) if all(check) else math.inf
                p_ranges.append(d)
            ranges.append(min(p_ranges))
        ranges = [2.83 if i > 2.83 else i for i in ranges] # longest length of laser
        return np.array(ranges)

    def intersection(self, p1, p1_dir, p2, p2_dir):
        """
        Calculate vectors' intersction point
        return np.array(POINT)
        """
        try:
            A = np.column_stack((p1_dir, p2_dir*-1))
            B = p2 - p1
            ans = np.linalg.inv(A).dot(B)
            return p1+p1_dir*ans[0] if ans[0] > 0 else None
        except:
            return None
