import numpy as np
from multiagent.core import World, Agent, Landmark, Wall, Background
from multiagent.scenario import BaseScenario
from multiagent.scenarios.room_arguments import RoomArgs

# room_args = get_room_args()
room_args = RoomArgs()

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # add walls
        world.walls = [Wall() for i in range(room_args.wall_num)]
        for i, wall in enumerate(world.walls):
            wall.name = 'wall %d' % i
            wall.collide = True
            wall.movable = False
        # add background
        world.backgrounds = [Background()]
        for i, bg in enumerate(world.backgrounds):
            bg.name = 'background %d' % i
            bg.collide = False
            bg.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # random properties for walls
        for i, wall in enumerate(world.walls):
            wall.color = np.array([0, 0.7, 0.0])
        # random properties for background
        for i, bg in enumerate(world.backgrounds):
            bg.color = np.array([0.0, 0.0, 0.0])
        # set random initial states
        for i, agent in enumerate(world.agents):
            p = np.array([[0.8, 0.8], [0.8, -0.8], [-0.8, 0.8], [-0.8, -0.8]])
            agent.state.p_pos = p[i]
            # agent.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            p = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]])
            # landmark.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p)
            landmark.state.p_pos = p[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, wall in enumerate(world.walls):
            # wall.state.p_pos = np.array(room_args.wall_centers[i]) + world.landmarks[0].state.p_pos
            wall.state.p_pos = np.array(room_args.wall_info['wall_centers'][i])
            wall.state.p_vel = np.zeros(world.dim_p)
            wall.W = room_args.wall_info['wall_shapes'][i][0]
            wall.L = room_args.wall_info['wall_shapes'][i][1]

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

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
