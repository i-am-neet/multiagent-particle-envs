import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from multiagent.utils.env_util import EventCounter
import random
import time

map_width = 600
map_height = 600

LAST_IMAGE_SIZE = 81

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True,
                 expert_callback=None):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.expert_callback = expert_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        # self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        self.force_discrete_action = True
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # for experiment logging
        self.time_n = [time.time()]*self.n
        self.cost_time_n = [0]*self.n
        self.collision_times_n = [0]*self.n

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                # u_action_space = spaces.Discrete(5)
                u_action_space = spaces.Discrete(9)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # # communication action space
            # if self.discrete_action_space:
            #     c_action_space = spaces.Discrete(world.dim_c)
            # else:
            #     c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            # if not agent.silent:
            #     total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world)) + LAST_IMAGE_SIZE
            # obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
            self.viewers_world = [None]
        else:
            self.viewers = [None] * self.n
            self.viewers_world = [None] * self.n
        self._reset_render()

    def get_info(self):
        return self.cost_time_n, self.collision_times_n

    # get expert action
    def get_expert_action_n(self):
        expert_action_n = []
        self.agents = self.world.policy_agents
        if self.expert_callback is None:
            return [np.zeros(9) for _ in range(len(self.agents))]
        # get expert action from each agent
        for agent in self.agents:
            expert_action_n.append(self.expert_callback(agent, self.world))

        return expert_action_n

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # check done_flag from simple_spread_room
        for i, d in enumerate(done_n):
            if len(d) == 2:
                if all(d):
                    # FIXME dont reset
                    print(f"agent {i} got trouble")
                    #done_n = [True] * len(done_n)
                    #info_n['n'][i] = f"agent {i} got trouble, resetting"
                    #break
                    done_n[i] = d[0]
                else:
                    done_n[i] = d[0]

        # experiment logging
        for j, i in enumerate(info_n['n']):
            if i['done'] and self.cost_time_n[j]==0:
                self.cost_time_n[j] = time.time() - self.time_n[j]
            self.collision_times_n[j] = i['collision']

        # all agents get total reward in cooperative case
        # reward = np.sum(reward_n)
        # if self.shared_reward:
        #     reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    # @EventCounter(schedules=[100, 200, 300, 400], matters=[0, 1, 2, 3, 4]) # Scheduling room
    @EventCounter(schedules=[8e+3, 16e+3, 32e+3, 64e+3], matters=[1, 2, 3, 4, 5]) # Scheduling room # ep
    def reset(self, testing=False):
        # experiment logging
        self.time_n = [time.time()]*self.n
        self.cost_time_n = [0]*self.n

        # reset world
        if testing:
            self.reset_callback(self.world, random.choice([0, 1, 2, 3, 4, 5]))
        else:
            self.reset_callback(self.world, self.reset.matter, True) # matter is room id
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    if self.force_discrete_action:
                        u_index = np.argmax(action[0])
                        if u_index == 0: agent.action.u = np.array([ 0,  0], dtype=np.float32)
                        if u_index == 1: agent.action.u = np.array([ 1,  0], dtype=np.float32)
                        if u_index == 2: agent.action.u = np.array([ 1,  1], dtype=np.float32)
                        if u_index == 3: agent.action.u = np.array([ 0,  1], dtype=np.float32)
                        if u_index == 4: agent.action.u = np.array([-1,  1], dtype=np.float32)
                        if u_index == 5: agent.action.u = np.array([-1,  0], dtype=np.float32)
                        if u_index == 6: agent.action.u = np.array([-1, -1], dtype=np.float32)
                        if u_index == 7: agent.action.u = np.array([ 0, -1], dtype=np.float32)
                        if u_index == 8: agent.action.u = np.array([ 1, -1], dtype=np.float32)
                    else:
                        agent.action.u[0] += action[0][1] - action[0][2]
                        agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        # if not agent.silent:
        #     # communication action
        #     if self.discrete_action_input:
        #         agent.action.c = np.zeros(self.world.dim_c)
        #         agent.action.c[action[0]] = 1.0
        #     else:
        #         agent.action.c = action[0]
        #     action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None
        self.render_geoms_world = None
        self.render_geoms_xform_world = None

    def get_world_array(self):
        for i in range(len(self.viewers_world)):
            # create viewers (if necessary)
            if self.viewers_world[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers_world[i] = rendering.Viewer(map_width,map_height)

        # create rendering geometry
        if self.render_geoms_world is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms_world = []
            self.render_geoms_xform_world = []
            for entity in self.world.entities_world:
                l, r, t, b = -entity.W/2, entity.W/2, entity.L/2, -entity.L/2
                geom = rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)])

                xform = rendering.Transform()

                geom.set_color(*entity.color)
                geom.add_attr(xform)

                self.render_geoms_world.append(geom)
                self.render_geoms_xform_world.append(xform)

            # add geoms to viewer
            for viewer in self.viewers_world:
                viewer.geoms = []
                for geom in self.render_geoms_world:
                    viewer.add_geom(geom)

        results_world = []
        for i in range(len(self.viewers_world)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers_world[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities_world):
                self.render_geoms_xform_world[e].set_translation(*entity.state.p_pos)
            results_world.append(self.viewers_world[i].get_array())

        return results_world

    # render environment
    def render(self, mode=''):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(map_width,map_height)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            import copy
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                if 'wall' in entity.name:
                    l, r, t, b = -entity.W/2, entity.W/2, entity.L/2, -entity.L/2
                    geom = rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)])
                else:
                    geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'landmark' in entity.name:
                    geom.set_color(*entity.color, alpha=0.2)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # print(f"Bounds: {pos[0]-cam_range}, {pos[0]+cam_range}, {pos[1]-cam_range}, {pos[1]+cam_range}")
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
