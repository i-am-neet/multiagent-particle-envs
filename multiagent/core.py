import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # collision times' counter
        self.collision_times = 0

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # id
        self.id = 0
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = 1.0 #None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 5.0 # 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

class Wall(Entity):
    def __init__(self):
        super(Wall, self).__init__()
        self.W = 0.3
        self.L = 1.0

class Background(Entity):
    def __init__(self):
        super(Background, self).__init__()
        self.img_path = "/home/noetic-neet/mapf_ws/src/gazebo/robot_gazebo/maps/resolution=001/home-area.png"

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls =[]
        self.backgrounds = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.55 #0.25
        # contact response parameters
        self.contact_force = 1e+3 # 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.landmarks + self.agents + self.walls

    # return environment entities ONLY
    @property
    def entities_world(self):
        # return self.agents + self.landmarks + self.walls + self.backgrounds
        return self.walls

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,entity in enumerate(self.entities):
            if "agent" not in entity.name: continue
            if entity.movable:
                noise = np.random.randn(*entity.action.u.shape) * entity.u_noise if entity.u_noise else 0.0
                p_force[i] = entity.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                if not entity_a.movable and not entity_b.movable: continue # PeihongYu
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                # if any([round(elem, 2) for elem in p_force[i]]):
                #     print(f'{p_force[i][0]:.2f}, {p_force[i][1]:.2f}')
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_pos = np.clip(entity.state.p_pos, -0.98, 0.98)

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        if 'wall' in entity_a.name and 'wall' in entity_b.name:
            return [None, None] # don't collide against between walls
        if 'wall' in entity_a.name: # PeihongYu
            delta_pos, dist, dist_min = self.get_dist_min_to_wall(entity_b, entity_a)
        elif 'wall' in entity_b.name:
            delta_pos, dist, dist_min = self.get_dist_min_to_wall(entity_a, entity_b)
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # FIXME little weird 
    def get_dist_min_to_wall(self, agent, wall):
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
        return delta_pos, dist, dist_min