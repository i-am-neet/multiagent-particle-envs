import numpy as np
from multiagent.core import World, Agent, Landmark, Wall, Background
from multiagent.scenario import BaseScenario
from multiagent.scenarios.room_arguments import RoomArgs
import yaml
import os
import math
import rospy
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from multiagent.algos.a_star import AStarPlanner

# room_args = get_room_args()
room_args = RoomArgs()

# set obstacle positions
ox, oy = [], []
for i in range(-100, 100):
    ox.append(i)
    oy.append(-100.0)
for i in range(-100, 100):
    ox.append(-100.0)
    oy.append(i)
for i in range(-100, 100):
    ox.append(i)
    oy.append(100.0)
for i in range(-100, 100):
    ox.append(100.0)
    oy.append(i)
for i in range(-100, -35):
    ox.append(0.0)
    oy.append(i)
for i in range(35, 100):
    ox.append(0.0)
    oy.append(i)

grid_size = 5.0  # [m]
robot_radius = 1.0  # [m]

a_star = AStarPlanner(ox, oy, grid_size, robot_radius)

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
            valid_pos = False
            while (not valid_pos):
                p = np.random.uniform(-0.8, +0.8, world.dim_p)
                tmpA = Agent()
                tmpA.state.p_pos = p
                tmpA.size = 0.07
                collide_walls = [ self.check_wall_collision(wall, tmpA) for wall in world.walls ]
                collide_agents = [ np.linalg.norm(world.agents[j].state.p_pos - tmpA.state.p_pos) < tmpA.size*2 for j in range(i)]
                if not any(collide_walls) and not any(collide_agents):
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
                tmpL.size = 0.07 # gap size
                collide_walls = [ self.check_wall_collision(wall, tmpL) for wall in world.walls ]
                collide_agents = [ np.linalg.norm(world.landmarks[j].state.p_pos - tmpL.state.p_pos) < tmpL.size*2 for j in range(i)]
                if not any(collide_walls) and not any(collide_agents):
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

    def check_wall_collision(self, wall, agent):
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
        # rew -= dist
        # rew -= 1
        rew -= 1 / (1 + np.exp(-dist+1)) # sigmoid

        # Arrived
        # if dist < 0.1:
        #     rew += dist
        #     rew += 5 * np.exp(-dist**0.05)

        collision = False
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and agent.name != a.name:
                    collision = True
            for w in world.walls:
                if self.check_wall_collision(w, agent):
                    collision = True
        if collision:
            rew -= 1
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

        # A*
        scale = 0.01
        p_start = tuple((agent.state.p_pos / scale).astype(int))
        p_goal = tuple((entity.state.p_pos / scale).astype(int))

        route = a_star.planning(p_start[0], p_start[1], p_goal[0], p_goal[1])
        route = route[:-1] # abandon last elem (first point)

        # next point of route according agent's pos (unit vector)
        next_dir = []
        if len(route) != 0:
            next_p_vec = np.array(route[-1]) - np.array(p_start)
            next_u_vec = next_p_vec / np.linalg.norm(next_p_vec)
        else:
            next_u_vec = np.array([0, 0])
        next_dir.append(next_u_vec)

        # rencent points of route according agent's pos (particle env coord)
        future_size = 5
        next_points = (np.array(route[-future_size:]) - np.array(p_start))*scale if len(route) != 0 else np.array([0, 0]*future_size)
        if len(next_points) < future_size:
            next_points = np.append(next_points, [0, 0]*(future_size - len(next_points)))
        else:
            next_points = next_points.flatten()

        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + ranges)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + ranges + next_dir + [next_points])

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

    def make_plan(self, start, goal):
        """
        Call service /robot0_move_base/make_plan
        srv: nav_msgs/GetPlan
        """
        scale_to_gazebo = 3
        q = quaternion_from_euler(0.0, 0.0, 0.0)
        ms = PoseStamped()
        mg = PoseStamped()
        ms.header.frame_id = "map"
        ms.pose.position.x = start[0]*scale_to_gazebo
        ms.pose.position.y = start[1]*scale_to_gazebo
        ms.pose.orientation = Quaternion(*q)
        mg.header.frame_id = "map"
        mg.pose.position.x = goal[0]*scale_to_gazebo
        mg.pose.position.y = goal[1]*scale_to_gazebo
        mg.pose.orientation = Quaternion(*q)

        req = GetPlan()
        req.start = ms
        req.goal = mg
        req.tolerance = 1.0

        path = []
        try:
            get_plan = rospy.ServiceProxy('/robot0_move_base/make_plan', GetPlan)
            resp = get_plan(req.start, req.goal, req.tolerance)
            for p in resp.plan.poses:
                x = p.pose.position.x / scale_to_gazebo
                y = p.pose.position.y / scale_to_gazebo
                path.append((x, y))
            return path
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))
