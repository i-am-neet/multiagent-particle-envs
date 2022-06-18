import numpy as np
from multiagent.core import World, Agent, Landmark, Wall, Background
from multiagent.scenario import BaseScenario
from multiagent.scenarios.room_arguments import RoomArgs
import yaml
import os
import math
# FIXME cannot import ros w/ tkinter
# import rospy
# from nav_msgs.srv import GetPlan
# from geometry_msgs.msg import PoseStamped, Quaternion
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
from multiagent.algos.a_star import AStarPlanner
from multiagent.utils.env_util import EventCounter
import time

room_args = RoomArgs()
room_args.get_room(0)

a_star = AStarPlanner(room_args.ox, room_args.oy, room_args.grid_size, room_args.robot_radius)

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

        self.route_n = [None]*amount
        self.done_flag = False

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

    def change_room(self, world, room_num):
        assert room_num < room_args.max_room_num
        # change A* obstacles
        room_args.get_room(room_num)
        a_star.change_obstacle(room_args.ox, room_args.oy)
        # re-constuct particle env
        world.walls = [Wall() for i in range(room_args.wall_num)]
        for i, wall in enumerate(world.walls):
            wall.name = 'wall %d' % i
            wall.collide = True
            wall.movable = False

    # TODO check schedules is working
    @EventCounter(schedules=[400, 800, 1600, 3200, 6400], matters=[0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
    def reset_world(self, world, room_num=0, scheduling=False):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array(color_args[f'color_{i}'])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.color = np.array(color_args[f'color_{i}'])
        # change room
        # room_num = 1 # Testing code
        if room_num != room_args.room_num:
            print(f"########## Change to room {room_num}, Resetting schedule... ##########")
            self.reset_world.__func__.counter = 0
            self.reset_world.__func__.schedules=[400, 800, 1600, 3200, 6400]
            self.reset_world.__func__.matters=[0.2, 0.4, 0.8, 1.2, 1.6, 2.0]
            self.change_room(world, room_num)
        # print(self.reset_world.counter) # Testing code
        # random properties for walls
        for i, wall in enumerate(world.walls):
            wall.color = np.array([0, 0.7, 0.0])
        # set random initial states
        for i, wall in enumerate(world.walls):
            wall.state.p_pos = np.array(room_args.wall_centers[i])
            wall.state.p_vel = np.zeros(world.dim_p)
            wall.W = room_args.wall_shapes[i][0]
            wall.L = room_args.wall_shapes[i][1]
        for i, agent in enumerate(world.agents):
            while (True):
                p = np.random.uniform(-0.8, +0.8, world.dim_p)
                tmpA = Agent()
                tmpA.state.p_pos = p
                tmpA.size = 0.2
                collide_walls = self.check_wall_collision(world.walls, tmpA)
                collide_agents = [ np.linalg.norm(world.agents[j].state.p_pos - tmpA.state.p_pos) < tmpA.size for j in range(i)]
                if not any(collide_walls) and not any(collide_agents):
                    break
            # agent.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
            agent.state.p_pos = p
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.collision_times = 0
        for i, landmark in enumerate(world.landmarks):
            # p = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]])
            # landmark.state.p_pos = p[i]
            st = time.time()
            while (True):
                if time.time() - st > 5:
                    print(f"Take times!!! agent_pos: {world.agents[i].state.p_pos}, matter: {self.reset_world.matter}")
                    """
                    Stocks! try to escape w/ random pos...
                    """
                    p = np.random.uniform(-0.8, +0.8, world.dim_p)
                    print(f"Try to escape with {p}")
                    tmpL = Landmark()
                    tmpL.state.p_pos = p
                    tmpL.size = 0.2 # gap size
                    collide_walls = self.check_wall_collision(world.walls, tmpL)
                    collide_landmarks = [ np.linalg.norm(world.landmarks[j].state.p_pos - tmpL.state.p_pos) < tmpL.size for j in range(i)]
                    collide_agents = [ np.linalg.norm(a.state.p_pos - tmpL.state.p_pos) < tmpL.size for a in world.agents]
                    if not any(collide_walls) and not any(collide_landmarks) and not any(collide_agents):
                        break
                    st = time.time()
                if scheduling:
                    ap = world.agents[i].state.p_pos
                    matter = self.reset_world.matter
                    nx = (ap[0] - matter) if (ap[0] - matter) > -0.9 else -0.9
                    px = (ap[0] + matter) if (ap[0] + matter) < +0.9 else +0.9
                    ny = (ap[1] - matter) if (ap[1] - matter) > -0.9 else -0.9
                    py = (ap[1] + matter) if (ap[1] + matter) < +0.9 else +0.9
                    x = np.random.uniform(nx, px, 1)
                    y = np.random.uniform(ny, py, 1)
                    p = np.concatenate((x, y))
                else:
                    p = np.random.uniform(-0.8, +0.8, world.dim_p)
                tmpL = Landmark()
                tmpL.state.p_pos = p
                tmpL.size = 0.1 # gap size
                collide_walls = self.check_wall_collision(world.walls, tmpL)
                collide_landmarks = [ np.linalg.norm(world.landmarks[j].state.p_pos - tmpL.state.p_pos) < tmpL.size*2 for j in range(i)]
                collide_agents = [ np.linalg.norm(a.state.p_pos - tmpL.state.p_pos) < tmpL.size*2 for a in world.agents]
                if not any(collide_walls) and not any(collide_landmarks) and not any(collide_agents):
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

    def check_wall_collision(self, walls, agent):
        """
        Check whether wall collide with agent
        return each wall whether it is collided with agent
        return [BOOL, BOOL, ..., BOOL]
        """
        p_ranges = []
        for wall in walls:
            agent_pos = agent.state.p_pos
            wall_pos = wall.state.p_pos
            wall_dir = np.array([1, 0] if wall.W > wall.L else [0, 1])
            if all(wall_dir == np.array([1, 0])):
                lidar_dir = np.array([0, wall_pos[1] - agent_pos[1]])
            else:
                lidar_dir = np.array([wall_pos[0] - agent_pos[0], 0])
            if all(lidar_dir == np.array([0, 0])): # singular matrix
                inter_pos = agent_pos
            else:
                inter_pos = self.intersection(agent_pos, lidar_dir, wall_pos, wall_dir)
            if inter_pos is None:
                d = math.inf
            else:
                # Check inter_pos is on the wall
                check = abs(inter_pos - wall_pos) <= np.array([wall.W/2, wall.L/2])
                d = np.linalg.norm(inter_pos - agent_pos) if all(check) else math.inf
            p_ranges.append(d)
        dist_min = agent.size
        collisions = [r < dist_min for r in p_ranges]
        return collisions

    def check_wall_distances(self, walls, agent):
        """
        Check whether wall collide with agent
        return each wall whether it is collided with agent
        return [dist to wall1, dist to wall2, ..., dist to wallN]
        """
        p_ranges = []
        for wall in walls:
            agent_pos = agent.state.p_pos
            wall_pos = wall.state.p_pos
            wall_dir = np.array([1, 0] if wall.W > wall.L else [0, 1])
            if all(wall_dir == np.array([1, 0])):
                lidar_dir = np.array([0, wall_pos[1] - agent_pos[1]])
            else:
                lidar_dir = np.array([wall_pos[0] - agent_pos[0], 0])
            if all(lidar_dir == np.array([0, 0])): # singular matrix
                inter_pos = agent_pos
            else:
                inter_pos = self.intersection(agent_pos, lidar_dir, wall_pos, wall_dir)
            if inter_pos is None:
                d = math.inf
            else:
                # Check inter_pos is on the wall
                check = abs(inter_pos - wall_pos) <= np.array([wall.W/2, wall.L/2])
                d = np.linalg.norm(inter_pos - agent_pos) if all(check) else math.inf
            p_ranges.append(d)
        return p_ranges

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        l = world.landmarks[agent.id]
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        rew -= 1 / (1 + np.exp(-dist*4+2)) # sigmoid

        # Arrived
        if dist < 0.1:
            rew += np.exp(-dist*10)*5

        # Collision
        collision = False
        at_field = []
        if agent.collide:
            for a in world.agents:
                if a.name == agent.name: continue
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
                at_force = np.exp(-2*dist) #1 / (1 + np.exp((dist-1)*6))
                at_field.append(at_force)
                if self.is_collision(a, agent):
                    collision = True
            if any(self.check_wall_collision(world.walls, agent)):
                collision = True
            wall_dists = self.check_wall_distances(world.walls, agent)
            at_forces = np.exp(-2*np.array(wall_dists)).tolist()
            at_field += at_forces
        rew -= max(at_field)

        if collision:
            agent.collision_times += 1
            # rew -= 0.5
            rew -= 2 / (1 + np.exp(-agent.collision_times + 1))

        # Direction reward which compare A* and action
        scale = 0.01
        p_start = tuple((agent.state.p_pos / scale).astype(int))
        p_goal = tuple((l.state.p_pos / scale).astype(int))

        if self.route_n[agent.id]:
            route = self.route_n[agent.id]
        else:
            route = a_star.planning(p_start[0], p_start[1], p_goal[0], p_goal[1])

        if len(route) >= 2:
            next_p_vec = np.array(route[-2]) - np.array(route[-1])
            next_u_vec = next_p_vec / np.linalg.norm(next_p_vec, 1) # get Norm-1 distance
        else:
            next_u_vec = np.array([0, 0])

        u = agent.action.u
        a_u_vec = u / np.linalg.norm(u, 1) if any(u) else u

        if all(u == np.array([0, 0])):
            rew -= 0.8
        elif not all(a_u_vec == next_u_vec):
            rew -= 0.4

        return rew

    def done(self, agent, world):
        # define termination of episode
        done = False

        l = world.landmarks[agent.id]
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))

        if dist < 0.1:
            done = True

        if agent.collision_times > 10:
            done = True

        if self.done_flag:
            self.done_flag = False
            return True, True

        return done, False

    def observation(self, agent, world):
        neighbor_range = 0.3
        # get lidar scanner data
        ranges = [self.lidar(agent, world, 12)]

        landmark_pos = []
        # for entity in world.landmarks:  # world.entities: # get positions of all entities in this agent's reference frame
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        landmark = world.landmarks[agent.id]
        landmark_pos.append(landmark.state.p_pos - agent.state.p_pos)

        # Find out other's pos and goal
        other_pos = []
        other_goal = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos)
            other_goal.append(world.landmarks[other.id].state.p_pos)
        assert len(other_pos)==len(world.agents)-1
        assert len(other_goal)==len(world.agents)-1

        # Find the vector from agent to neighbor, and filter by neighbor_range
        neighbor_pos = []
        for op in other_pos:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - op)))
            if dist < neighbor_range:
                neighbor_pos.append(op - agent.state.p_pos)
            else:
                neighbor_pos.append(np.array([0, 0]))
        assert len(neighbor_pos)==len(other_pos)

        # A*
        scale = 0.01
        p_start = tuple((agent.state.p_pos / scale).astype(int))
        p_goal = tuple((landmark.state.p_pos / scale).astype(int))

        if np.linalg.norm(landmark_pos) < 0.1:
            route = []
        else:
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
        future_size = 8
        next_points = (np.array(route[-future_size:]) - np.array(p_start))*scale if len(route) != 0 else np.array([0, 0]*future_size)
        if len(next_points) < future_size:
            next_points = np.append(next_points, [0, 0]*(future_size - len(next_points)))
        else:
            next_points = next_points.flatten()

        # Find the vector from agent to neighbor's goal, and filter by neighbor_range
        neighbors_goal = []
        for og in other_goal:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - og)))
            if dist < neighbor_range:
                neighbors_goal.append(og - agent.state.p_pos)
            else:
                neighbors_goal.append(np.array([0, 0]))
        assert len(neighbors_goal)==len(other_goal)

        # Personal info: [agent.state.p_vel] + [agent.state.p_pos] + landmark_pos + ranges + next_dir + [next_points]
        # Collaborate info: other_pos + neighbors_goal
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmark_pos + ranges + next_dir + [next_points] + neighbor_pos + neighbors_goal)
        other_gridmap = self.other_gridmap(agent, other_pos)
        other_plan_gridmap = self.other_plan_gridmap(agent, other_pos, other_goal)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmark_pos + ranges + next_dir + [next_points] + other_gridmap + other_plan_gridmap)

    def expert_action(self, agent, world):
        # A*
        landmark = world.landmarks[agent.id]
        scale = 0.01
        p_start = tuple((agent.state.p_pos / scale).astype(int))
        p_goal = tuple((landmark.state.p_pos / scale).astype(int))

        route = a_star.planning(p_start[0], p_start[1], p_goal[0], p_goal[1])

        # if using this function, record route to avoid re-planing again
        self.route_n[agent.id] = route

        if len(route) >= 2:
            next_p_vec = np.array(route[-2]) - np.array(route[-1])
            next_u_vec = next_p_vec / np.linalg.norm(next_p_vec, 1) # get Norm-1 distance
        else:
            next_u_vec = np.array([0, 0])
            if np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) >= 0.1: # log for debugging
                # FIXME: why A* got trouble?
                #print("WTF")
                #print(f"{agent.state.p_pos - landmark.state.p_pos}")
                self.done_flag = True

        u_index = 0
        if all(np.sign(next_u_vec) == np.array([ 0,  0])): u_index = 0
        if all(np.sign(next_u_vec) == np.array([ 1,  0])): u_index = 1
        if all(np.sign(next_u_vec) == np.array([ 1,  1])): u_index = 2
        if all(np.sign(next_u_vec) == np.array([ 0,  1])): u_index = 3
        if all(np.sign(next_u_vec) == np.array([-1,  1])): u_index = 4
        if all(np.sign(next_u_vec) == np.array([-1,  0])): u_index = 5
        if all(np.sign(next_u_vec) == np.array([-1, -1])): u_index = 6
        if all(np.sign(next_u_vec) == np.array([ 0, -1])): u_index = 7
        if all(np.sign(next_u_vec) == np.array([ 1, -1])): u_index = 8

        # one-hot action for agent's action space
        expert_action = np.zeros(9)
        expert_action[u_index] = 1
        return expert_action

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
        v = p1 + p1_dir*t
        w = p2 + p2_dir*u
        where v = w
        [p1_dir, -p2_dir][t, u] = [p2 - p1]
        [t, u] = inv([p1_dir, -p2_dir]).dot([p2 - p1])
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

    def other_gridmap(self, agent, other, grid_size=9, grid_w=0.075, grid_h=0.075):
        """
        Draw the occupied grid map about others' position
        params:
            grid_size: Define grid that grid is (grid * grid) array
            grid_w: Define each grid's width
            grid_h: Define each grid's height
        vx = other_x - agent_x
        vy = other_y - agent_y
        grid_pos_x = vx / grid_w
        grid_pos_y = vy / grid_h
        IF grid_pos in grid's size
        THEN marked 1 to represent there is a agent occupied
        Return:
            [gridmap.flatten()]
        """

        grid = np.zeros((grid_size, grid_size))

        for op in other:
            v = op - agent.state.p_pos
            vx = v[0]
            vy = v[1]
            # project to grid's coordination
            x =  int((vx + np.sign(vx)*grid_w/2) / grid_w) + (grid_size//2)
            y = (int((vy + np.sign(vy)*grid_w/2) / grid_h) - (grid_size//2))*-1
            if x in range(grid_size) and y in range(grid_size):
                grid[y][x] = 1

        return [grid.flatten()]

    def other_plan_gridmap(self, agent, other, other_goal, grid_size=9, grid_w=0.075, grid_h=0.075):
        """
        Draw the occupied grid map about others' plan
        IFF other in grid's range
        THEN draw the path of other on the grid map
        Return:
            [gridmap.flatten()]
        """

        scale = 0.01
        grid = np.zeros((grid_size, grid_size))

        for op, og in zip(other, other_goal):
            v = op - agent.state.p_pos
            vx = v[0]
            vy = v[1]
            # project to grid's coordination
            x =  int((vx + np.sign(vx)*grid_w/2) / grid_w) + (grid_size//2)
            y = (int((vy + np.sign(vy)*grid_w/2) / grid_h) - (grid_size//2))*-1
            if x in range(grid_size) and y in range(grid_size): # IFF other in range
                p_start = tuple((op / scale).astype(int))
                p_goal = tuple((og / scale).astype(int))
                route = a_star.planning(p_start[0], p_start[1], p_goal[0], p_goal[1])
                for p in reversed(route):
                    px = int((p[0] + np.sign(p[0])*grid_w/2) / grid_w) + (grid_size//2)
                    py = (int((p[1] + np.sign(p[1])*grid_w/2) / grid_h) - (grid_size//2))*-1
                    if px in range(grid_size) and py in range(grid_size):
                        grid[py][px] = 1
                    else:
                        break
        return [grid.flatten()]
