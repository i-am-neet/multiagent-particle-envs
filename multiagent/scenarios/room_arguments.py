class RoomArgs(object):
    def __init__(self, room_num=0):
        self.room_num = room_num
        self.max_room_num = 6 # rooms' amount
        self.wall_num = 0 # walls' maximum amount
        ## TODO
        # Use wall's center points (px, py), angle THETA, length L, and width W
        # Or specify polygon points
        # to repersent wall
        # self.wall_info = {'wall_centers':[], 'wall_angs':[], 'wall_lengths':[], 'wall_widths':[],
        #                   'wall_points':[]}

        self.wall_centers = 0
        self.wall_shapes = 0
        self.ox = 0
        self.oy = 0
        assert room_num < self.max_room_num
        self.get_room(room_num)

        # A* configs
        self.grid_size = 2.0  # [m]
        self.robot_radius = 10.0  # [m]

    def get_room(self, room_id=0):
        self.room_num = room_id
        wall_centers, wall_shapes, ox, oy = [], [], [], []
        if room_id == 0: # empty
            # particle env
            T = 0.03 # thickness
            wall_centers = [[-1, 0], [0, 1], [1, 0], [0, -1]]
            wall_shapes  = [[T, 2], [2, T], [T, 2], [2, T]]
            self.wall_num = len(wall_centers)
            # set obstacle positions for A*
            for i in range(-100, 100):
                ox.append(i)
                oy.append(-100)
            for i in range(-100, 100):
                ox.append(-100)
                oy.append(i)
            for i in range(-100, 100):
                ox.append(i)
                oy.append(100)
            for i in range(-100, 100):
                ox.append(100)
                oy.append(i)
        if room_id == 1: # easy-2-walls
            # particle env
            T = 0.03
            wall_centers = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0.7], [0, -0.7]]
            wall_shapes  = [[T, 2], [2, T], [T, 2], [2, T], [T, 0.7], [T, 0.7]]
            self.wall_num = len(wall_centers)
            # set obstacle positions for A*
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
        if room_id == 2: # easy-4-walls
            # particle env
            T = 0.03
            wall_centers = [[-1, 0], [0, 1], [1, 0], [0, -1], [-0.7, 0], [0, 0.7], [0.7, 0], [0, -0.7]]
            wall_shapes  = [[T, 2], [2, T], [T, 2], [2, T], [0.7, T], [T, 0.7], [0.7, T], [T, 0.7]]
            self.wall_num = len(wall_centers)
            # set obstacle positions for A*
            for i in range(-100, 100):
                ox.append(i)
                oy.append(-100)
            for i in range(-100, 100):
                ox.append(-100)
                oy.append(i)
            for i in range(-100, 100):
                ox.append(i)
                oy.append(100)
            for i in range(-100, 100):
                ox.append(100)
                oy.append(i)
            for i in range(-100, -35):
                ox.append(i)
                oy.append(0)
            for i in range(-100, -35):
                ox.append(0)
                oy.append(i)
            for i in range(35, 100):
                ox.append(i)
                oy.append(0)
            for i in range(35, 100):
                ox.append(0)
                oy.append(i)

        if room_id == 3: # center-bar
            # particle env
            T = 0.03
            wall_centers = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]]
            wall_shapes  = [[T, 2], [2, T], [T, 2], [2, T], [1.2, T]]
            self.wall_num = len(wall_centers)
            # set obstacle positions for A*
            for i in range(-100, 100):
                ox.append(i)
                oy.append(-100)
            for i in range(-100, 100):
                ox.append(-100)
                oy.append(i)
            for i in range(-100, 100):
                ox.append(i)
                oy.append(100)
            for i in range(-100, 100):
                ox.append(100)
                oy.append(i)
            for i in range(-60, 60):
                ox.append(i)
                oy.append(0)

        if room_id == 4: # center-cross
            # particle env
            T = 0.03
            wall_centers = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0], [0, 0]]
            wall_shapes  = [[T, 2], [2, T], [T, 2], [2, T], [1.2, T], [T, 1.2]]
            self.wall_num = len(wall_centers)
            # set obstacle positions for A*
            for i in range(-100, 100):
                ox.append(i)
                oy.append(-100)
            for i in range(-100, 100):
                ox.append(-100)
                oy.append(i)
            for i in range(-100, 100):
                ox.append(i)
                oy.append(100)
            for i in range(-100, 100):
                ox.append(100)
                oy.append(i)
            for i in range(-60, 60):
                ox.append(i)
                oy.append(0)
            for i in range(-60, 60):
                ox.append(0)
                oy.append(i)

        if room_id == 5: # home-area
            # particle env
            T = 0.03
            wall_centers = [[-1, 0], [0, 1], [1, 0], [0, -1], [-0.5, 0], [0, 0.7], [0.3, 0.35], [0.65, 0], [0, -0.7]]
            wall_shapes  = [[T, 2], [2, T], [T, 2], [2, T], [1, T], [T, 0.7], [T, 0.7], [0.7, T], [T, 0.7]]
            self.wall_num = len(wall_centers)
            # set obstacle positions for A*
            for i in range(-100, 100):
                ox.append(i)
                oy.append(-100)
            for i in range(-100, 100):
                ox.append(-100)
                oy.append(i)
            for i in range(-100, 100):
                ox.append(i)
                oy.append(100)
            for i in range(-100, 100):
                ox.append(100)
                oy.append(i)
            for i in range(-100, 0):
                ox.append(i)
                oy.append(0)
            for i in range(35, 100):
                ox.append(0)
                oy.append(i)
            for i in range(0, 70):
                ox.append(30)
                oy.append(i)
            for i in range(30, 100):
                ox.append(i)
                oy.append(0)
            for i in range(-100, -35):
                ox.append(0)
                oy.append(i)

        self.wall_centers = wall_centers
        self.wall_shapes = wall_shapes
        self.ox = ox
        self.oy = oy
