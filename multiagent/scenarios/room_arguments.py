class RoomArgs(object):
    def __init__(self, room_num=1, wall_num=9):
        self.room_num = room_num # rooms' amount
        self.wall_num = wall_num # walls' maximum amount
        ## TODO
        # Use wall's center points (px, py), angle THETA, length L, and width W
        # Or specify polygon points
        # to repersent wall
        # self.wall_info = {'wall_centers':[], 'wall_angs':[], 'wall_lengths':[], 'wall_widths':[],
        #                   'wall_points':[]}
        self.wall_info = {'wall_centers':[], 'wall_shapes':[]}

        self.wall_info['wall_centers'], self.wall_info['wall_shapes'] = self.get_room(0)

    def get_room(self, room_id=0):
        if room_id == 0: # home-area
            T = 0.03 # thickness
            wall_centers = [[-1, 0], [0, 1], [1, 0], [0, -1], [-0.5, 0], [0, 0.7], [0.3, 0.35], [0.65, 0], [0, -0.7]]
            wall_shapes  = [[T, 2], [2, T], [T, 2], [2, T], [1, T], [T, 0.7], [T, 0.7], [0.7, T], [T, 0.7]]
            self.wall_num = len(wall_centers)
        if room_id == 1: # empty
            wall_centers = []
            wall_shapes  = []
            self.wall_num = len(wall_centers)

        return wall_centers, wall_shapes
