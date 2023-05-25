import json

import numpy as np
import matplotlib.pyplot as plt
import mplsoccer as mpl
import networkx as nx

from tqdm import tqdm
from datetime import datetime
from typing import List
from typing import Dict
from typing import Tuple
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree
from scipy.spatial import voronoi_plot_2d


class GameLoader:
    """
    This class receives the path for the game data files and the name of a particular game
    and loads the relevant information about the game.
    """    
    def __init__(self, base_path, game_name):
        with open(f'{base_path}/{game_name}-tracks.json', 'r') as handle:
            self.game_tracks = json.load(handle)
        
        with open(f'{base_path}/{game_name}-info_live.json', 'r') as handle:
            self.game_info = json.load(handle)
            
        with open(f'{base_path}/{game_name}-events.json', 'r') as handle:
            self.game_events = json.load(handle)
        
        self.__ball_jersey_num_cleaner()
        self.__interpolate_ball_coordinates()
        self.__interpolate_ball_possessor()
    
    def get_team_players(self, home :bool=True) -> List[Dict]:
        """
        This function returns the name of the players of a team with their jersey numbers.
        
        :param home (bool) - Default True: returns the players of the Home team if True,
                                        returns the players of the Away team otherwise
        """
        # specify the team for which we want to extract the team members
        key = 'team_home_players' if home else 'team_away_players'
        
        return self.game_info[key]
    
    def get_pitch_size(self) -> List:
        """
        This function extracts the pitch size using the info file.
        """
            
        return self.game_info['calibration']['pitch_size']
    
    def get_game_distinct_events(self) -> List:
        """
        This function lists the distinct events happening in the game.
        """
        event_types = [event['type'] for event in self.game_events]
        event_types = np.unique(event_types).tolist()
        
        return event_types
    
    @staticmethod
    def time_interpreter(utc_time: int):
        """
        This function converts the utc timestamps available in the data to 
        human readable datetime in the following format: (yyyy, MM, dd, hh, mm, ss, ms)
        
        :param utc_time (int): the timestamp in integer
        """
        # we must make sure that utc_time is an integer
        assert isinstance(utc_time, int)
        
        # if utc_time has a length greater than 10 (which is the case for this dataset)
        # it also consists of milliseconds
        if len(str(utc_time)) > 10:
            utc_time = float(str(utc_time)[:10] + '.' + str(utc_time)[10:])
        
        return datetime.fromtimestamp(utc_time)

    def __ball_jersey_num_cleaner(self):
        """
        At some instances of the game, the jersey number of the ball possessor is unknown
        but the possessor's team and track id is known. At such instances, this function 
        refills the jersey number of the possessor using the given information.
        """
        new_tracks_list = []
        counter = 0
        for track in self.game_tracks:
            # find a track for which the jersey number of possessor is unknown
            if track['ball']['jersey_number'] is None or track['ball']['jersey_number'] == -1:
                # find the jersey number of the possessor's team and track id is known
                if track['ball']['team'] is not None and track['ball']['track_id'] is not None:
                    new_track = track
                    team_info = track[track['ball']['team']]
                    
                    for item in team_info:
                        if item['track_id'] == track['ball']['track_id']:
                            new_track['ball']['jersey_number'] = item['jersey_number']
                            counter += 1
                            break
                    new_tracks_list.append(new_track)
                else:
                    new_tracks_list.append(track)
            else:
                new_tracks_list.append(track)
        
        print(f'{counter} records cleaned')
        self.game_tracks = new_tracks_list

    def __interpolate_ball_coordinates(self):
        """
        This functions interpolates the ball position for every instance of the game at which
        the ball coordinates is unknown
        """
        # keep track of the index of the element in the ball_trajectory list
        index = 0
        
        # iterate through the sequence of ball coordinates
        while index < len(self.game_tracks):
            # we need to find the number of consecutive frames which have NaN values
            num_consecutive_nones = 0
            # check if this element of trajectory is None
            if self.game_tracks[index]['ball']['position'] is None:
                # fix the starting index (the last index which is not NaN)
                start_index = index - 1
                
                # swing the index until we get to a non-NaN value and increase 
                # the number of consecutive NaN frames
                end_index = index
                while self.game_tracks[end_index]['ball']['position'] is None:
                    num_consecutive_nones += 1
                    end_index += 1
                    
                # define the steps for linear interpolation
                steps = np.linspace(0, 1., num_consecutive_nones)
                
                # fill the NaN values by interpolating the line connecting the two non-NaN values
                init = np.array(self.game_tracks[start_index]['ball']['position'], dtype=np.float16)
                finit = np.array(self.game_tracks[end_index]['ball']['position'], dtype=np.float16)
                
                for j, step in enumerate(steps):
                    value = np.round((1 - step) * init + step * finit, 2)
                    self.game_tracks[start_index + j + 1]['ball']['position'] = value.tolist()
                
                index = end_index
            else:
                index += 1
    
    def __interpolate_ball_possessor(self):
        """
        This function finds the ball possessor for the instances at which the possessor is unknwon.
        Such instances usually happen in continuous sequences. For a found sequence, we look at the
        events information and see whether there is a pass event happend before the sequence. If so,
        we interpolate the ball possessor with the information of the pass receiver. Otherwise, we 
        take the simple approach and fill the information of the ball possessor using the concept of
        Voronoi cells; i.e., the ball is assigned to the closest player.
        """
        def get_list_discontinuities(lst: List):
            """
            This function receives a list of numbers and extract the indices of the list
            at which a jump of greater than one happens between two consecutive number.
            
            We use this function to find the beginning moment of losing the possessor's
            information and the beginning moment of we regain the information of the 
            possessor.
            
            :param lst (List): the list in which we want to find discontinuity points
            """
            discontinuities = [lst[0]]
            
            for i in range(1, len(lst) - 1):
                if lst[i + 1] - lst[i] > 1:
                    discontinuities.append(lst[i])
                    discontinuities.append(lst[i + 1])
        
            return discontinuities
        
        def assign_ball_based_on_distance(track: Dict):
            """
            This function receives an instance of the tracking information and assigns the ball
            to the player who is the closest to the ball
            """
            # finding the ball coordinates
            ball_coordinates = np.array(track['ball']['position'][:2])
            
            # getting the players info in this particular instance
            home_team_info = track['home_team']
            away_team_info = track['away_team']
            
            # find the closest home team player to the ball
            home_closest = home_team_info[0]
            home_distance = np.sqrt(((np.array(home_closest['position']) - ball_coordinates) ** 2).sum())
            
            for player in home_team_info:
                player_distance = np.sqrt(((np.array(player['position']) - ball_coordinates) ** 2).sum())
                
                if player_distance < home_distance:
                    home_closest = player
                    home_distance = player_distance
                    
            # find the closest away team player to the ball
            away_closest = away_team_info[0]
            away_distance = np.sqrt(((np.array(away_closest['position']) - ball_coordinates) ** 2).sum())
            
            for player in away_team_info:
                player_distance = np.sqrt(((np.array(player['position']) - ball_coordinates) ** 2).sum())
                
                if player_distance < away_distance:
                    away_closest = player
                    away_distance = player_distance
            
            # find the closest player to the ball and fill in the possessor information
            if home_distance < away_distance:
                track['ball']['jersey_number'] = home_closest['jersey_number']
                track['ball']['team'] = 'home_team'
                track['ball']['track_id'] = home_closest['track_id']
            else:
                track['ball']['jersey_number'] = away_closest['jersey_number']
                track['ball']['team'] = 'away_team'
                track['ball']['track_id'] = away_closest['track_id']
            
            return track

        
        # find the instances for which the ball possessor is unknown
        missing_instances = []
        for i, t in enumerate(self.game_tracks):
            if t['ball']['jersey_number'] is None or t['ball']['jersey_number'] == -1:
                missing_instances.append(i)

        print(f'[*] {len(missing_instances)} instances from {len(self.game_tracks)} instances with missing ball possessor information')
        print(f'[*] Interpolating the ball positions and possessors...')
        
        # find the break point of the sequences at which the possessor is missing
        break_points = get_list_discontinuities(missing_instances)
        
        # fill in the information of the possessor
        iters = tqdm(range(0, len(break_points), 2))
        for i in iters:
            start = break_points[i] - 1
            if start == -1:
                start = 0
                self.game_tracks[start] = assign_ball_based_on_distance(self.game_tracks[start])
            
            if i == len(break_points) - 1:
                end = missing_instances[-1] + 1
            else:
                end = break_points[i + 1] + 1
                
            found = False
            for e in self.game_events:
                if e['utc_time'] == self.game_tracks[start]['utc_time'] and e['type'] == 'pass':
                    found = True
                    for i in range(start + 1, end):
                        self.game_tracks[i]['ball']['jersey_number'] = self.game_tracks[end]['ball']['jersey_number']
                        self.game_tracks[i]['ball']['team'] = self.game_tracks[end]['ball']['team']
                        self.game_tracks[i]['ball']['track_id'] = self.game_tracks[end]['ball']['track_id']
                    break
            
            if not found:
                for i in range(start + 1, end):
                    self.game_tracks[i] = assign_ball_based_on_distance(self.game_tracks[i])
        
        missing_instances = []
        for i, t in enumerate(self.game_tracks):
            if t['ball']['jersey_number'] is None or t['ball']['jersey_number'] == -1:
                missing_instances.append(i)
        
        print(f'[*] {len(missing_instances)} instances from {len(self.game_tracks)} instances with missing ball possessor information')
        if len(missing_instances) > 0:
            print(missing_instances)
        
        
class PlayerTracker:
    """
    This class tracks the location of a target player along with the ball coordinates
    for every instance that the player happens to exist.
    """
    def __init__(self, game: GameLoader, jersey_number: int, home: bool=True) -> None:
        """
        The initializer/constructor gets an instance of the GameLoader which contains 
        the players and all the tracking information, the jersey number of the target
        player and whether the player belongs to the home team.
        NOTE: if you do not know the jersey number of the player, you can invoke the 
              get_team_players() method of the game instance to see the players names
              with their corresponding jersey numbers
              
        :param game (GameLoader): an instance of the game.
        :param jersey_number (int): the jersey number of the target player.
        :param home (bool) - Default True: whether the player belongs to the home team
        """
        
        self.game = game
        self.jersey_number = jersey_number
        self.home = home
        self.player_trajectory = None
        self.ball_trajectory = None
        self.team_possession_trajectory = None
        self.track_id = None
        self.missing_instances = []
        self.visible_polygons = []
    
    
    def track(self) -> None:
        """
        This function generates the trajectory of the movement of the player along 
        the game using the tracking data.
        
        This function updates two attributes self.player_trajectory and self.ball_trajectory
        """
        # create an empty list to record the coordinates of the player
        player_trajectory = []
        
        # if we also want to see the location of the ball at each frame
        ball_trajectory = []
        
        team_possession_trajectory = []
        
        # specify the team in which the player plays
        key = 'home_team' if self.home else 'away_team'
        # we should also find the track id of this player for future uses
        track_id = None
        
        print('[*] Tracking the player and the ball...')
        for ind, track in enumerate(self.game.game_tracks):
            # get the information of all players at an instance
            players = track[key]
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['jersey_number'] == self.jersey_number, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                # find the player's coordinates at this frame
                coords = target_player[0]['position']
                # find the player's tracking id
                track_id = target_player[0]['track_id']
                # fill in the player's trajectory list
                player_trajectory.append(coords)

                # find the position of the ball
                ball_trajectory.append(track['ball']['position'])
                
                # determining which team possesses the ball in this instance
                team_possession_trajectory.append(track['ball']['team'])
            else:
                self.missing_instances.append(ind)
        
        self.track_id = track_id
        self.player_trajectory = player_trajectory
        self.ball_trajectory = ball_trajectory

    def get_player_events(self) -> List:
        """
        This function uses the tracking id of the player and temporally lists
        all the events pertaining to the player associated to the specified tracking
        id.
        """
        target_events = []
        for event in self.game.game_events:
            if 'track_id' in event.keys():
                if event['track_id'] == self.track_id:
                    target_events.append(event)
            elif 'from_track_id' in event.keys():
                if event['from_track_id'] == self.track_id:
                    target_events.append(event)
            else:
                continue
                
        return target_events

    def get_player_gaze(self) -> Tuple[List, List, List, List]:
        """
        This function uses the player's trajectory and the ball trajectory already tracked
        and generates the gaze vectors as the composition of movement vectors and player-to-ball
        vectors.
        """
        # a list to store the x coordinates of the gaze vectors tails
        gaze_start_x = []
        
        # a list to store the y cooordinates of the gaze vectors tails
        gaze_start_y = []
        
        # a list to store the x coordinates of the gaze vectors heads
        gaze_end_x = []
        
        # a list to store the y coordinates of the gaze vectors heads
        gaze_end_y = []
        
        # iterate through all the position coordinates of the player and the ball
        for i in range(len(self.player_trajectory) - 1):
            # find the length of the vector from the player to the ball (used to normalize the vectors)
            length = np.sqrt((self.ball_trajectory[i][0] - self.player_trajectory[i][0]) ** 2 + (self.ball_trajectory[i][1] - self.player_trajectory[i][1]) ** 2)
            # find the length of the movement vector of the player (used to normaluze the vectors)
            p_length = np.sqrt((self.player_trajectory[i + 1][0] - self.player_trajectory[i][0]) ** 2 + (self.player_trajectory[i + 1][1] - self.player_trajectory[i][1]) ** 2)
            
            # find the end-location of the normalized movement vectors and player-to-ball vectors
            p_end_x = self.player_trajectory[i][0] + (self.player_trajectory[i + 1][0] - self.player_trajectory[i][0]) / p_length if p_length != 0 else self.player_trajectory[i][0]
            p_end_y = self.player_trajectory[i][1] + (self.player_trajectory[i + 1][1] - self.player_trajectory[i][1]) / p_length if p_length != 0 else self.player_trajectory[i][1]
            p_b_end_x = self.player_trajectory[i][0] + (self.ball_trajectory[i][0] - self.player_trajectory[i][0]) / length
            p_b_end_y = self.player_trajectory[i][1] + (self.ball_trajectory[i][1] - self.player_trajectory[i][1]) / length
            
            # constructing the gaze vectors
            gaze_start_x.append(self.player_trajectory[i][0])
            gaze_start_y.append(self.player_trajectory[i][1])
            gaze_end_x.append(-self.player_trajectory[i][0] + (p_end_x + p_b_end_x))
            gaze_end_y.append(-self.player_trajectory[i][1] + (p_end_y + p_b_end_y))
        
        return gaze_start_x, gaze_start_y, gaze_end_x, gaze_end_y
    
    def get_visible_polygon(self) -> List:
        """
        This function uses the start and end coordinates of the gaze vectors and computes
        the vertices of the visible polygon at each frame.
        """
        # find the width and height of the pitch
        # NOTE: the (0, 0) coordinate is placed at the centeral point of the pitch
        size_x, size_y = self.game.get_pitch_size()
        
        # define the x, y values of the left, right, upper, and lower boundaries
        left_x, right_x, upper_y, lower_y = -size_x / 2, size_x / 2, size_y / 2, -size_y / 2
        
        print('[*] Computing the gaze vectors of the player...')
        start_x_list, start_y_list, end_x_list, end_y_list = self.get_player_gaze()
        
        print('[*] Computing the visible polygons of the player...')
        for index in range(len(start_x_list)):
            start_x = start_x_list[index]
            start_y = start_y_list[index]
            end_x = end_x_list[index]
            end_y = end_y_list[index]
            # if the gaze vecotr is not a vertical vector
            if end_x - start_x != 0:
                gaze_slope = (end_y - start_y) / (end_x - start_x)
            # if the gaze vector is vertical
            else:
                gaze_slope = None
            
            # if the gaze vector is horizontal 
            if gaze_slope == 0:
                # find the intersection of the gaze line with the boundary
                point1 = (start_x, upper_y)
                point2 = (start_x, lower_y)
                
                # check if the player is moving to the left or right
                left_facing = True if end_x < start_x else False
                
                # if the player is moving to the left (consider the left polygon)
                if left_facing:
                    # get the two other vertices of the visible polygon
                    point3 = (left_x, upper_y)
                    point4 = (left_x, lower_y)
                else:
                    point3 = (right_x, upper_y)
                    point4 = (right_x, lower_y)
            # if the gaze vector is vertical
            elif gaze_slope is None:
                # find the intersection of the gaze line with the boundary
                point1 = (left_x, start_y)
                point2 = (right_x, start_y)
                
                # check if the player is moving downwards or upwards
                down_facing = True if end_y < start_y else False
                
                # if moving downwards (lower polygon)
                if down_facing:
                    point3 = (left_x, lower_y)
                    point4 = (right_x, lower_y)
                # if moving upwards (upper polygon)
                else:
                    point3 = (left_x, upper_y)
                    point4 = (right_x, upper_y)
            # if the gaze line is an oblique line
            else:
                # find the gaze line
                line_slope = - 1 / gaze_slope
                interception = start_y - line_slope * start_x
                
                # find the intersection of the gaze line with the boundary
                point1 = ((upper_y - interception) / line_slope, upper_y)
                point2 = ((lower_y - interception) / line_slope, lower_y)
                
                # check if the player is moving to the left or right
                left_facing = True if end_x < start_x else False
            
                # if the player is moving to the left (left polygon)
                if left_facing:
                    point3 = (left_x, upper_y)
                    point4 = (left_x, lower_y)
                # if the player is moving to the right (right polygon)
                else:
                    point3 = (right_x, upper_y)
                    point4 = (right_x, lower_y)
                
            self.visible_polygons.append([point1, point2, point3, point4])
    
    def player_movement_arrows(self) -> Tuple[List, List, List, List]:
        """
        This function uses the trajectory of the player and generates the movement
        vectors from the trajectory.
        """
        # a list to store the x coordinate of the tail of the movement vectors
        start_x = []
        
        # a list to store the y coordinate of the tail of the movement vectors
        start_y = []
        
        # a list to store the x coordinate of the head of the movement vectors
        end_x = []
        
        # a list to store the y coordinates of the head of the movement vectors
        end_y = []
        
        for index in range(len(self.player_trajectory) - 1):
            start_x.append(self.player_trajectory[index][0])
            start_y.append(self.player_trajectory[index][1])
            end_x.append(self.player_trajectory[index + 1][0])
            end_y.append(self.player_trajectory[index + 1][1])
        
        return start_x, start_y, end_x, end_y

    def ball_movement_arrows(self) -> Tuple[List, List, List, List]:
        """
        This function uses the trajectory of the ball and generates the movement
        vectors from the trajectory.
        """
        ball_start_x = []
        ball_start_y = []
        ball_end_x = []
        ball_end_y = []
        
        for index in range(len(self.ball_trajectory) - 1):
            ball_start_x.append(self.ball_trajectory[index][0])
            ball_start_y.append(self.ball_trajectory[index][1])
            ball_end_x.append(self.ball_trajectory[index + 1][0])
            ball_end_y.append(self.ball_trajectory[index + 1][1])
        
        return ball_start_x, ball_start_y, ball_end_x, ball_end_y


class AllTracker:
    """
    This class tracks the ball throughout the game. The difference between this class
    and PlayerTracker class is that Playertracker class tracks the ball only in the 
    frames that the target player is present. However, this class tracks the ball in
    all the frames, and also keeps track of all the players in all frames.
    
    This class is useful when someone wants to plot the general configuration of the pitch
    in a particular snapshot, or determine who possesses the ball at a particular instance
    of the game.
    """
    def __init__(self, game: GameLoader) -> None:
        """
        The initializer/constructor gets an instance of the GameLoader which contains 
        the players and all the tracking information.
        """
        self.game = game
        self.ball_trajectory = None
        self.home_coordinates = None
        self.away_coordinates = None
    
    
    def track(self):
        """
        This function iterates through all the frames, and at each frame it stores the coordinate of 
        the players appearing on the pitch divided into two separate lists home_coordinates and 
        away_coordinates. At each frame, the coordinates of the ball is also recorded and by the end
        of iterations, the missing ball locations get linearly interpolated.
        
        This function updates three attributes self.ball_trajectory, self.home_coordinates, self.away_coordinates
        """
        # ball_trajectory is a list, each of which element is a list specifying the coordinate of the ball
        # e.g., [[0, 0, 0], [1, 0, 0], ...]
        ball_trajectory = []
        
        # home_coordinates is a list, each of which element is again a list which contains the coordinates of
        # the home players; e.g., [[[0, 0], [0, 1], [1, 0], .., [4, 6]], [[0, 1], [1, 2], [3, 5], ..., [8, 12]]]
        home_coordinates = []
        
        # similar to home_teams but for the away players
        away_coordinates = []
        
        print('[*] Tracking the players and the ball...')
        for index in tqdm(range(len(self.game.game_tracks))):
            track = self.game.game_tracks[index]
            # find the position of the ball
            ball_trajectory.append(track['ball']['position'])
            
             # get the information of all players at an instance
            home_players = track['home_team']
            away_players = track['away_team']
            
            home_coordinates.append(home_players)
            away_coordinates.append(away_players)
        
        self.ball_trajectory = ball_trajectory
        self.home_coordinates = home_coordinates
        self.away_coordinates = away_coordinates
    
    def get_ball_possessor_deprecated(self, instance_no: int, show: bool=True, return_voronoi: bool=False) -> Tuple[bool, int] | Tuple[bool, int, Voronoi]:
        """
        This function receives na instance number and extracts the coordinates of the home players, 
        away players, and the ball, and evaluates the voronoi cells of the players and determines 
        the player that is supposed to possess the ball
        
        :param instance_no (int): the number of instance at which we want to know who owns the ball.
        :param show (bool) - Default True: if True, plots the Voronoi cell as well as the players and the ball
        :param return_voronoi (bool) - Default False: if True, returns the Voronoi cell as well.
                                        
        :returns Tuple[bool, int]: returns a tuple; the first element is True if the ball owner belongs to the
                                   home team, and False otherwise. The second element specifies the jersey number
                                   of the ball owner.
                                   If return_voronoi is True, the function also returns the voronoi cell
        """
        # combining the coordinates
        home_coordinates = self.home_coordinates[instance_no]
        away_coordinates = self.away_coordinates[instance_no]
        ball_coordinates = self.ball_trajectory[instance_no]
        
        all_coordinates = []
        all_coordinates.extend(home_coordinates)
        all_coordinates.extend(away_coordinates)
        
        home = False
        
        voronoi_cells = Voronoi(np.array(all_coordinates))
        
        if show:
            voronoi_plot_2d(voronoi_cells, show_points=False, show_vertices=False, line_colors='darkgreen')
            plt.scatter([ball_coordinates[0]], [ball_coordinates[1]], c='orange')
            plt.scatter(np.array(home_coordinates)[:, 0], np.array(home_coordinates)[:, 1], s=35, c='blue')
            plt.scatter(np.array(away_coordinates)[:, 0], np.array(away_coordinates)[:, 1], s=35, c='red')
            plt.show()
        
        voronoi_kd_tree = cKDTree(np.array(all_coordinates))
        distance, player_index = voronoi_kd_tree.query(ball_coordinates[:2])
        
        if player_index <= 10:
            home = True
        
        player_index = player_index % 11
        
        if home:
            jersey_number = self.game.get_team_players(True)[player_index]['jersey_number']
        else:
            jersey_number = self.game.get_team_players(False)[player_index]['jersey_number']
        
        
        if return_voronoi:
            return home, jersey_number, voronoi_cells
        
        return home, jersey_number  

    def get_ball_possessor(self, instance_no):
        jersey_number = self.game.game_tracks[instance_no]['ball']['jersey_number']
        team = self.game.game_tracks[instance_no]['ball']['team']
        
        home = True if team == 'home_team' else False
        
        return home, jersey_number
    
    def get_reachable_teammates(self, instance_no: int, jersey_number: int, home: bool=True) -> int:
        """
        This function receives an instance number, a jersey number, the voronoi cells 
        associate to the instance of the game, and whether the player belongs to the
        home team. The function then uses the Voronoi cells to construct an adjacency
        matrix of the teammates. If there is a path between the target player and any
        of the teammates, then that teammate is reachable.
        
        :param instance_no (int): the number of instance at which we want to count the 
                                  reachable teammates.
        :param jersey_number (int): the jersey number of the target player
        :param voronoi_cells (Voronoi): the voronoi cells of the particular instance 
                                        of the game
        :param home (bool) - Default True: determines whether the player belongs to
                                           home team or away team.
        """
        # combining the coordinates
        home_info = self.home_coordinates[instance_no]
        away_info = self.away_coordinates[instance_no]
        
        all_coordinates = []
        home_players = []
        away_players = []
        
        for i in range(len(home_info)):
            all_coordinates.append(home_info[i]['position'])
            home_players.append(home_info[i]['jersey_number'])
        for i in range(len(away_info)):
            all_coordinates.append(away_info[i]['position'])
            away_players.append(away_info[i]['jersey_number'])
        
        voronoi_cells = Voronoi(np.array(all_coordinates))
        
        # regardless of the player belonging to either teams, find the
        # player's index and the teammates indices in the voronoi cells
        if home:
            player_index = home_players.index(jersey_number)
            teammate_indices = [i for i in range(len(home_players))]
            adjacency_matrix = np.zeros((len(home_players), len(home_players)))
        else:
            player_index = away_players.index(jersey_number) + len(home_players)
            teammate_indices = [i + len(home_players) for i in range(len(away_players))]
            adjacency_matrix = np.zeros((len(away_players), len(away_players)))
        
        # extract the voronoi region each index belongs to
        voronoi_point_regions = voronoi_cells.point_region
        voronoi_regions = voronoi_cells.regions
        
        # construct the adjacency matrix of the teammates
        for i in range(len(teammate_indices)):
            for j in range(len(teammate_indices)):
                if i != j:
                    # if the voronoi cells of two players has at least two similar vertices
                    # it means they are adjacent
                    region_i = set(voronoi_regions[voronoi_point_regions[teammate_indices[i]]])
                    region_j = set(voronoi_regions[voronoi_point_regions[teammate_indices[j]]])
                    
                    if len(region_i.intersection(region_j)) > 1:
                        adjacency_matrix[i, j] = 1
        
        # construct the graph 
        graph = nx.from_numpy_array(adjacency_matrix)
        # find the paths between the player and the teammates
        shortest_paths = nx.single_target_shortest_path(graph, player_index)
        
        return len(shortest_paths) - 1
        

class Visualizers:
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker) -> None:
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        
        if player.player_trajectory is None:
            self.player.track()
        
        if all_tracker.ball_trajectory is None:
            self.all_tracker.track()
        
    def plot_trajectory(self, start_time: int=None, end_time: int=None) -> None:
        """
        This function receives the movement vectors of a player along the game and plots the trajectory.
        
        :param start_time (int) - Default None: the starting frame for the visualization; if None
                                                it starts from the beginning
        :param end_time (int) - Default None: the ending frame for the visualization; if None it
                                              it ends at the last frame
        """
        # extract the pitch size
        pitch_size = self.game.get_pitch_size()
        start_x_coords, start_y_coords, end_x_coords, end_y_coords = self.player.player_movement_arrows()
        ball_start_x, ball_start_y, ball_end_x, ball_end_y = self.player.ball_movement_arrows()
        
        start_time = 0 if start_time is None else start_time
        end_time = len(start_x_coords) if end_time is None else end_time
        
        # define the pitch graphical object
        pitch = mpl.Pitch(pitch_type='skillcorner', 
                          pitch_length=pitch_size[0], 
                          pitch_width=pitch_size[1], 
                          axis=True, 
                          label=True)
        _, ax = pitch.draw(figsize=(9, 6))

        start_x_coords = start_x_coords[start_time: end_time]
        start_y_coords = start_y_coords[start_time: end_time]
        end_x_coords = end_x_coords[start_time: end_time]
        end_y_coords = end_y_coords[start_time: end_time]
        
        ball_start_x = ball_start_x[start_time: end_time]
        ball_start_y = ball_start_y[start_time: end_time]
        ball_end_x = ball_end_x[start_time: end_time]
        ball_end_y = ball_end_y[start_time: end_time]
        
        
        # draw movement vectors as arrows
        pitch.arrows(start_x_coords, start_y_coords, end_x_coords, end_y_coords, 
                     alpha=0.8, 
                     color=(0, 0, 1), 
                     headaxislength=13, 
                     headlength=13, 
                     headwidth=14, 
                     width=2, 
                     ax=ax)
            
        # draw ball movement vectors as arrows
        pitch.arrows(ball_start_x, ball_start_y, ball_end_x, ball_end_y, 
                     alpha=0.4, 
                     color=(1, 0, 0), 
                     headaxislength=3, 
                     headlength=3, 
                     headwidth=4, 
                     width=2, 
                     ax=ax)
        
        ball_x_coords = np.array(self.player.ball_trajectory)[start_time: end_time, 0]
        ball_y_coords = np.array(self.player.ball_trajectory)[start_time: end_time, 1]
            
        pitch.scatter(ball_x_coords, ball_y_coords, ax=ax, facecolor='yellow', s=5, edgecolor='k')
            
        plt.show()

    def plot_presence_heat_map(self) -> None:
        """
        This function uses the trajectory of the player and plots the heat map
        of the player's presence in the pitch.
        """
        pitch_size = self.game.get_pitch_size()
        
        pitch = mpl.Pitch(line_color='white',
                          pitch_type='skillcorner', 
                          pitch_length=pitch_size[0], 
                          pitch_width=pitch_size[1], 
                          axis=True, 
                          label=True)
        
        player_trajectory = np.array(self.player.player_trajectory)
        statistics = pitch.bin_statistic(player_trajectory[:, 0], player_trajectory[:, 1], statistic='count', bins=(25, 25))
        
        fig, ax = pitch.draw(figsize=(9, 6))
        fig.set_facecolor('#22312b')
        
        pcm = pitch.heatmap(statistics, ax=ax, cmap='hot', edgecolors='#22312b')
        fig.colorbar(pcm, ax=ax, shrink=0.6)
        plt.show()
    
    def plot_game_snapshot(self, instance_no: int) -> None:
        """
        This function receives the index of a snapshot of the game and plots the coordinates of the home-team
        players (blue), and away players (red) as well as the ball (yellow) on the pitch.
        
        :param instance_no (int): the index of the snapshot we want to illustrate.
        """
            
        ball_trajectory = self.all_tracker.ball_trajectory
        home_coordinates = self.all_tracker.home_coordinates
        away_coordinates = self.all_tracker.away_coordinates
            
        # extract the pitch size
        pitch_size = self.game.get_pitch_size()
        
        # define the pitch graphical object
        pitch = mpl.Pitch(pitch_type='skillcorner', 
                          pitch_length=pitch_size[0], 
                          pitch_width=pitch_size[1], 
                          axis=True, 
                          label=True)
        _, ax = pitch.draw(figsize=(9, 6))
                
        pitch.scatter(np.array(home_coordinates[instance_no])[:, 0], np.array(home_coordinates[instance_no])[:, 1], ax=ax, facecolor='blue', s=30, edgecolor='k')
        pitch.scatter(np.array(away_coordinates[instance_no])[:, 0], np.array(away_coordinates[instance_no])[:, 1], ax=ax, facecolor='red', s=30, edgecolor='k')
        pitch.scatter([ball_trajectory[instance_no][0]], [ball_trajectory[instance_no][1]], ax=ax, facecolor='yellow', s=20, edgecolor='k')
            
        plt.show()
    
    def plot_voronoi_cell(self, instance_no: int) -> None:
        """
        This function receives the number of a snapshot of the game and plots the Voronoi cell
        of the players in that snapshot.
        
        :param instance_no (int): an integer specifying the index of the snapshot of the game
        """ 
        ball_trajectory = self.all_tracker.ball_trajectory
        home_coordinates = self.all_tracker.home_coordinates
        away_coordinates = self.all_tracker.away_coordinates
            
        ball_coordinates = ball_trajectory[instance_no]
        all_coordinates = home_coordinates[instance_no].copy()
        all_coordinates.extend(away_coordinates[instance_no])
        
        voronoi_cells = Voronoi(np.array(all_coordinates))

        voronoi_plot_2d(voronoi_cells, show_points=False, show_vertices=False, line_colors='darkgreen')
        plt.scatter([ball_coordinates[0]], [ball_coordinates[1]], c='orange')
        plt.scatter(np.array(home_coordinates[instance_no])[:, 0], np.array(home_coordinates[instance_no])[:, 1], s=35, c='blue')
        plt.scatter(np.array(away_coordinates[instance_no])[:, 0], np.array(away_coordinates[instance_no])[:, 1], s=35, c='red')
        plt.show()
        
        plt.show()
    

def interpolate_ball_trajectory(ball_trajectory: List):
    """
    This function receives the ball trajectory and fills in the missing data
    with linear interpolation.
    
    :param ball_trajectory (List): a list of ball coordinates which contains
    """
    # keep track of the index of the element in the ball_trajectory list
    index = 0
    
    # iterate through the sequence of ball coordinates
    while index < len(ball_trajectory):
        # we need to find the number of consecutive frames which have NaN values
        num_consecutive_nones = 0
        # check if this element of trajectory is None
        if ball_trajectory[index] is None:
            # fix the starting index (the last index which is not NaN)
            start_index = index - 1
            
            # swing the index until we get to a non-NaN value and increase 
            # the number of consecutive NaN frames
            end_index = index
            while ball_trajectory[end_index] is None:
                num_consecutive_nones += 1
                end_index += 1
                
            # define the steps for linear interpolation
            steps = np.linspace(0, 1., num_consecutive_nones)
            
            # fill the NaN values by interpolating the line connecting the two non-NaN values
            for j, step in enumerate(steps):
                init = np.array(ball_trajectory[start_index], dtype=np.float16)
                finit = np.array(ball_trajectory[end_index], dtype=np.float16)
                value = np.round((1 - step) * init + step * finit, 2)
                ball_trajectory[start_index + j + 1] = value.tolist()
            
            index = end_index
        else:
            index += 1
    
    return ball_trajectory



if __name__ == '__main__':
    game = GameLoader('all_Games', 'AIK__BK Häcken.1')
    player = PlayerTracker(game, 4, True)
    all_tracker = AllTracker(game)
    
    vis = Visualizers(game, player, all_tracker)
    
    _, _, v = all_tracker.get_ball_possessor(150, False, True)
    vis.plot_voronoi_cell(150)
    all_tracker.get_reachable_teammates(150, 4, v, True)
    vis.plot_trajectory(start_time=None, end_time=200)
    vis.plot_presence_heat_map()
    
    for i in range(1093, 1124):
        vis.plot_game_snapshot(i)