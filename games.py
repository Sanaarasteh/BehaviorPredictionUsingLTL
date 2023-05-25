import json
import pprint

import numpy as np
import mplsoccer as mpl
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple
from typing import Dict
from datetime import datetime
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree
from scipy.spatial import voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# global variable specifying the name of the game we are analyzing
GAME_NAME = 'AIK__BK Häcken.1'

# done
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

# done
def get_team_players(home :bool=True) -> List[Dict]:
    """
    This function returns the name of the players of a team with their jersey numbers.
    
    :param home (bool) - Default True: returns the players of the Home team if True,
                                       returns the players of the Away team otherwise
    """
    # open the info_live file which contains the pertaining information
    with open(f'all_Games/{GAME_NAME}-info_live.json', 'r') as handle:
        game_info = json.load(handle)
    
    # specify the team for which we want to extract the team members
    key = 'team_home_players' if home else 'team_away_players'
    
    return game_info[key]

# done
def get_pitch_size() -> List:
    """
    This function extracts the pitch size using the info file.
    """
    # open the info_live file which contains information about the pitch
    with open(f'all_Games/{GAME_NAME}-info_live.json', 'r') as handle:
        game_info = json.load(handle)
        
    return game_info['calibration']['pitch_size']

# done
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
                
# done
def track_player_coords(jersey_number: int, home: bool=True, track_ball: bool=True) -> List:
    """
    This function receives a jersey number and the team the player plays in and
    generates the trajectory of the movement of the player along the game using the
    tracking data. This function can also track the ball simultaneously.
    
    :param jersey_number (int): the jersey number of the player.
    :param home (bool) - Default True: Home team if True, Away otherwise.
    :param track_ball (bool) - Default True: If True, tracks the ball as well.
    """
    # load the tracking data
    with open(f'all_Games/{GAME_NAME}-tracks.json', 'r') as handle:
        game_tracks = json.load(handle)
    
    # create an empty list to record the coordinates of the player
    player_trajectory = []
    
    # if we also want to see the location of the ball at each frame
    if track_ball:
        ball_trajectory = []
    
    # specify the team in which the player plays
    key = 'home_team' if home else 'away_team'
    # we should also find the track id of this player for future uses
    track_id = None
    
    print('[*] Tracking the player and the ball...')
    for track in game_tracks:
        # get the information of all players at an instance
        players = track[key]
        # check whether the target player has any record in this frame
        target_player = list(filter(lambda x: x['jersey_number'] == jersey_number, players))
        
        # if the player's record exists in this frame...
        if len(target_player) > 0:
            # find the player's coordinates at this frame
            coords = target_player[0]['position']
            # find the player's tracking id
            track_id = target_player[0]['track_id']
            # fill in the player's trajectory list
            player_trajectory.append(coords)
            
            # if we are also supposed to track the ball
            if track_ball:
                # find the position of the ball
                if track['ball']['position'] is not None:
                    ball_trajectory.append(track['ball']['position'])
                # add None if the position of the ball is missing
                else:
                    ball_trajectory.append(None)
                
    if track_ball:
        print('[*] Filling the missing ball coordinates...')
        ball_trajectory = interpolate_ball_trajectory(ball_trajectory)
        return player_trajectory, track_id, ball_trajectory
    
    return player_trajectory, track_id

# done
def movement_arrows_from_trajectory(trajectory: List) -> Tuple[List, List, List, List]:
    """
    This function receives the trajectory of a single player and generates the movement
    vectors from the trajectory.
    
    :param trajectory (List): a list of coordinates of the player along the game
    """
    # a list to store the x coordinate of the tail of the movement vectors
    start_x = []
    
    # a list to store the y coordinate of the tail of the movement vectors
    start_y = []
    
    # a list to store the x coordinate of the head of the movement vectors
    end_x = []
    
    # a list to store the y coordinates of the head of the movement vectors
    end_y = []
    
    for index in range(len(trajectory) - 1):
        start_x.append(trajectory[index][0])
        start_y.append(trajectory[index][1])
        end_x.append(trajectory[index + 1][0])
        end_y.append(trajectory[index + 1][1])
    
    return start_x, start_y, end_x, end_y

# done
def get_player_gaze(trajectory: List, ball_trajectory: List) -> Tuple[List, List, List, List]:
    """
    This function receives the trajectory of the player along with the trajectory of the ball
    and generates the gaze vectors as the composition of movement vectors and player-to-ball
    vectors.
    
    :param trajectory (List): a list containing the coordinates of the player.
    :param ball_trajectory (List): a list containing the coordinates of the ball.
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
    for i in range(len(trajectory) - 1):
        # find the length of the vector from the player to the ball (used to normalize the vectors)
        length = np.sqrt((ball_trajectory[i][0] - trajectory[i][0]) ** 2 + (ball_trajectory[i][1] - trajectory[i][1]) ** 2)
        # find the length of the movement vector of the player (used to normaluze the vectors)
        p_length = np.sqrt((trajectory[i + 1][0] - trajectory[i][0]) ** 2 + (trajectory[i + 1][1] - trajectory[i][1]) ** 2)
        
        # find the end-location of the normalized movement vectors and player-to-ball vectors
        p_end_x = trajectory[i][0] + (trajectory[i + 1][0] - trajectory[i][0]) / p_length if p_length != 0 else trajectory[i][0]
        p_end_y = trajectory[i][1] + (trajectory[i + 1][1] - trajectory[i][1]) / p_length if p_length != 0 else trajectory[i][1]
        p_b_end_x = trajectory[i][0] + (ball_trajectory[i][0] - trajectory[i][0]) / length
        p_b_end_y = trajectory[i][1] + (ball_trajectory[i][1] - trajectory[i][1]) / length
        
        # constructing the gaze vectors
        gaze_start_x.append(trajectory[i][0])
        gaze_start_y.append(trajectory[i][1])
        gaze_end_x.append(-trajectory[i][0] + (p_end_x + p_b_end_x))
        gaze_end_y.append(-trajectory[i][1] + (p_end_y + p_b_end_y))
    
    return gaze_start_x, gaze_start_y, gaze_end_x, gaze_end_y

# done
def get_visible_polygon(start_x: float, start_y: float, end_x: float, end_y: float) -> List:
    """
    This function receives the start and end coordinates of a single gaze vector and computes
    the vertices of the visible polygon.
    
    :param start_x (float): the x coordinate of the starting point of the gaze vector
    :param start_y (float): the y coordinate of the starting point of the gaze vector
    :param end_x (float): the x coordinate of the ending point of the gaze vector
    :param end_y (float): the y coordinate of the ending point of the gaze vector
    """
    # find the width and height of the pitch
    # NOTE: the (0, 0) coordinate is placed at the centeral point of the pitch
    size_x, size_y = get_pitch_size()
    
    # define the x, y values of the left, right, upper, and lower boundaries
    left_x, right_x, upper_y, lower_y = -size_x / 2, size_x / 2, size_y / 2, -size_y / 2
    
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
        
    return [point1, point2, point3, point4]

# done
def get_player_events(track_id: int) -> List:
    """
    This function receives the tracking id of a player and temporally lists
    all the events pertaining to the player associated to the specified tracking
    id.
    
    :param track_id (int): the tracking id of the player we want to track.
    """
    with open(f'all_Games/{GAME_NAME}-events.json', 'r') as handle:
        game_events = json.load(handle)
    target_events = []
    for event in game_events:
        if 'track_id' in event.keys():
            if event['track_id'] == track_id:
                target_events.append(event)
        elif 'from_track_id' in event.keys():
            if event['from_track_id'] == track_id:
                target_events.append(event)
        else:
            continue
            
    return target_events

# done
def get_game_distinct_events() -> List:
    """
    This function lists the distinct events happening in the game.
    """
    with open(f'all_Games/{GAME_NAME}-events.json', 'r') as handle:
        game_events = json.load(handle)
    
    event_types = [event['type'] for event in game_events]
    event_types = np.unique(event_types).tolist()
    
    return event_types


def player_ball_distance_history(player_trajectory: List, ball_trajectory: List, show: bool=True) -> List:
    """
    This function receives a player's trajectory along with the recorded ball trajectory
    and calculates the continuous distance of the player to the ball.
    
    :param player_trajectory (List): a list containing the ongoing coordinates of the player.
    :param ball_trajectory (List): a list containing the ongoing coordinates of the ball.
    :param show (bool) - Default True: if True, plots the time-series of the distances of the
                                       player towards the ball.
    """
    distances = []
    
    for p_traj, b_traj in zip(player_trajectory, ball_trajectory):
        dist = np.sqrt((p_traj[0] - b_traj[0]) ** 2 + (p_traj[1] - b_traj[1]) ** 2 + (b_traj[2]) ** 2)
        distances.append(dist)
        
        
    if show:
        print(f'[*] Player Average Distance to the Ball: {np.mean(distances)} +/- {np.std(distances)}')
        plt.plot(distances)
        plt.show()

    return distances

# done
def get_ball_possessor(home_coordinates: List, away_coordinates: List, ball_coordinates: List, show: bool=True) -> Tuple[bool, int]:
    """
    This function receives the coordinates of the home players, away players, and the ball, and
    evaluates the voronoi cells of the players and determines the player that is supposed to possess
    the ball
    
    :param home_coordinates (List): a list containing the coordinates of all players of the home team
    :param away_coordinates (List): a list containing the coordinates of all players of the away team
    :param ball_coordinates (List): a list containing 3 numbers corresponding to the ball's x, y, and z
                                    coordinates.
    :param show (bool) - Default True: if True, plots the Voronoi cell as well as the players and the ball
                                    
    :returns Tuple[bool, int]: returns a tuple; the first element is True if the ball owner belongs to the
                               home team, and False otherwise. The second element specifies the jersey number
                               of the ball owner.
    """
    # combining the coordinates
    all_coordinates = home_coordinates.copy()
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
        jersey_number = get_team_players(True)[player_index]['jersey_number']
    else:
        jersey_number = get_team_players(False)[player_index]['jersey_number']
    
    
    return home, jersey_number
    
# done
def find_progressive_passing_lane():
    # load the tracking data
    with open(f'all_Games/{GAME_NAME}-tracks.json', 'r') as handle:
        game_tracks = json.load(handle)
        
    instance_no = 500
    ball_trajectory = []
    home_coordinates = []
    away_coordinates = []
    
    print('[*] Tracking the player and the ball...')
    for index in range(min(len(game_tracks), instance_no)):
        track = game_tracks[index]
        # find the position of the ball
        if track['ball']['position'] is not None:
            ball_trajectory.append(track['ball']['position'])
        # add None if the position of the ball is missing
        else:
            ball_trajectory.append(None)
                
    print('[*] Filling the missing ball coordinates...')
    ball_trajectory = interpolate_ball_trajectory(ball_trajectory)
    
    track = game_tracks[min(len(game_tracks) - 1, instance_no)]
    # get the information of all players at an instance
    home_players = track['home_team']
    away_players = track['away_team']
    
    # find the home player's coordinates at this frame
    for player in home_players:
        home_coordinates.append(player['position'])
    
    # find the away player's coordinates at this frame
    for player in away_players:
        away_coordinates.append(player['position'])
        
    ball_coordinates = ball_trajectory[-1]
    
    get_ball_possessor(home_coordinates, away_coordinates, ball_coordinates)
        
    

######################################
# Plotting functions
######################################
# done
def plot_trajectory(start_x_coords: List, start_y_coords: List, end_x_coords: List, 
                    end_y_coords: List, ball_trajectory: List=None, timestamps: int=None) -> None:
    """
    This function receives the movement vectors of a player along the game and plots the trajectory.
    
    :param start_x_coords (List): player's x-coordinates of the start locations
    :param start_y_coords (List): player's y-coordinates of the start locations
    :param end_x_coords (List): player's x-coordinates of the end locations
    :param end_y_coords (List): player's y-coordinates of the end locations
    :param ball_trajectory (List) - Default None: if available plots the position of the ball as well.
    :param time_steps (int) - Default None: if set, as many snapshots as the specified timestamps are plotted
    """
    # extract the pitch size
    pitch_size = get_pitch_size()
    
    # define the pitch graphical object
    pitch = mpl.Pitch(pitch_type='skillcorner', 
                      pitch_length=pitch_size[0], 
                      pitch_width=pitch_size[1], 
                      axis=True, 
                      label=True)
    _, ax = pitch.draw(figsize=(9, 6))

    # define the used RGB colors in the pitch
    blue= (44/255,123/255,182/255)
    red = (1, 0, 0)
    
    # if the timestamp is given, the first 'timestamps' frames are plotted
    if timestamps:
        start_x_coords = start_x_coords[:timestamps]
        start_y_coords = start_y_coords[:timestamps]
        end_x_coords = end_x_coords[:timestamps]
        end_y_coords = end_y_coords[:timestamps]
        if ball_trajectory:
            ball_trajectory = ball_trajectory[:timestamps]
    
    # draw movement vectors as arrows
    pitch.arrows(start_x_coords, start_y_coords, end_x_coords, end_y_coords, 
                 alpha=0.8, 
                 color=blue, 
                 headaxislength=13, 
                 headlength=13, 
                 headwidth=14, 
                 width=2, 
                 ax=ax)
    
    # if we are also supposed to draw the ball trajectory
    if ball_trajectory:
        ball_x_coords = np.array(ball_trajectory)[:, 0]
        ball_y_coords = np.array(ball_trajectory)[:, 1]
        
        ball_start_x = []
        ball_start_y = []
        ball_end_x = []
        ball_end_y = []
        
        for index in range(len(ball_trajectory) - 1):
            ball_start_x.append(ball_trajectory[index][0])
            ball_start_y.append(ball_trajectory[index][1])
            ball_end_x.append(ball_trajectory[index + 1][0])
            ball_end_y.append(ball_trajectory[index + 1][1])
        
        # draw ball movement vectors as arrows
        pitch.arrows(ball_start_x, ball_start_y, ball_end_x, ball_end_y, 
                     alpha=0.4, 
                     color=red, 
                     headaxislength=3, 
                     headlength=3, 
                     headwidth=4, 
                     width=2, 
                     ax=ax)
            
        pitch.scatter(ball_x_coords, ball_y_coords, ax=ax, facecolor='yellow', s=5, edgecolor='k')
        
    plt.show()

# done
def plot_presence_heat_map(player_trajectory: List) -> None:
    """
    This function receives the trajectory of the player and plots the heat map
    of the player's presence in the pitch.
    
    :param player_trajecotry (List): a list containing the sequence of the player's
                                     coordinates.
    """
    pitch_size = get_pitch_size()
    
    pitch = mpl.Pitch(line_color='white',
                      pitch_type='skillcorner', 
                      pitch_length=pitch_size[0], 
                      pitch_width=pitch_size[1], 
                      axis=True, 
                      label=True)
    
    player_trajectory = np.array(player_trajectory)
    statistics = pitch.bin_statistic(player_trajectory[:, 0], player_trajectory[:, 1], statistic='count', bins=(25, 25))
    
    fig, ax = pitch.draw(figsize=(9, 6))
    fig.set_facecolor('#22312b')
    
    pcm = pitch.heatmap(statistics, ax=ax, cmap='hot', edgecolors='#22312b')
    fig.colorbar(pcm, ax=ax, shrink=0.6)
    plt.show()

# done
def plot_game_snapshot(instance_no: int) -> None:
    """
    This function receives the index of a snapshot of the game and plots the coordinates of the home-team
    players (blue), and away players (red) as well as the ball (yellow) on the pitch.
    
    :param instance_no (int): the index of the snapshot we want to illustrate.
    """
    # loading the games
    with open(f'all_Games/{GAME_NAME}-tracks.json', 'r') as handle:
        game_tracks = json.load(handle)
        
    ball_trajectory = []
    home_coordinates = []
    away_coordinates = []
    
    print('[*] Tracking the player and the ball...')
    for index in range(min(len(game_tracks), instance_no)):
        track = game_tracks[index]
        # find the position of the ball
        if track['ball']['position'] is not None:
            ball_trajectory.append(track['ball']['position'])
        # add None if the position of the ball is missing
        else:
            ball_trajectory.append(None)
                
    print('[*] Filling the missing ball coordinates...')
    ball_trajectory = interpolate_ball_trajectory(ball_trajectory)
    
    track = game_tracks[min(len(game_tracks) - 1, instance_no)]
    # get the information of all players at an instance
    home_players = track['home_team']
    away_players = track['away_team']
    
    # find the home player's coordinates at this frame
    for player in home_players:
        home_coordinates.append(player['position'])
    
    # find the away player's coordinates at this frame
    for player in away_players:
        away_coordinates.append(player['position'])
        
    # extract the pitch size
    pitch_size = get_pitch_size()
    
    # define the pitch graphical object
    pitch = mpl.Pitch(pitch_type='skillcorner', 
                      pitch_length=pitch_size[0], 
                      pitch_width=pitch_size[1], 
                      axis=True, 
                      label=True)
    _, ax = pitch.draw(figsize=(9, 6))
            
    pitch.scatter(np.array(home_coordinates)[:, 0], np.array(home_coordinates)[:, 1], ax=ax, facecolor='blue', s=30, edgecolor='k')
    pitch.scatter(np.array(away_coordinates)[:, 0], np.array(away_coordinates)[:, 1], ax=ax, facecolor='red', s=30, edgecolor='k')
    pitch.scatter([ball_trajectory[-1][0]], [ball_trajectory[-1][1]], ax=ax, facecolor='yellow', s=20, edgecolor='k')
        
    plt.show()

# done
def plot_voronoi_cell(instance_no: int) -> None:
    """
    This function receives the number of a snapshot of the game and plots the Voronoi cell
    of the players in that snapshot.
    
    :param instance_no (int): an integer specifying the index of the snapshot of the game
    """
    # load the tracking data
    with open(f'all_Games/{GAME_NAME}-tracks.json', 'r') as handle:
        game_tracks = json.load(handle)
        
    ball_trajectory = []
    home_coordinates = []
    away_coordinates = []
    
    print('[*] Tracking the player and the ball...')
    for index in range(min(len(game_tracks), instance_no)):
        track = game_tracks[index]
        # find the position of the ball
        if track['ball']['position'] is not None:
            ball_trajectory.append(track['ball']['position'])
        # add None if the position of the ball is missing
        else:
            ball_trajectory.append(None)
                
    print('[*] Filling the missing ball coordinates...')
    ball_trajectory = interpolate_ball_trajectory(ball_trajectory)
    
    track = game_tracks[min(len(game_tracks) - 1, instance_no)]
    # get the information of all players at an instance
    home_players = track['home_team']
    away_players = track['away_team']
    
    # find the home player's coordinates at this frame
    for player in home_players:
        home_coordinates.append(player['position'])
    
    # find the away player's coordinates at this frame
    for player in away_players:
        away_coordinates.append(player['position'])
        
    ball_coordinates = ball_trajectory[-1]
    all_coordinates = home_coordinates.copy()
    all_coordinates.extend(away_coordinates)
    
    voronoi_cells = Voronoi(np.array(all_coordinates))
    voronoi_plot_2d(voronoi_cells, show_points=False, show_vertices=False, line_colors='darkgreen')
    plt.scatter([ball_coordinates[0]], [ball_coordinates[1]], c='orange')
    plt.scatter(np.array(home_coordinates)[:, 0], np.array(home_coordinates)[:, 1], s=35, c='blue')
    plt.scatter(np.array(away_coordinates)[:, 0], np.array(away_coordinates)[:, 1], s=35, c='red')
    plt.show()
    
    plt.show()
    

######################################
# Feature vector extraction functions
######################################

# done
def player_location_to_feature_vector(trajectory: List, grid_size: Tuple=(10, 10)) -> np.ndarray:
    """
    This function receives the trajectory of a player along with a grid size. The function
    divids the pitch into a grid with the specified size. At each timestamp (each element of
    the trajectory) the function assigns 1 to the cell in which the player is located.
    The final bit vectors trace is returned by the function.
    
    :param trajectory (List): a list contining the coordinates of the player
    :param grid_size (Tuple) - Default (10, 10): the size of the pitch grid.
    """
    # create the feature vector for the player's location (if the grid size is 10x10, the trace has 
    # feature vetors of dimension 100)
    feature_vec_trace = np.zeros((grid_size[0] * grid_size[1], len(trajectory)))
    
    # setup the graphical pitch
    pitch_size = get_pitch_size()
    pitch = mpl.Pitch(line_color='white',
                      pitch_type='skillcorner', 
                      pitch_length=pitch_size[0], 
                      pitch_width=pitch_size[1], 
                      axis=True, 
                      label=True)
    
    # find the location of the player on the grid
    for index, position in enumerate(trajectory):
        stat = pitch.bin_statistic([position[0]], [position[1]], statistic='count', bins=grid_size)
        # convert the location to the one-hot vector (or the bit vector)
        feature_vec_trace[:, index] = stat['statistic'].flatten().astype(np.int32)
    
    return feature_vec_trace

# done
def get_teammate_density_vector(trajectory: List, ball_trajectory: List, jersey_number: int, home: bool=True) -> np.ndarray:
    """
    This function receives the player trajectory, ball trajectory, the jersey number of the target player,
    whether the player belongs to the home team. The function computes the gaze vectors from the trajectory
    and the ball trajectory. Then it iterates through the tracking data and for the snapshots in which the 
    target player is present it calculates the visible polygon of the target player and count the teammates
    inside the visible polygon. The number of teammates inside the visible polygon is turned into a one-hot
    vector (if the i-th element of the vector is 1 it indicates that there are i teammates inside the visible
    polygon)
    
    :param trajectory (List): a list containing the coordinates of the target player.
    :param ball_trajectory (List): a list containing the coordinates of the ball.
    :param jersey_number (int): the jersey number of the target player.
    :param home (bool) - Default True: indicates whether the player belongs to the home team.
    """
    # teammate density vector is a trace with as many elements as the player trajectory; each
    # element is an 11 dimensional vector representing the number of teammates inside the player's
    # visible polygon
    teammate_density_vector = np.zeros((11, len(trajectory)))
    
    # specify whether the player belongs to the home team
    key = 'home_team' if home else 'away_team'
    
    # open the tracking data to extract other players' coordinates
    with open(f'all_Games/{GAME_NAME}-tracks.json', 'r') as handle:
        game_tracks = json.load(handle)
        
    # compute the player's gaze
    gaze_start_x, gaze_start_y, gaze_end_x, gaze_end_y = get_player_gaze(trajectory, ball_trajectory)
    
    # iterate through the tracking data
    index = 0
    for track in game_tracks:
        if index == len(gaze_start_x):
            break
        # get the coordinates of all players in the team
        players = track[key]
        # check if the target player exists in the current frame
        target_player = list(filter(lambda x: x['jersey_number'] == jersey_number, players))
        
        # if the target player exists in the current frame
        if len(target_player) > 0:
            # count the number of teammates in his/her visible polygon
            num_teammates_ahead = 0
            # find the players visible polygon
            visible_polygon = get_visible_polygon(gaze_start_x[index], gaze_start_y[index], gaze_end_x[index], gaze_end_y[index])
            visible_polygon = Polygon(visible_polygon)
            # iterate through all the teammates
            for player in players:
                if player['jersey_number'] != target_player[0]['jersey_number']:  
                    coords = Point(*player['position'])
                    # check if the player's coordinate lies inside the visible polygon
                    if visible_polygon.contains(coords):
                        num_teammates_ahead += 1
            # update the feature vector
            teammate_density_vector[num_teammates_ahead, index] = 1
            index += 1
            
    return teammate_density_vector

# done
def get_opponent_density_vector(trajectory: List, ball_trajectory: List, jersey_number: int, home: bool=True) -> np.ndarray:
    """
    This function receives the player trajectory, ball trajectory, the jersey number of the target player,
    whether the player belongs to the home team. The function computes the gaze vectors from the trajectory
    and the ball trajectory. Then it iterates through the tracking data and for the snapshots in which the 
    target player is present it calculates the visible polygon of the target player and count the opponents
    inside the visible polygon. The number of opponents inside the visible polygon is turned into a one-hot
    vector (if the i-th element of the vector is 1 it indicates that there are i teammates inside the visible
    polygon)
    
    :param trajectory (List): a list containing the coordinates of the target player.
    :param ball_trajectory (List): a list containing the coordinates of the ball.
    :param jersey_number (int): the jersey number of the target player.
    :param home (bool) - Default True: indicates whether the player belongs to the home team.
    """
    # teammate density vector is a trace with as many elements as the player trajectory; each
    # element is an 11 dimensional vector representing the number of teammates inside the player's
    # visible polygon
    opponent_density_vector = np.zeros((12, len(trajectory)))
    
    # specify whether the player belongs to the home team
    key = 'home_team' if home else 'away_team'
    opponent_key = 'away_team' if home else 'home_team'
    
    # open the tracking data to extract other players' coordinates
    with open(f'all_Games/{GAME_NAME}-tracks.json', 'r') as handle:
        game_tracks = json.load(handle)
        
    # compute the player's gaze
    gaze_start_x, gaze_start_y, gaze_end_x, gaze_end_y = get_player_gaze(trajectory, ball_trajectory)
    
    # iterate through the tracking data
    index = 0
    for track in game_tracks:
        if index == len(gaze_start_x):
            break
        # get the coordinates of all players in the team
        players = track[key]
        # check if the target player exists in the current frame
        target_player = list(filter(lambda x: x['jersey_number'] == jersey_number, players))
        
        # get the coordinates of all players in the team
        opponent_players = track[opponent_key]
        
        # if the target player exists in the current frame
        if len(target_player) > 0:
            # count the number of teammates in his/her visible polygon
            num_opponents_ahead = 0
            # find the players visible polygon
            visible_polygon = get_visible_polygon(gaze_start_x[index], gaze_start_y[index], gaze_end_x[index], gaze_end_y[index])
            visible_polygon = Polygon(visible_polygon)
            # iterate through all the teammates
            for player in opponent_players:
                coords = Point(*player['position'])
                # check if the player's coordinate lies inside the visible polygon
                if visible_polygon.contains(coords):
                    num_opponents_ahead += 1
            # update the feature vector
            opponent_density_vector[num_opponents_ahead, index] = 1
            index += 1
            
    return opponent_density_vector

# done
def has_ball_feature_vector(jersey_number: int, ball_trajectory: List, home: bool=True) -> bool:
    """
    This function receives a jersey number, the ball trajectory, and whether the player
    belongs to the home team or away team. The function then returns a True/False value
    for each instance of the game indicating whether the player possesses the ball in that
    instance.
    
    :param jersey_number (int): the jersey number of the target player.
    :param ball_trajectory (List): a list containing the coordinates of the ball at each
                                   instance of the gaem
    :param home (bool) - Default True: determines whether the player belongs to the home team
                                       or away team.
    """
    
    # load the tracking data
    with open(f'all_Games/{GAME_NAME}-tracks.json', 'r') as handle:
        game_tracks = json.load(handle)
        
    feature_vector = np.zeros((1, len(ball_trajectory)))
    key = 'home_team' if home else 'away_team'
    index = 0
    for track in game_tracks:
        # get the information of all players at an instance
        players = track[key]
        # check whether the target player has any record in this frame
        target_player = list(filter(lambda x: x['jersey_number'] == jersey_number, players))
        
        # if the player's record exists in this frame...
        if len(target_player) > 0:
            home_coordinates = []
            away_coordinates = []
            
            home_players = track['home_team']
            away_players = track['away_team']
            
            # find the home player's coordinates at this frame
            for player in home_players:
                home_coordinates.append(player['position'])
            
            # find the away player's coordinates at this frame
            for player in away_players:
                away_coordinates.append(player['position'])
                
            ball_coordinates = ball_trajectory[index]
            
            belongs_to_home, jersey_num = get_ball_possessor(home_coordinates, away_coordinates, ball_coordinates, show=False)
            
            if (belongs_to_home == home) and jersey_num == jersey_number:
                feature_vector[0, index] = 1
            index += 1
    
    return feature_vector


if __name__ == '__main__':
    # printer = pprint.PrettyPrinter()
    # with open('all_Games/AIK__BK Häcken.1-stats.json', 'r') as handle:
    #     stats = json.load(handle)
    
    # print('[*] Game Distinct Events:')
    # printer.pprint(get_game_distinct_events())
    
    # print('\n[*] Home Team Players Information:')
    # printer.pprint(get_team_players(home=True))
    
    # print('\n[*] Away Team Players Information:')
    # printer.pprint(get_team_players(home=False))
    
    # print('\n[*] Plotting Different Snapshots of the Game:')
    # plot_game_snapshot(20)
    # plot_game_snapshot(500)
    # plot_game_snapshot(1000)
    
    # # lets say we want to track the location of the Mikael Lustig with jersey number 33
    # trajectory, track_id, ball_trajectory = track_player_coords(36, home=True, track_ball=True)
    # print(track_id)
    # print(len(trajectory), len(ball_trajectory))
    # s_x, s_y, e_x, e_y = movement_arrows_from_trajectory(trajectory)
    # plot_trajectory(s_x, s_y, e_x, e_y, ball_trajectory=ball_trajectory, timestamps=300)
    # plot_presence_heat_map(trajectory)
    # player_ball_distance_history(trajectory, ball_trajectory, show=True)
    # player_events = get_player_events(track_id)
    
    # player_events_types =  np.unique([event['type'] for event in player_events], return_counts=True)
    # plt.bar(player_events_types[0].tolist(), player_events_types[1].tolist())
    # plt.show()
    
    # gaze_start_x, gaze_start_y, gaze_end_x, gaze_end_y = get_player_gaze(trajectory, ball_trajectory)
    # plot_trajectory(gaze_start_x, gaze_start_y, gaze_end_x, gaze_end_y, ball_trajectory=ball_trajectory, timestamps=5)
    # loc_bit_vector = player_location_to_feature_vector(trajectory, grid_size=(5, 5))
    # print(loc_bit_vector.shape[1])
    # print(loc_bit_vector[:, 100], trajectory[100])
    
    # teammate_density_bit_vector = get_teammate_density_vector(trajectory, ball_trajectory, 36, True)
    # print(teammate_density_bit_vector.shape)
    # print(teammate_density_bit_vector[:, 100])
    
    # opponent_density_bit_vector = get_opponent_density_vector(trajectory, ball_trajectory, 36, True)
    # print(opponent_density_bit_vector.shape)
    # print(opponent_density_bit_vector[:, 100])
    
    # find_progressive_passing_lane()
    # has_ball_vector = has_ball_feature_vector(36, ball_trajectory, True)
    # print(has_ball_vector.shape)
    plot_voronoi_cell(520)
    plot_voronoi_cell(530)
    plot_voronoi_cell(540)
    plot_voronoi_cell(550)
    
