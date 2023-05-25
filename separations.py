from typing import List
from typing import Tuple

from utils import GameLoader
from utils import PlayerTracker


def team_possession_separation(traces: List, game: GameLoader, player_tracker: PlayerTracker) -> Tuple[List, List]:
    """
    This function receives the traces, the game information, and the target player information.
    The function splits the traces into positive and negative traces where positive traces are
    the instances in which the player's team possesses the ball and the negative traces are the
    instances where the opponent team possesses the ball.
    
    :param traces (List): a list of traces made by the Pipeline.
    :param game (GameLoader): an instance of GameLoader which contains the information about the
                              the game including the game tracking information
    :param player_tracker (PlayerTracker): an instance of PlayerTracker which contains the target
                                           player information including their position and whether
                                           they belong to the home team or away team
    """
    # crating empty lists for positive and negative traces
    positive_traces = []
    negative_traces = []
    
    # detemining the player team
    player_team = 'home_team' if player_tracker.home else 'away_team'
    
    # creating an empty list to only store the tracks in which the target player exists
    possession_in_player_instances = []
    # creating a list to store the instance numbers at which the ball possession changes from one team to another
    changing_points = []
    
    # fill the possession_in_player_instances list
    for index in range(len(game.game_tracks) - 1):
        if index not in player_tracker.missing_instances:
            possession_in_player_instances.append(game.game_tracks[index])
    
    # finding the instances at which the ball possession changes
    for index in range(len(possession_in_player_instances) - 1):
        current_track = possession_in_player_instances[index]
        next_track = possession_in_player_instances[index + 1]
        
        if next_track['ball']['team'] != current_track['ball']['team']:
            changing_points.append(index + 1)
    
    # separting the instances into positive and negative sequences
    # if at the beginning the ball is possessed by the player's team
    # it is considered as a positive sequence and at each changing point
    # the possession alternates
    # If at the beginning the ball is possessed by the opponent team 
    # it is considered as a negative sequence and at each changing point
    # the possession alternates
    print('[*] Separating the traces based on which team possesses the ball...')
    if possession_in_player_instances[0]['ball']['team'] == player_team:
        for i, _ in enumerate(changing_points):
            if i == 0:
                positive_traces.append(traces[:changing_points[i]])
            elif i % 2 == 0:
                positive_traces.append(traces[changing_points[i - 1]: changing_points[i]])
            else:
                negative_traces.append(traces[changing_points[i - 1]: changing_points[i]])
            
            if i == len(changing_points) - 1:
                if i % 2 == 0:
                    negative_traces.append(traces[changing_points[i]:])
                else:
                    positive_traces.append(traces[changing_points[i]:])
    else:
        for i, _ in enumerate(changing_points):
            if i == 0:
                negative_traces.append(traces[:changing_points[i]])
            elif i % 2 == 0:
                negative_traces.append(traces[changing_points[i - 1]: changing_points[i]])
            else:
                positive_traces.append(traces[changing_points[i - 1]: changing_points[i]])
            
            if i == len(changing_points) - 1:
                if i % 2 == 0:
                    positive_traces.append(traces[changing_points[i]:])
                else:
                    negative_traces.append(traces[changing_points[i]:])
        
                   
    return positive_traces, negative_traces