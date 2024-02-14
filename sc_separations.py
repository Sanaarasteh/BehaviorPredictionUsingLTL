import random

from typing import List
from typing import Tuple

from sc_utils import GameLoader
from sc_utils import PlayerTracker
from sc_utils import AllTracker
from sc_utils import PassAnalyzer


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
    player_team = 'home team' if player_tracker.home else 'away team'
    
    # creating an empty list to only store the tracks in which the target player exists
    player_possession_instances = []
    # creating a list to store the instance numbers at which the ball possession changes from one team to another
    changing_points = []
    
    # fill the player_possession_instances list
    for index in range(len(game.game_tracks)):
        if index in player_tracker.present_instances:
            player_possession_instances.append(game.game_tracks[index])
    
    # finding the instances at which the ball possession changes
    changing_points.append(0)
    possessions = []
    for index in range(len(player_possession_instances) - 1):
        current_track = player_possession_instances[index]
        next_track = player_possession_instances[index + 1]
        
        if next_track['possession']['group'] != current_track['possession']['group']:
            if current_track['possession']['group'] != 'out':
                changing_points.append(index)
                possessions.append(current_track['possession']['group'])
            if next_track['possession']['group'] != 'out':
                changing_points.append(index + 1)
                
        if index == len(player_possession_instances) - 2:
            possessions.append(current_track['possession']['group'])
            changing_points.append(len(player_possession_instances) - 1)
    
    # separting the instances into positive and negative sequences
    # if at the beginning the ball is possessed by the player's team
    # it is considered as a positive sequence and at each changing point
    # the possession alternates
    # If at the beginning the ball is possessed by the opponent team 
    # it is considered as a negative sequence and at each changing point
    # the possession alternates
    print('[*] Separating the traces based on which team possesses the ball...')
    for i in range(len(possessions)):
        if possessions[i] == player_team:
            positive_traces.extend(traces[changing_points[2 * i]: changing_points[2 * i + 1] + 1])
        else:
            negative_traces.extend(traces[changing_points[2 * i]: changing_points[2 * i + 1] + 1])
                   
    return positive_traces, negative_traces


def attack_defense_separation(traces: List, game: GameLoader, all_tracker: AllTracker, player_tracker: PlayerTracker, 
                              attack_mode: str='penetration', defense_mode: str='retreat') -> Tuple[List, List]:
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
    
    # getting the attacking indices of the player's team
    attacking_indices = all_tracker.attack_detection(player_tracker.home, mode=attack_mode)
    
    # getting the defense indices of the player's team
    defense_indices = all_tracker.defense_detection(player_tracker.home, mode=defense_mode)
    
    # pruning the indices so that we can just have the indices in which the player exists
    positive_indices = []
    
    for index_pair in attacking_indices:
        for i in range(index_pair[0], index_pair[1] + 1):
            if game.player_exists(i, player_tracker.track_id)[0]:
                positive_indices.append(player_tracker.present_instances.index(i))
                
    negative_indices = []
    for index_pair in defense_indices:
        for i in range(index_pair[0], index_pair[1] + 1):
            if game.player_exists(i, player_tracker.track_id)[0]:
                negative_indices.append(player_tracker.present_instances.index(i))
    
    for index in positive_indices:
        try:
            positive_traces.append(traces[index])
        except:
            continue
    
    for index in negative_indices:
        try:
            negative_traces.append(traces[index])
        except:
            continue
                
                   
    return positive_traces, negative_traces


def passing_style_separation(traces: List, game: GameLoader, player_tracker: PlayerTracker, all_tracker: AllTracker) -> Tuple[List, List]:
    # crating empty lists for positive and negative traces
    positive_traces = []
    negative_traces = []
    
    
    pass_analyzer = PassAnalyzer('pass_instances.csv', game, all_tracker, n_cluster=2)
    
    cluster_instances = pass_analyzer.instance_labels
    
    # extract the instances of positive and negative traces
    possible_positive_indices = cluster_instances[0]
    possible_negative_indices = cluster_instances[1]
    
    
    # pruning the indices so that we can just have the indices in which the player exists
    positive_indices = []
    
    for index in possible_positive_indices:
        if game.player_exists(index, player_tracker.track_id)[0]:
            positive_indices.append(player_tracker.present_instances.index(index))
                
    negative_indices = []
    for index in possible_negative_indices:
        if game.player_exists(index, player_tracker.track_id)[0]:
            negative_indices.append(player_tracker.present_instances.index(index))
    
    for index in positive_indices:
        try:
            positive_traces.append(traces[index])
        except:
            continue
    
    for index in negative_indices:
        try:
            negative_traces.append(traces[index])
        except:
            continue
        
    
    if len(positive_traces) / len(negative_traces) > 1.5:
        positive_traces = random.sample(positive_traces, len(negative_traces))
    
    if len(negative_traces) / len(positive_indices) > 1.5:
        negative_traces = random.sample(negative_traces, len(positive_traces))
        
                   
    return positive_traces, negative_traces
