import numpy as np
import mplsoccer as mpl

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Dict
from typing import Tuple
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString

from utils import GameLoader
from utils import PlayerTracker
from utils import AllTracker


class BaseFeature(ABC):
    """
    This is an abstract class with no implmentation. All features must inherit this class.
    Inheriting this class forces all feature classes to follow the same convention of having
    a "generate" method to generate the feature vector and a "translate" method to translate
    the feature vector to a literal proposition, proper to be fed to an LTL formula generator.
    """
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def generate(self, *args) -> None:
        """
        This method specifies how the features are generated from the data.
        NOTE: update a self.feature attribute in this class to contain the generated features
              (similar to the example below)
        '''
        def generate(self, x):
            self.feature = np.zeros((100, 10))
            
            for i in range(100):
                self.feature[i] = //update based on x and i
        """
        pass
    
    @abstractmethod
    def translate(self) -> List:
        """
        This method specifies how the features are translated.
        NOTE: read the self.feature attribute update from the generate method.
        NOTE: this method must return a list; each element of the list should be
              a proposition describing the feature generated for a particular instance 
              of the game
              
        '''
        def translate(self):
            propositions = []
            
            for feature in self.features:
                proposition.append(//some translation of the feature)
            
            return propositions
        """
        pass


class PlayerLocFeature(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker, grid_size: Tuple=(10, 10)) -> None:
        """
        :param game (GameLoader): an instance of the game containing the information of
                                  pitch and the players.
        :param grid_size (Tuple) - Default (10, 10): the size of the pitch grid.
        """
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.grid_size = grid_size
        self.features = None
        
        
    def generate(self, on_possession=False) -> None:
        """
        This function uses the trajectory of a player. The function
        divides the pitch into a grid with the specified size. At each timestamp (each element of
        the trajectory) the function assigns 1 to the cell in which the player is located.
        The final bit vectors trace is returned by the function.
        """
        # create the feature vector for the player's location (if the grid size is 10x10, the trace has 
        # feature vetors of dimension 100)
        if self.player.player_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.player.track()
        
        if self.all_tracker.ball_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.all_tracker.track()
            
        print('[*] Generating the feature vectors for the player\'s locations...')
        self.features = np.zeros((len(self.player.player_trajectory), self.grid_size[0] * self.grid_size[1]))
        
        # setup the graphical pitch
        pitch_size = self.game.get_pitch_size()
        pitch = mpl.Pitch(line_color='white',
                          pitch_type='skillcorner', 
                          pitch_length=pitch_size[0], 
                          pitch_width=pitch_size[1], 
                          axis=True, 
                          label=True)
        
        # find the location of the player on the grid
        for index in tqdm(range(len(self.player.player_trajectory))):
            if on_possession:
                if (self.player.home and self.player.team_possession_trajectory[index] == 'home_team') or (not self.player.home and self.player.team_possession_trajectory[index] == 'away_team'):
                    position = self.player.player_trajectory[index]
                    stat = pitch.bin_statistic([position[0]], [position[1]], statistic='count', bins=self.grid_size)
                    # convert the location to the one-hot vector (or the bit vector)
                    self.features[index] = stat['statistic'].flatten().astype(np.int32)
            else:
                position = self.player.player_trajectory[index]
                stat = pitch.bin_statistic([position[0]], [position[1]], statistic='count', bins=self.grid_size)
                # convert the location to the one-hot vector (or the bit vector)
                self.features[index] = stat['statistic'].flatten().astype(np.int32)
    
    def translate(self) -> List:
        """
        This function translates the numerical features representing the location of a target
        player to the propositions describing the player's location; for example, if the player's
        location feature is [1, 0, 0, 0], it will be translated to (at player location 0).
        """
        assert self.features is not None, '[!] There are no features available! try generating the features first.'
        propositions = []
        
        # iterating thourgh the features
        for feature in self.features:
            # finding the location of the player
            location = str(np.argmax(feature))
            
            # translating the location
            translation = f'(at player location {location})'
            
            # updating the propositions
            propositions.append(translation)
        
        return propositions
        
        
class TeammateDensityFeature(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.features = None
    
    
    def generate(self, on_possession=False) -> None:
        """
        This function uses the player trajectory, ball trajectory, the jersey number of the target player,
        whether the player belongs to the home team. The function computes the gaze vectors from the trajectory
        and the ball trajectory. Then it iterates through the tracking data and for the snapshots in which the 
        target player is present it calculates the visible polygon of the target player and count the teammates
        inside the visible polygon. The number of teammates inside the visible polygon is turned into a one-hot
        vector (if the i-th element of the vector is 1 it indicates that there are i teammates inside the visible
        polygon)
        """
        # teammate density vector is a trace with as many elements as the player trajectory; each
        # element is an 11 dimensional vector representing the number of teammates inside the player's
        # visible polygon
        if self.player.player_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.player.track()
        
        if len(self.player.visible_polygons) == 0:
            print('[!] Found no information about the visible polygons of the player. Filling the information...')
            self.player.get_visible_polygon()
            
            
        self.features = np.zeros((len(self.player.visible_polygons), 11))
        
        # specify whether the player belongs to the home team
        key = 'home_team' if self.player.home else 'away_team'
        
        # iterate through the tracking data
        print('[*] Generating the feature vectors for the player\'s teammates density...')
        index = 0        
        for j in tqdm(range(len(self.game.game_tracks))):
            if index == len(self.player.visible_polygons):
                break
            
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home_team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away_team')):
                    continue
            
            track = self.game.game_tracks[j]
            # get the information of all players at an instance
            players = track[key]
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['jersey_number'] == self.player.jersey_number, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                # find the player's coordinates at this frame
                # count the number of teammates in his/her visible polygon
                num_teammates_ahead = 0
                # find the players visible polygon
                visible_polygon = self.player.visible_polygons[index]
                visible_polygon = Polygon(visible_polygon)
                # iterate through all the teammates
                for player in players:
                    if player['jersey_number'] != target_player[0]['jersey_number']:  
                        coords = Point(*player['position'])
                        # check if the player's coordinate lies inside the visible polygon
                        if visible_polygon.contains(coords):
                            num_teammates_ahead += 1
                # update the feature vector
                self.features[index, num_teammates_ahead] = 1
                index += 1
                
    def translate(self) -> List:
        """
        This function translates the numerical features representing the number of teammates 
        in the player's visible polygon to their appropriate propositions; 
        for example, if the player's teammate density feature is [1, 0, 0, 0], it will be 
        translated to (ahead teammates 0).
        """
        propositions = []
        
        for feature in self.features:
            num_teammates = np.argmax(feature)
            translation = f'(ahead teammates {num_teammates})'
            # translation = str(feature.tolist())
            propositions.append(translation)
        
        return propositions


class OpponentsDensityFeature(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.features = None
    
    
    def generate(self, on_possession=False) -> None:
        """
        This function uses the player trajectory, ball trajectory, the jersey number of the target player,
        whether the player belongs to the home team. The function computes the gaze vectors from the trajectory
        and the ball trajectory. Then it iterates through the tracking data and for the snapshots in which the 
        target player is present it calculates the visible polygon of the target player and count the pponents
        inside the visible polygon. The number of opponents inside the visible polygon is turned into a one-hot
        vector (if the i-th element of the vector is 1 it indicates that there are i opponents inside the visible
        polygon)
        """
        # opponents density vector is a trace with as many elements as the player trajectory; each
        # element is an 12 dimensional vector representing the number of opponents inside the player's
        # visible polygon
        if self.player.player_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.player.track()
        
        if len(self.player.visible_polygons) == 0:
            print('[!] Found no information about the visible polygons of the player. Filling the information...')
            self.player.get_visible_polygon()
            
            
        self.features = np.zeros((len(self.player.visible_polygons), 12))
        
        # specify whether the player belongs to the home team
        home_key = 'home_team' if self.player.home else 'away_team'
        away_key = 'away_team' if self.player.home else 'home_team'
        
        # iterate through the tracking data
        print('[*] Generating the feature vectors for the player\'s opponents density...')
        index = 0        
        for j in tqdm(range(len(self.game.game_tracks))):
            if index == len(self.player.visible_polygons):
                break
            
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home_team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away_team')):
                    continue
                
            track = self.game.game_tracks[j]
            # get the information of all players at an instance
            home_players = track[home_key]
            away_players = track[away_key]
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['jersey_number'] == self.player.jersey_number, home_players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                # find the player's coordinates at this frame
                # count the number of opponents in his/her visible polygon
                num_opponents_ahead = 0
                # find the players visible polygon
                visible_polygon = self.player.visible_polygons[index]
                visible_polygon = Polygon(visible_polygon)
                # iterate through all the opponents
                for player in away_players:
                    if player['jersey_number'] != target_player[0]['jersey_number']:  
                        coords = Point(*player['position'])
                        # check if the player's coordinate lies inside the visible polygon
                        if visible_polygon.contains(coords):
                            num_opponents_ahead += 1
                # update the feature vector
                self.features[index, num_opponents_ahead] = 1
                index += 1
                
    def translate(self) -> List:
        """
        This function translates the numerical features representing the number of opponents 
        in the player's visible polygon to their appropriate propositions; 
        for example, if the player's teammate density feature is [1, 0, 0, 0], it will be 
        translated to (ahead opponents 0).
        """
        propositions = []
        
        for feature in self.features:
            num_opponents = np.argmax(feature)
            translation = f'(ahead opponents {num_opponents})'
            # translation = str(feature.tolist())
            propositions.append(translation)
        
        return propositions


class HasBallFeature(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.features = None
        
    def generate(self, on_possession=False) -> None:
        """
        This function uses the jersey number, ball trajectory, and whether the player
        belongs to the home team or away team. The function then returns a True/False value
        for each instance of the game indicating whether the player possesses the ball in that
        instance.
        
        :param jersey_number (int): the jersey number of the target player.
        :param ball_trajectory (List): a list containing the coordinates of the ball at each
                                    instance of the gaem
        :param home (bool) - Default True: determines whether the player belongs to the home team
                                        or away team.
        """
        if self.all_tracker.ball_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.all_tracker.track()
            
        self.features = np.zeros((len(self.player.player_trajectory), 1))
        key = 'home_team' if self.player.home else 'away_team'
        
        print('[*] Generating the feature vectors for the player\'s ball possession...')
        index = 0
        for j in tqdm(range(len(self.game.game_tracks))):
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home_team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away_team')):
                    continue
                
            track = self.game.game_tracks[j]
            
            # get the information of all players at an instance
            players = track[key]
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['jersey_number'] == self.player.jersey_number, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:          
                belongs_to_home, jersey_num = self.all_tracker.get_ball_possessor(j)
                
                if belongs_to_home == self.player.home and jersey_num == self.player.jersey_number:
                    self.features[index, 0] = 1
                    
                index += 1
                
        print(f' in {self.features.sum()} instances the player had the ball')
        
    def translate(self) -> List:
        propositions = []
        
        for feature in self.features.flatten():
            translation = '(has_ball)' if feature == 1 else '~(has_ball)'
            propositions.append(translation)
            
        return propositions
        

class CanPassToFeature(BaseFeature):
    """
    This feature determines at each instance of the game, the number of teammates to 
    which the player can pass the ball. Therefore, this feature consists of a 10-dimensional 
    vector (1 component for each teammate), where each component is 1 if that teammate is 
    reachable.
    
    The criteria for reachability is that there should be adjacent voronoi cells occupied by
    teammates taking the ball from the passer to receiver.
    """
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.features = None
        
    def generate(self, on_possession=False) -> None:
        if self.all_tracker.ball_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.all_tracker.track()
        
        self.features = np.zeros((len(self.player.player_trajectory), 1))
        key = 'home_team' if self.player.home else 'away_team'
        
        print('[*] Generating the feature vectors for the player\'s number of reachable teammates...')
        index = 0
        for j in tqdm(range(len(self.game.game_tracks))):
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home_team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away_team')):
                    continue
                
            track = self.game.game_tracks[j]
            
            # get the information of all players at an instance
            players = track[key]
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['jersey_number'] == self.player.jersey_number, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:          
                belongs_to_home, jersey_num = self.all_tracker.get_ball_possessor(j)
                
                # if the player possesses the ball
                if belongs_to_home == self.player.home and jersey_num == self.player.jersey_number:
                    # compute the number of teammates reachable to the player
                    self.features[index, 0] = self.all_tracker.get_reachable_teammates(j, jersey_num, self.player.home)
                else:
                    self.features[index, 0] = 0
                    
                index += 1

        print(f' in total the player has had {self.features.sum()} reachable teammates throughout the game')
        
    def translate(self) -> List:
        propositions = []
        
        for feature in self.features.flatten():
            translation = f'(has {int(feature)} reachable teammates)'
            propositions.append(translation)
            
        return propositions


class ProgressivePassingLane(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker, threshold: str) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.features = None
        self.thresh = threshold
        
        if threshold == 'low':
            self.threshold = 3
        elif threshold == 'medium':
            self.threshold = 5
        elif threshold == 'high':
            self.threshold = 10
        else:
            raise Exception('[!] Invalid threshold; expected "low", "medium", "high".')
        
    def generate(self, on_possession=False) -> None:
        """
        This function uses the player trajectory, ball trajectory, the jersey number of the target player,
        whether the player belongs to the home team, and a threshold value. The function loads the visible
        polygon of the data for each frame. In each visible polygon, the passing lanes from the player to
        each teammate in the visible polygon is calculated. If there is an opponent player whose distance
        from the passing lane is less than the specified threshold, the passing lane is considered unsafe.
        Otherwise the passing lane is considered safe. The function counts the safe passing lanes inside 
        the visible polygon of the player.
        """
        # teammate density vector is a trace with as many elements as the player trajectory; each
        # element is an 11 dimensional vector representing the number of teammates inside the player's
        # visible polygon
        if self.player.player_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.player.track()
        
        if len(self.player.visible_polygons) == 0:
            print('[!] Found no information about the visible polygons of the player. Filling the information...')
            self.player.get_visible_polygon()
            
            
        self.features = np.zeros((len(self.player.visible_polygons), 1))
        
        # specify whether the player belongs to the home team
        key = 'home_team' if self.player.home else 'away_team'
        opposite = 'away_team' if self.player.home else 'home_team'
        
        # iterate through the tracking data
        print('[*] Generating the feature vectors for the player\'s safe progressive passing lanes...')
        index = 0        
        for j in tqdm(range(len(self.game.game_tracks))):
            if index == len(self.player.visible_polygons):
                break
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home_team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away_team')):
                    continue
                
            track = self.game.game_tracks[j]
            # get the information of all players at an instance
            players = track[key]
            opponents = track[opposite]
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['jersey_number'] == self.player.jersey_number, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                # find the players visible polygon
                visible_polygon = self.player.visible_polygons[index]
                visible_polygon = Polygon(visible_polygon)
                
                # find the player's coordinates at this frame
                target_player_coords = Point(*target_player[0]['position'])
                # iterate through all the teammates
                num_safe_passing_lanes = 0
                for player in players:
                    if player['jersey_number'] != target_player[0]['jersey_number']:  
                        coords = Point(*player['position'])
                        # check if the player's coordinate lies inside the visible polygon
                        # progressive passing lane analysis
                        if visible_polygon.contains(coords):
                            # find the connecting line between the target player and the teamamte
                            line = LineString([target_player_coords, coords])
                            safe_pass = True
                            # iterate through all the opponents
                            for opponent in opponents:
                                opponent_coords = Point(*opponent['position'])
                                if visible_polygon.contains(opponent_coords):
                                    # compute the distance between the opponent and the passing lane
                                    distance = line.distance(opponent_coords)
                                    if distance < self.threshold:
                                        safe_pass = False
                                        break
                                    # check if the distance is less than a threshold, the passing is
                                    # not safe. Otherwise the passing is safe
                            if safe_pass:
                                num_safe_passing_lanes += 1
                                
                self.features[index] = num_safe_passing_lanes                    
                # update the feature vector
                index += 1
    
    def translate(self) -> List:
        """
        This function translates the numerical features representing the number of safe 
        passsing lanes inside the player's visible polygon; for example, if the number of
        safe passing lanes is 4, it will be translated to (safe progressive passing lanes 4).
        """
        propositions = []
        
        for feature in self.features:
            safe_passes = feature[0]
            translation = f'({self.thresh} risk safe progressive passing lanes {int(safe_passes)})'
            propositions.append(translation)
        
        return propositions


class BackwardPassingLane(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker, threshold: str) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.features = None
        
        self.thresh = threshold
        
        if threshold == 'low':
            self.threshold = 3
        elif threshold == 'medium':
            self.threshold = 5
        elif threshold == 'high':
            self.threshold = 10
        else:
            raise Exception('[!] Invalid threshold; expected "low", "medium", "high".')
        
    def generate(self, on_possession=False) -> None:
        """
        This function uses the player trajectory, ball trajectory, the jersey number of the target player,
        whether the player belongs to the home team, and a threshold value. The function loads the visible
        polygon of the data for each frame. Outside each visible polygon, the passing lanes from the player
        to each teammate outside the visible polygon is calculated. If there is an opponent player whose distance
        from the passing lane is less than the specified threshold, the passing lane is considered unsafe.
        Otherwise the passing lane is considered safe. The function counts the safe passing lanes outside 
        the visible polygon of the player.
        """
        # teammate density vector is a trace with as many elements as the player trajectory; each
        # element is an 11 dimensional vector representing the number of teammates inside the player's
        # visible polygon
        if self.player.player_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.player.track()
        
        if len(self.player.visible_polygons) == 0:
            print('[!] Found no information about the visible polygons of the player. Filling the information...')
            self.player.get_visible_polygon()
            
            
        self.features = np.zeros((len(self.player.visible_polygons), 1))
        
        # specify whether the player belongs to the home team
        key = 'home_team' if self.player.home else 'away_team'
        opposite = 'away_team' if self.player.home else 'home_team'
        
        # iterate through the tracking data
        print('[*] Generating the feature vectors for the player\'s safe backward passing lanes...')
        index = 0        
        for j in tqdm(range(len(self.game.game_tracks))):
            if index == len(self.player.visible_polygons):
                break
            
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home_team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away_team')):
                    continue
            
            track = self.game.game_tracks[j]
            # get the information of all players at an instance
            players = track[key]
            opponents = track[opposite]
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['jersey_number'] == self.player.jersey_number, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                # find the players visible polygon
                visible_polygon = self.player.visible_polygons[index]
                visible_polygon = Polygon(visible_polygon)
                
                # find the player's coordinates at this frame
                target_player_coords = Point(*target_player[0]['position'])
                # iterate through all the teammates
                num_safe_passing_lanes = 0
                for player in players:
                    if player['jersey_number'] != target_player[0]['jersey_number']:  
                        coords = Point(*player['position'])
                        # check if the player's coordinate lies inside the visible polygon
                        # progressive passing lane analysis
                        if not visible_polygon.contains(coords):
                            # find the connecting line between the target player and the teamamte
                            line = LineString([target_player_coords, coords])
                            safe_pass = True
                            # iterate through all the opponents
                            for opponent in opponents:
                                opponent_coords = Point(*opponent['position'])
                                if not visible_polygon.contains(opponent_coords):
                                    # compute the distance between the opponent and the passing lane
                                    distance = line.distance(opponent_coords)
                                    if distance < self.threshold:
                                        safe_pass = False
                                        break
                                    # check if the distance is less than a threshold, the passing is
                                    # not safe. Otherwise the passing is safe
                            if safe_pass:
                                num_safe_passing_lanes += 1
                                
                self.features[index] = num_safe_passing_lanes                    
                # update the feature vector
                index += 1
    
    def translate(self) -> List:
        """
        This function translates the numerical features representing the number of safe 
        passsing lanes outside the player's visible polygon; for example, if the number of
        safe passing lanes is 4, it will be translated to (safe progressive passing lanes 4).
        """
        propositions = []
        
        for feature in self.features:
            safe_passes = feature[0]
            translation = f'(safe backward passing lanes {int(safe_passes)})'
            propositions.append(translation)
        
        return propositions
 

class OpponentPressureFeature(BaseFeature):
    """
    This feature determines at each instance of the game, the opponent pressure.
    Here the opponent pressure is defined as the distance between the player and
    the closest opponent player. If the opponent distance is less than 5 the
    pressure is set to high. If the opponent distance is between 5 and 10 the
    pressure is set to medium and if the distance is more than 10 the pressure
    is set to low.
    """
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.features = None
        
    def generate(self, on_possession=False) -> None:
        if self.all_tracker.ball_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.all_tracker.track()
        
        self.features = np.zeros((len(self.player.player_trajectory), 1))
        key = 'home_team' if self.player.home else 'away_team'
        
        print('[*] Generating the feature vectors for the player\'s opponent pressure...')
        index = 0
        for j in tqdm(range(len(self.game.game_tracks))):
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home_team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away_team')):
                    continue
                
            track = self.game.game_tracks[j]
            
            # get the information of all players at an instance
            players = track[key]
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['jersey_number'] == self.player.jersey_number, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                if self.player.home:       
                    coordinates = self.all_tracker.away_coordinates[index]
                    coordinates = np.array(list(map(lambda x: x['position'], coordinates)))
                else:
                    coordinates = self.all_tracker.home_coordinates[index]
                    coordinates = np.array(list(map(lambda x: x['position'], coordinates)))
                
                distances = np.sqrt(np.sum((coordinates - np.array(target_player[0]['position']).reshape(1, 2).repeat(np.array(coordinates).shape[0], axis=0)) ** 2, axis=1))
                min_dist = np.min(distances)
                
                if 0 <= min_dist < 5:
                    self.features[index, 0] = 1
                elif 5 <= min_dist < 10:
                    self.features[index, 0] = 2
                elif 10 <= min_dist:
                    self.features[index, 0] = 3
                    
                index += 1

        print(self.features.sum())
        
    def translate(self) -> List:
        propositions = []
        
        for feature in self.features.flatten():
            if feature == 1:
                translation = f'(has high opponent pressure)'
            elif feature == 2:
                translation = f'(has medium opponent pressure)'
            elif feature == 3:
                translation = f'(has low opponent pressure)'
            else:
                translation = f'(has medium opponent pressure)'
                
            propositions.append(translation)
        
        return propositions


if __name__ == '__main__':
    game = GameLoader('all_Games', 'AIK__BK Häcken.1')
    print(game.get_team_players())
    player = PlayerTracker(game=game, jersey_number=4, home=True)
    
    all_tracker = AllTracker(game=game)
    all_tracker.track()
    
    feat1 = PlayerLocFeature(game=game, player=player, all_tracker=all_tracker)
    feat1.generate()
    props = feat1.translate()
    print(len(props))
    
    feat2 = TeammateDensityFeature(game=game, player=player, all_tracker=all_tracker)
    feat2.generate()
    props = feat2.translate()
    print(len(props))
    
    feat3 = OpponentsDensityFeature(game=game, player=player, all_tracker=all_tracker)
    feat3.generate()
    props = feat3.translate()
    print(len(props))
    
    feat4 = HasBallFeature(game=game, player=player, all_tracker=all_tracker)
    feat4.generate()
    props = feat4.translate()
    print(len(props))
    
    feat5 = CanPassToFeature(game=game, player=player, all_tracker=all_tracker)
    feat5.generate()
    props = feat5.translate()
    print(len(props))
    
    feat6 = ProgressivePassingLane(game=game, player=player, all_tracker=all_tracker, threshold='medium')
    feat6.generate()
    props = feat6.translate()
    print(len(props))
    
    feat7 = BackwardPassingLane(game=game, player=player, all_tracker=all_tracker, threshold='medium')
    feat7.generate()
    props = feat7.translate()
    print(len(props))
    
    feat8 = OpponentPressureFeature(game=game, player=player, all_tracker=all_tracker)
    feat8.generate()
    props = feat8.translate()
    print(len(props))
