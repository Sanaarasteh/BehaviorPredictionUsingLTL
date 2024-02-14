import numpy as np
import mplsoccer as mpl

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString

from sc_utils import GameLoader
from sc_utils import PlayerTracker
from sc_utils import AllTracker


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
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker, grid_size: Tuple=(4, 3)) -> None:
        """
        :param game (GameLoader): an instance of the game containing the information of
                                  pitch and the players.
        :param grid_size (Tuple) - Default (4, 3): the size of the pitch grid.
        """
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.grid_size = grid_size
        self.features = None
        
        self.left_to_right_positions = {
            0: 'left back',
            1: 'left defense midfielder',
            2: 'left attack midfielder',
            3: 'left forward',
            4: 'central back',
            5: 'central defense midfielder',
            6: 'central attack midfielder',
            7: 'central forward', 
            8: 'right back',
            9: 'right defense midfielder',
            10: 'right attack midfielder',
            11: 'right forward',
        }
        
        self.right_to_left_positions = {
            0: 'right forward',
            1: 'right attack midfielder',
            2: 'right defense midfielder',
            3: 'right back',
            4: 'central forward',
            5: 'central attack midfielder',
            6: 'central defense midfielder',
            7: 'central back', 
            8: 'left forward',
            9: 'left attack midfielder',
            10: 'left defense midfielder',
            11: 'left back'
        }
        
        
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
                if (self.player.home and self.player.team_possession_trajectory[index] == 'home team') or (not self.player.home and self.player.team_possession_trajectory[index] == 'away team'):
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
        for feature, instance in zip(self.features, self.player.present_instances):
            # finding the location of the player
            location = self.__cordinate_to_position(feature, instance)
            
            # translating the location
            translation = f'(at player location {location})'
            
            # updating the propositions
            propositions.append(translation)
        
        return propositions
    
    def __cordinate_to_position(self, feature_vector, instance):
        team_side = self.game.get_team_side(instance, self.player.home)
        
        location = np.argmax(feature_vector)
        
        if team_side == 'left_to_right':
            return self.left_to_right_positions[location]
        else:
            return self.right_to_left_positions[location]

 
class PlayerLoc2Feature(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker) -> None:
        """
        :param game (GameLoader): an instance of the game containing the information of
                                  pitch and the players.
        :param grid_size (Tuple) - Default (4, 3): the size of the pitch grid.
        """
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
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
        self.features = np.zeros((len(self.player.player_trajectory), 2))
        
        # find the location of the player on the grid
        instance = 0
        for index in tqdm(range(len(self.player.player_trajectory))):
            if on_possession:
                if (self.player.home and self.player.team_possession_trajectory[index] == 'home team') or (not self.player.home and self.player.team_possession_trajectory[index] == 'away team'):
                    position = self.player.player_trajectory[index]
                    if position[0] < 0:
                        left = 1
                    else:
                        left = 0
                    
                    if -52.5 <= position[0] <= -36 and -20 <= position[1] <= 20 and self.game.get_team_side(instance, self.player.home) == 'left_to_right':
                        box = 1
                    elif 36 <= position[0] <= 52.5 and -20 <= position[1] <= 20 and self.game.get_team_side(instance, self.player.home) == 'right_to_left':
                        box = 1
                    else:
                        box = 0
                    # convert the location to the one-hot vector (or the bit vector)
                    self.features[index] = np.array([left, box])
            else:
                position = self.player.player_trajectory[index]
                if position[0] < 0:
                    left = 1
                else:
                    left = 0
                
                if -52.5 <= position[0] <= -36 and -20 <= position[1] <= 20 and self.game.get_team_side(instance, self.player.home) == 'left_to_right':
                    box = 1
                elif 36 <= position[0] <= 52.5 and -20 <= position[1] <= 20 and self.game.get_team_side(instance, self.player.home) == 'right_to_left':
                    box = 1
                else:
                    box = 0
                # convert the location to the one-hot vector (or the bit vector)
                self.features[index] = np.array([left, box])

            instance += 1
            
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
            left = 'left' if feature[0] == 1 else 'right'
            box = 'in box' if feature[1] == 1 else 'out of box'
            # translating the location
            translation = f'(is player at {left} side of pitch and {box})'
            
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
                
            # get the information of all players at an instance
            players = self.game.get_team_instance_info(j, self.player.home)
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, players))
            
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
                    if player['trackable_object'] != target_player[0]['trackable_object']:  
                        coords = Point(player['x'], player['y'])
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
            
            # get the information of all players at an instance
            home_players = self.game.get_team_instance_info(j, True)
            away_players = self.game.get_team_instance_info(j, False)
            # TODO: check if the player belongs to the home team
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, home_players))
            
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
                    if player['trackable_object'] != target_player[0]['trackable_object']:  
                        coords = Point(player['x'], player['y'])
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
        
        print('[*] Generating the feature vectors for the player\'s ball possession...')
        index = 0
        for j in tqdm(range(len(self.game.game_tracks))):
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away team')):
                    continue
            
            # get the information of all players at an instance
            players = self.game.get_team_instance_info(j, self.player.home)
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:          
                belongs_to_home, trackable_obj = self.all_tracker.get_ball_possessor(j)
                
                if belongs_to_home == self.player.home and trackable_obj == self.player.track_id:
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
        
        print('[*] Generating the feature vectors for the player\'s number of reachable teammates...')
        index = 0
        for j in tqdm(range(len(self.game.game_tracks))):
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away team')):
                    continue
            
            # get the information of all players at an instance
            players = self.game.get_team_instance_info(j, self.player.home)
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:          
                belongs_to_home, trackable_obj = self.all_tracker.get_ball_possessor(j)
                
                # if the player possesses the ball
                if belongs_to_home == self.player.home and trackable_obj == self.player.track_id:
                    # compute the number of teammates reachable to the player
                    self.features[index, 0] = self.all_tracker.get_reachable_teammates(j, self.player.first_name, self.player.last_name, self.player.home)
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
        
        # iterate through the tracking data
        print('[*] Generating the feature vectors for the player\'s safe progressive passing lanes...')
        index = 0        
        for j in tqdm(range(len(self.game.game_tracks))):
            if index == len(self.player.visible_polygons):
                break
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away team')):
                    continue
            
            # get the information of all players at an instance
            players = self.game.get_team_instance_info(j, self.player.home)
            opponents = self.game.get_team_instance_info(j, not self.player.home)
            
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                # find the players visible polygon
                visible_polygon = self.player.visible_polygons[index]
                visible_polygon = Polygon(visible_polygon)
                
                # find the player's coordinates at this frame
                target_player_coords = Point(target_player[0]['x'], target_player[0]['y'])
                # iterate through all the teammates
                num_safe_passing_lanes = 0
                for player in players:
                    if player['trackable_object'] != target_player[0]['trackable_object']:  
                        coords = Point(player['x'], player['y'])
                        # check if the player's coordinate lies inside the visible polygon
                        # progressive passing lane analysis
                        if visible_polygon.contains(coords):
                            # find the connecting line between the target player and the teamamte
                            line = LineString([target_player_coords, coords])
                            safe_pass = True
                            # iterate through all the opponents
                            for opponent in opponents:
                                opponent_coords = Point(opponent['x'], opponent['y'])
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
            translation = f'({self.thresh} risk progressive passing lanes {int(safe_passes)})'
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
        
        
        # iterate through the tracking data
        print('[*] Generating the feature vectors for the player\'s safe backward passing lanes...')
        index = 0        
        for j in tqdm(range(len(self.game.game_tracks))):
            if index == len(self.player.visible_polygons):
                break
            
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away team')):
                    continue
            
            
            # get the information of all players at an instance
            players = self.game.get_team_instance_info(j, self.player.home)
            opponents = self.game.get_team_instance_info(j, not self.player.home)
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                # find the players visible polygon
                visible_polygon = self.player.visible_polygons[index]
                visible_polygon = Polygon(visible_polygon)
                
                # find the player's coordinates at this frame
                target_player_coords = Point(target_player[0]['x'], target_player[0]['y'])
                # iterate through all the teammates
                num_safe_passing_lanes = 0
                for player in players:
                    if player['trackable_object'] != target_player[0]['trackable_object']:  
                        coords = Point(player['x'], player['y'])
                        # check if the player's coordinate lies inside the visible polygon
                        # progressive passing lane analysis
                        if not visible_polygon.contains(coords):
                            # find the connecting line between the target player and the teamamte
                            line = LineString([target_player_coords, coords])
                            safe_pass = True
                            # iterate through all the opponents
                            for opponent in opponents:
                                opponent_coords = Point(opponent['x'], opponent['y'])
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
            translation = f'({self.thresh} risk backward passing lanes {int(safe_passes)})'
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
        
        print('[*] Generating the feature vectors for the player\'s opponent pressure...')
        index = 0
        for j in tqdm(range(len(self.game.game_tracks))):
            if on_possession:
                if not ((self.player.home and self.player.team_possession_trajectory[index] == 'home_team') or 
                        (not self.player.home and self.player.team_possession_trajectory[index] == 'away_team')):
                    continue
            
            # get the information of all players at an instance
            players = self.game.get_team_instance_info(j, self.player.home)
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                if self.player.home:       
                    coordinates = self.all_tracker.away_coordinates[j]
                    coordinates = np.array(list(map(lambda x: [x['x'], x['y']], coordinates)))
                else:
                    coordinates = self.all_tracker.home_coordinates[j]
                    coordinates = np.array(list(map(lambda x: [x['x'], x['y']], coordinates)))
                
                distances = np.sqrt(np.sum((coordinates - np.array([target_player[0]['x'], target_player[0]['y']]).reshape(1, 2).repeat(np.array(coordinates).shape[0], axis=0)) ** 2, axis=1))
                min_dist = np.min(distances)
                
                if 0 <= min_dist < 5:
                    self.features[index, 0] = 1
                elif 5 <= min_dist < 10:
                    self.features[index, 0] = 2
                elif 10 <= min_dist:
                    self.features[index, 0] = 3
                    
                index += 1
        
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


class VelocityFeature(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker, interval: int=10) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.interval = interval
        self.features = None
        
    def generate(self, on_possession=False) -> None:
        if self.all_tracker.ball_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.all_tracker.track()
        
        self.features = []
        
        print('[*] Generating the feature for player speed...')
        for i in tqdm(range(self.interval, len(self.game.game_tracks))): 
            continuous = True
            
            for j in range(i - self.interval, i):
                # get the information of all players at an instance
                players = self.game.get_team_instance_info(j, self.player.home)
                # check whether the target player has any record in this frame
                target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, players))
                
                # if the player's record exists in this frame...
                if len(target_player) == 0:          
                    continuous = False
                    break
                
                if j == i - self.interval:
                    start_pos = np.array([target_player[0]['x'], target_player[0]['y']])     
            
            if continuous:
                end_pos = np.array([target_player[0]['x'], target_player[0]['y']])
                
                # computing velocity in terms of meter/second
                velocity = np.sqrt(((end_pos - start_pos) ** 2).sum()) / (self.interval * 0.1)
                
                self.features.append(velocity)
                   
    def translate(self) -> List:
        propositions = []
        
        for feature in self.features:
            if feature <= 1.8:
                velocity = 'low'
            elif 1.8 < feature <= 3.6:
                velocity = 'medium-low'
            elif 3.6 < feature <= 5.5:
                velocity = 'medium-high'
            else:
                velocity = 'high'
                
            translation = f'(has {velocity} speed in previous {self.interval * 0.1} seconds)'
            propositions.append(translation)
            
        return propositions


class ExpansionContractionFeature(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker, interval: int=10, k: int=3) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.interval = interval
        self.num_neighbors = k
        
        self.features = None
        
    def generate(self, on_possession=False) -> None:
        if self.all_tracker.ball_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.all_tracker.track()
        
        self.features = []
        
        print('[*] Generating the feature for closest teammates expansion/contraction...')
        for i in tqdm(range(self.interval, len(self.game.game_tracks))):
            continuous = True
            
            for j in range(i - self.interval, i):
                # get the information of all players at an instance
                players = self.game.get_team_instance_info(j, self.player.home)
                # check whether the target player has any record in this frame
                target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, players))
                
                # if the player's record exists in this frame...
                if len(target_player) == 0:          
                    continuous = False
                    break
                
                if j == i - self.interval:
                    teammates = list(filter(lambda x: x['trackable_object'] != self.player.track_id, players))
                    teammates_locations = np.array(list(map(lambda x: [x['x'], x['y']], teammates)))
                    player_location = np.array([target_player[0]['x'], target_player[0]['y']]).reshape(1, 2).repeat(len(teammates_locations), axis=0)
                    
                    player_teammate_distances = np.sqrt(((teammates_locations - player_location) ** 2).sum(axis=1))
                    
                    initial_avg_distance = np.sort(player_teammate_distances)[:self.num_neighbors].mean()
            
            if continuous:
                teammates = list(filter(lambda x: x['trackable_object'] != self.player.track_id, players))
                teammates_locations = np.array(list(map(lambda x: [x['x'], x['y']], teammates)))
                player_location = np.array([target_player[0]['x'], target_player[0]['y']]).reshape(1, 2).repeat(len(teammates_locations), axis=0)
                
                player_teammate_distances = np.sqrt(((teammates_locations - player_location) ** 2).sum(axis=1))
                
                final_avg_distance = np.sort(player_teammate_distances)[:self.num_neighbors].mean()
                
                if final_avg_distance > initial_avg_distance:
                    self.features.append(1)
                else:
                    self.features.append(0)
      
    def translate(self) -> List:
        propositions = []
        
        for feature in self.features:
            if feature == 1:
                translation = f'({self.num_neighbors}-closest teammates are expanding)'
            else:
                translation = f'({self.num_neighbors}-closest teammates are contracting)'
                   
            propositions.append(translation)
            
        return propositions


class PositionSpectralFeature(BaseFeature):
    def __init__(self, game: GameLoader, player: PlayerTracker, all_tracker: AllTracker, interval: int=10) -> None:
        super().__init__()
        
        self.game = game
        self.player = player
        self.all_tracker = all_tracker
        self.interval = interval
        
        self.features = None
        
    def generate(self, on_possession=False) -> None:
        if self.all_tracker.ball_trajectory is None:
            print('[!] Found no information about the player trajectory. Filling the information...')
            self.all_tracker.track()
        
        self.features = []
        self.sides = []
        
        print('[*] Generating the feature for the player (spectral) dominant direction...')
        for i in tqdm(range(self.interval, len(self.game.game_tracks))):
            continuous = True
            player_locations = []
            for j in range(i - self.interval, i):
                # get the information of all players at an instance
                players = self.game.get_team_instance_info(j, self.player.home)
                # check whether the target player has any record in this frame
                target_player = list(filter(lambda x: x['trackable_object'] == self.player.track_id, players))
                
                # if the player's record exists in this frame...
                if len(target_player) == 0:          
                    continuous = False
                    break
                
                player_location = [target_player[0]['x'], target_player[0]['y']]
                player_locations.append(player_location)
                
            
            if continuous:
                _, angle = self.compute_dominant_direction(np.array(player_locations))
                attacking_direction = self.game.get_team_side(i, self.player.home)
                
                if attacking_direction == 'left_to_right':
                    self.sides.append(1)
                    if angle <= np.pi / 8 or angle >= 15 * np.pi / 8:
                        self.features.append(1) # offensive
                    elif np.pi / 8 < angle <= 3 * np.pi / 8:
                        self.features.append(2) # progressive
                    elif 3 * np.pi / 8 < angle <= 5 * np.pi / 8:
                        self.features.append(3) # neutral
                    elif 5 * np.pi / 8 < angle <= 7 * np.pi / 8:
                        self.features.append(4) # retractive
                    elif 7 * np.pi / 8 < angle <= 9 * np.pi / 8:
                        self.features.append(5) # defensive
                    elif 9 * np.pi / 8 < angle <= 11 * np.pi / 8:
                        self.features.append(6) # retractive
                    elif 11 * np.pi / 8 < angle <= 13 * np.pi / 8:
                        self.features.append(7) # neutral
                    elif 13 * np.pi / 8 < angle < 15 * np.pi / 8:
                        self.features.append(8) # progressive
                else:
                    self.sides.append(2)
                    if angle <= np.pi / 8 or angle >= 15 * np.pi / 8:
                        self.features.append(5) # defensive
                    elif np.pi / 8 < angle <= 3 * np.pi / 8:
                        self.features.append(4) # progressive
                    elif 3 * np.pi / 8 < angle <= 5 * np.pi / 8:
                        self.features.append(3) # neutral
                    elif 5 * np.pi / 8 < angle <= 7 * np.pi / 8:
                        self.features.append(2) # progressive
                    elif 7 * np.pi / 8 < angle <= 9 * np.pi / 8:
                        self.features.append(1) # offensive
                    elif 9 * np.pi / 8 < angle <= 11 * np.pi / 8:
                        self.features.append(8) # retractive
                    elif 11 * np.pi / 8 < angle <= 13 * np.pi / 8:
                        self.features.append(7) # neutral
                    elif 13 * np.pi / 8 < angle < 15 * np.pi / 8:
                        self.features.append(6) # progressive
    
    def translate(self) -> List:
        propositions = []
        
        for side, feature in zip(self.features, self.sides):
            if side == 1:
                if feature == 1:
                    propositions.append(f'(is player at offensive direction)')
                elif feature == 2 or feature == 8:
                    propositions.append(f'(is player at progressive direction)')
                elif feature == 3 or feature == 7:
                    propositions.append(f'(is player at neutral direction)')
                elif feature == 4 or feature == 6:
                    propositions.append(f'(is player at retractive direction)')
                elif feature == 5:
                    propositions.append(f'(is player at defensive direction)')
            else:
                if feature == 1:
                    propositions.append(f'(is player at defensive direction)')
                elif feature == 2 or feature == 8:
                    propositions.append(f'(is player at retractive direction)')
                elif feature == 3 or feature == 7:
                    propositions.append(f'(is player at neutral direction)')
                elif feature == 4 or feature == 6:
                    propositions.append(f'(is player at progressive direction)')
                elif feature == 5:
                    propositions.append(f'(is player at offensive direction)')
        
        return propositions
    
    def compute_dominant_direction(self, coords):
        # 1 - compute the covariance matrix of the coordinates
        covariance = np.cov(coords.T)
        
        if covariance.shape != (2, 2):
            covariance = np.cov(coords)

        # 2 - compute the eigenvalues and eigenvectors of the covariance
        eig_vals, eig_vecs = np.linalg.eig(covariance)
        
        # multiplying an eigenvector with its corresponding eigenvalue makes the method
        # incorporate the strength of the component; otherwise, the vectors are normalized
        dominant_direction = np.max(eig_vals) * eig_vecs[:, np.argmax(eig_vals)]
        
        angle = np.arctan2(dominant_direction[1], dominant_direction[0])
        
        return dominant_direction, angle
        
        

if __name__ == '__main__':
    game = GameLoader('data/matches', '852654-Manchester City-Chelsea')
    first_name = 'Kevin'
    last_name = 'de Bruyne'
    
    player = PlayerTracker(game=game, first_name=first_name, last_name=last_name, home=True)
    player.track()
    
    all_tracker = AllTracker(game=game)
    all_tracker.track()
    
    feat1 = PlayerLocFeature(game=game, player=player, all_tracker=all_tracker)
    feat1.generate()
    props = feat1.translate()
    print(props[1000], len(props))
    
    feat2 = PlayerLoc2Feature(game=game, player=player, all_tracker=all_tracker)
    feat2.generate()
    props = feat2.translate()
    print(props[1000], len(props))
    
    feat3 = TeammateDensityFeature(game=game, player=player, all_tracker=all_tracker)
    feat3.generate()
    props = feat3.translate()
    print(props[1000], len(props))
    
    feat4 = OpponentsDensityFeature(game=game, player=player, all_tracker=all_tracker)
    feat4.generate()
    props = feat4.translate()
    print(props[1000], len(props))
    
    feat5 = HasBallFeature(game=game, player=player, all_tracker=all_tracker)
    feat5.generate()
    props = feat5.translate()
    print(props[1000], len(props))
    
    feat6 = CanPassToFeature(game=game, player=player, all_tracker=all_tracker)
    feat6.generate()
    props = feat6.translate()
    print(props[1000], len(props))
    
    feat7 = ProgressivePassingLane(game=game, player=player, all_tracker=all_tracker, threshold='low')
    feat7.generate()
    props = feat7.translate()
    print(props[1000], len(props))
    
    feat8 = BackwardPassingLane(game=game, player=player, all_tracker=all_tracker, threshold='medium')
    feat8.generate()
    props = feat8.translate()
    print(props[1000], len(props))
    
    feat9 = OpponentPressureFeature(game=game, player=player, all_tracker=all_tracker)
    feat9.generate()
    props = feat9.translate()
    print(props[1000], len(props))
    
    feat10 = VelocityFeature(game=game, player=player, all_tracker=all_tracker, interval=20)
    feat10.generate()
    props = feat10.translate()
    print(props[1000], len(props))
    
    feat11 = ExpansionContractionFeature(game=game, player=player, all_tracker=all_tracker, interval=20, k=3)
    feat11.generate()
    props = feat11.translate()
    print(props[1200], len(props))
    
    feat12 = PositionSpectralFeature(game=game, player=player, all_tracker=all_tracker, interval=200)
    feat12.generate()
    props = feat12.translate()
    print(props[1200], len(props))
