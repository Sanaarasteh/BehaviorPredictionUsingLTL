import os
import copy
import json
import torch
import time
import ltl_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplsoccer as mpl
import networkx as nx

from pprint import pprint
from datetime import datetime
from tqdm import tqdm
from datetime import datetime
from typing import List
from typing import Dict
from typing import Tuple
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree
from scipy.spatial import voronoi_plot_2d
from scipy.spatial import convex_hull_plot_2d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


LEFT_UP_CORNER = (-52.50, 34)
LEFT_BOTTOM_CORNER = (-52.50, -34)
RIGHT_UP_CORNER = (52.50, 34)
RIGHT_BOTTOM_CORNER = (52.50, -34)


class GameLoader:
    """
    This class receives the path for the game data files and the name of a particular game
    and loads the relevant information about the game.
    """    
    def __init__(self, base_path, game_name, name=None):
        with open(f'{base_path}/{game_name}.json', 'r') as handle:
            self.game_tracks = json.load(handle)
        
        with open(f'{base_path}/{game_name}-match-info.json', 'r') as handle:
            self.game_info = json.load(handle)
        
        self.ball_id = self.game_info['ball']['trackable_object']
        self.home_id = self.game_info['home_team']['id']
        self.away_id = self.game_info['away_team']['id']
        self.home_players = self.get_team_players(home=True)
        self.away_players = self.get_team_players(home=False)
        self.home_trackable_objects = list(map(lambda x: x['trackable_object'], self.home_players))
        self.away_trackable_objects = list(map(lambda x: x['trackable_object'], self.away_players))
        
        self.__interpolate_ball_coordinates()
        self.__pass_detection()
        # exit()
        # self.__interpolate_pitch_state()
        
        if name is not None:
            player_track_object = self.get_trackable_object(name[0], name[1])
            self.__interpolate_player_position(player_track_object, threshold=50, mode='linear')
        
        self.__interpolate_ball_possessor()
    
    
    def get_team_players(self, home :bool=True) -> List[Dict]:
        """
        This function returns the name of the players of a team with their jersey numbers.
        
        :param home (bool) - Default True: returns the players of the Home team if True,
                                        returns the players of the Away team otherwise
        """
        # specify the team for which we want to extract the team members
        key = 'home_team' if home else 'away_team'
        
        self.team_id = self.game_info[key]['id']
        
        team_players = [item for item in self.game_info['players'] if item['team_id'] == self.team_id]
        
        return team_players
    
    def get_pitch_size(self) -> List:
        """
        This function extracts the pitch size using the info file.
        """
        
        length, width = self.game_info['pitch_length'], self.game_info['pitch_width']
            
        return [length, width]
    
    def get_ball_coords(self, instance_no: int) -> List:
        for elem in self.game_tracks[instance_no]['data']:
            if elem['trackable_object'] == self.ball_id:
                return [elem['x'], elem['y'], elem['z']]
    
    def get_player_coords(self, instance_no: int, trackable_object: int) -> List:
        """
        This method receives an instance number and a trackable object corresponding
        to a player and returns the player's coordinates if available
        """
        
        exists, _ = self.player_exists(instance_no, trackable_object)
        if exists:
            home_info = self.get_team_instance_info(instance_no, True)
            
            for player in home_info:
                if player['trackable_object'] == trackable_object:
                    return [player['x'], player['y']]
            
            away_info = self.get_team_instance_info(instance_no, False)
            
            for player in away_info:
                if player['trackable_object'] == trackable_object:
                    return [player['x'], player['y']]
        else:
            return None
    
    def get_team_instance_info(self, instance_no: int, home: bool=True) -> List[Dict]:
        """
        This method receives an instance number and an indicator specifying whether the
        information of the homw team should be retreived or the away team. The method then
        collects the team player information at the specified instance number.
        
        :param instance_no (int): specified the frame/instance number
        :param home (bool) - default True: specifies the team whose information should be
                                           retrieved.
        """
        
        track = self.game_tracks[instance_no]['data']
        
        info = []
        
        for item in track:
            if home:
                if item['trackable_object'] in self.home_trackable_objects:
                    info.append(item)
            else:
                if item['trackable_object'] in self.away_trackable_objects:
                    info.append(item)
        
        return info
    
    def get_trackable_object(self, first_name: str, last_name: str) -> int:
        """
        This method receives the first name and last name of a player and returns his/her
        trackable_object attribute
        
        :param first_name (str): the player's first name
        :param last_name (str): the player's last name
        """
        
        for player in self.home_players:
            if player['first_name'] == first_name and player['last_name'] == last_name:
                return player['trackable_object']
        
        for player in self.away_players:
            if player['first_name'] == first_name and player['last_name'] == last_name:
                return player['trackable_object']
        
        raise Exception('[!] No player with such first name and last name exists!')
    
    def get_team_from_trackable_object(self, trackable_object: int) -> str:
        """
        This method receives a trackable object and returns either "home_team" if the 
        corresponding player belongs to the home team or "away_team" if the corresponding
        player belongs to the away team.
        
        :param trackable_object (int): the trackable object of a player
        """
        
        for player in self.home_players:
            if player['trackable_object'] == trackable_object:
                return 'home_team'
        
        for player in self.away_players:
            if player['trackable_object'] == trackable_object:
                return 'away_team'
        
        raise Exception('[!] No player with such a trackable object exists!')
    
    def assign_ball_based_on_distance(self, instance_no: int, prior: str=None):
        """
        This function receives an instance of the tracking information and assigns the ball
        to the player who is the closest to the ball
        """
        assert prior == 'home team' or prior == 'away team' or prior is None
        
        # finding the ball coordinates
        ball_coordinates = self.get_ball_coords(instance_no)[:2]
        
        # getting the players info in this particular instance
        home_team_info = self.get_team_instance_info(instance_no, home=True)
        away_team_info = self.get_team_instance_info(instance_no, home=False)
        
        teams_info = home_team_info.copy()
        teams_info.extend(away_team_info)
        
        # find the closest home team player to the ball
        all_players_info = [[player['x'], player['y']] for player in home_team_info]
        all_players_info.extend([[player['x'], player['y']] for player in away_team_info])
        all_players_info = np.array(all_players_info)
        
        ball_coordinates = np.array([ball_coordinates] * len(all_players_info))
        
        distances = np.sqrt(np.sum((all_players_info - ball_coordinates) ** 2, axis=1))
        sorted_indices = np.argsort(distances)
        
        target_team = None
        if prior == 'home team':
            target_team = 'home_team'
        elif prior == 'away team':
            target_team = 'away_team'
        
        if target_team:
            for index in sorted_indices:
                if self.get_team_from_trackable_object(teams_info[index]['trackable_object']) == target_team:
                    self.game_tracks[instance_no]['possession']['trackable_object'] = teams_info[index]['trackable_object']
                    break
        else:
            if self.game_tracks[instance_no]['possession']['group'] is None:
                self.game_tracks[instance_no]['possession']['group'] = self.get_team_from_trackable_object(teams_info[sorted_indices[0]]['trackable_object'])
                
            self.game_tracks[instance_no]['possession']['trackable_object'] = teams_info[sorted_indices[0]]['trackable_object']
    
    def is_ball_out_of_pitch(self, instance_no: int):
        """
        At an instance, this method determines whether the ball is inside the pitch or not
        """
        
        ball_coords = self.get_ball_coords(instance_no)
        
        if LEFT_UP_CORNER[0] <= ball_coords[0] <= RIGHT_UP_CORNER[0] and LEFT_BOTTOM_CORNER[1] <= ball_coords[1] <= LEFT_UP_CORNER[1]:
            return False
        
        return True
    
    def get_team_possession_from_trackable_object(self, instance_no):
        """
        In some instances of the game, the ball possessor is known but the team
        who possesses the ball is uknown.
        """
        
        possessor_team = self.game_tracks[instance_no]['possession']['group']
        possessor_player = self.game_tracks[instance_no]['possession']['trackable_object']
        game_started_condition = len(self.get_team_instance_info(instance_no)) > 0
        
        if possessor_team is None and possessor_player and game_started_condition is not None:
            self.game_tracks[instance_no]['possession']['group'] = self.get_team_from_trackable_object(possessor_player)
    
    def get_team_side(self, instance_no: int, home: bool=True) -> str:
        """
        This method determines whether a team is attacking from left to right or
        right to left at a particular instance.
        """
        try:
            period = self.game_tracks[instance_no]['period'] - 1
        except:
            if instance_no / 600 > 45:
                period = 1
            else:
                period = 0
        
        if home:
            return self.game_info['home_team_side'][period]
        else:
            return self.game_info['home_team_side'][1 - period]
    
    def player_exists(self, instance_no: int, trackable_object: int) -> bool:
        """
        This method checks if a player with the given trackable object appears
        in a particular instance of the game.
        
        :param instance_no (int): the instance of the game at which we want to check
                                  the existence of the player
        "param trackable_object (int): the trackable object of the player
        """
        
        track = self.game_tracks[instance_no]
        
        possession = track['possession']['group']
        
        exists = False
        for item in track['data']:
            if item['trackable_object'] == trackable_object:
                exists = True
                break
        
        return exists, possession
    
    def __ball_exists(self, instance_no: int) -> bool:
        """
        This method determines whether the tracking information of the ball exists in a particular instance.
        
        :param instance_no (int): an integer specifying the tracking instance number
        """
        for elem in self.game_tracks[instance_no]['data']:
            if elem['trackable_object'] == self.ball_id:
                return True
        
        return False
    
    def __missing_possession_counter(self):
        count = 0
        for i in range(len(self.game_tracks)):
            possessor_team = self.game_tracks[i]['possession']['group']
            possessor_player = self.game_tracks[i]['possession']['trackable_object']
            game_started_condition = len(self.get_team_instance_info(i)) > 0
            
            if (possessor_team is None or possessor_player is None) and game_started_condition:
                count += 1
                
        return count
    
    def __interpolate_ball_coordinates(self):
        """
        This functions interpolates the ball position for every instance of the game at which
        the ball coordinates is unknown
        """
        # keep track of the index of the element in the ball_trajectory list
        index = 0
        
        # iterate through the sequence of ball coordinates
        print('[*] Linearly interpolating the ball coordinates...')
        while index < len(self.game_tracks):
            # we need to find the number of consecutive frames which have NaN values
            num_consecutive_nones = 0
            
            # check if the ball info exists in this instance
            ball_exists = self.__ball_exists(index)
                
            if not ball_exists:
                # fix the starting index (the last index which is not NaN)
                start_index = index - 1
                
                # swing the index until we get to a non-NaN value and increase 
                # the number of consecutive NaN frames
                end_index = index
                while not self.__ball_exists(end_index):
                    num_consecutive_nones += 1
                    end_index += 1
                    
                # define the steps for linear interpolation
                steps = np.linspace(0, 1., num_consecutive_nones)
                
                # fill the NaN values by interpolating the line connecting the two non-NaN values
                init = np.array(self.get_ball_coords(start_index), dtype=np.float16)
                finit = np.array(self.get_ball_coords(end_index), dtype=np.float16)
                
                for j, step in enumerate(steps):
                    value = np.round((1 - step) * init + step * finit, 2).tolist()
                    self.game_tracks[start_index + j + 1]['data'].append({'track_id': self.ball_id, 
                                                                          'trackable_object': self.ball_id, 
                                                                          'is_visible': False, 
                                                                          'x': value[0], 
                                                                          'y': value[1], 
                                                                          'z': value[2]})
                
                index = end_index
            else:
                index += 1
    
    def __interpolate_pitch_state(self):
        # constructing the network
        neural_net = torch.nn.Sequential(
            torch.nn.Linear(47, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 44)
        )
        
        if os.path.exists('saved_models/pitch_state_interpolator.pt'):
            print('[*] Loading the neural network')
            neural_net.load_state_dict(torch.load('saved_models/pitch_state_interpolator.pt'))
            
            # iterate through the game instances and fill in the gaps with the neural net
            print('[*] Interpolating the pitch state using the neural network...')
            for i in tqdm(range(1, len(self.game_tracks))):
                # check the sufficient condition for state interpolation
                if len(self.game_tracks[i]['data']) <= 1:
                    previous_data = self.game_tracks[i - 1]['data'].copy()
                    if len(previous_data) > 1:
                        previous_positions = []
                        for info in previous_data:
                            previous_positions.append(info['x'])
                            previous_positions.append(info['y'])
                            if 'z' in info.keys():
                                previous_positions.append(info['z'])
                        
                        previous_positions = torch.tensor(previous_positions, dtype=torch.float32).view(1, -1)
                        current_estimate = copy.deepcopy(previous_data)
                        with torch.no_grad():
                            estimate = neural_net(previous_positions).squeeze(0).tolist()
                            
                        for j in range(22):
                            current_estimate[j]['x'] = np.round(estimate[2 * j], decimals=2)
                            current_estimate[j]['y'] = np.round(estimate[2 * j + 1], decimals=2)
                        
                        previous_ball_info = self.game_tracks[i]['data'][-1]
                        self.game_tracks[i]['data'] = current_estimate
                        self.game_tracks[i]['data'][-1] = previous_ball_info
                        
        else:
            print('[!] No existing pitch state interpolator neural net found! Training a new neural network...')
            os.makedirs('saved_models', exist_ok=True)
            self.__train_neural_net(neural_net)
            self.__interpolate_pitch_state()
        
    def __train_neural_net(self, neural_net: torch.nn.Module):
        # preparing the dataset for the neural network
        temp_access = self.game_tracks
        
        class PositionDataset(Dataset):
            def __init__(self):
                positions = []
                timestamps = []
                for instance in temp_access:
                    data = instance['data']
                    
                    # extract the instances for which the positions of the players
                    # are known
                    sample = []
                    if len(data) > 1:
                        for info in data:
                            sample.append(info['x'])
                            sample.append(info['y'])
                            if 'z' in info.keys():
                                sample.append(info['z'])
                    
                        positions.append(sample)
                        timestamps.append(datetime.strptime(instance['timestamp'], '%H:%M:%S.%f'))
                
                self.intput_output = []
                
                for i in range(len(positions) - 1):
                    if (timestamps[i + 1] - timestamps[i]).total_seconds() == 0.1:
                        inp = positions[i]
                        out = positions[i + 1][:44]
                        
                        self.intput_output.append((inp, out))
            
            def __len__(self):
                return len(self.intput_output)
            
            def __getitem__(self, index):
                inp = torch.tensor(self.intput_output[index][0], dtype=torch.float32)
                out = torch.tensor(self.intput_output[index][1], dtype=torch.float32)
                
                sample = {'x': inp, 'y': out}
                
                return sample
        
        whole_dataset = PositionDataset()
        
        train_len = int(0.8 * len(whole_dataset))
        train_data, test_data = random_split(whole_dataset, [train_len, len(whole_dataset) - train_len])
        
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
        
        # defining the optimizer
        optimizer = torch.optim.Adam(neural_net.parameters(), lr=1e-4)
        
        # defining the loss function
        loss_fn = torch.nn.MSELoss()
        
        # training phase
        best_test_loss = float('inf')
        train_losses = []
        test_losses = []
        for epoch in range(200):
            start_time = time.time()
            avg_train_loss = 0.
            
            for sample in train_loader:
                inp, out = sample['x'], sample['y']
                
                optimizer.zero_grad()
                
                output = neural_net(inp)
                
                loss = loss_fn(output, out) + 0.1 * torch.norm(output - inp[:, :44])
                
                loss.backward()
                optimizer.step()
                
                avg_train_loss += loss.item()
            
            avg_train_loss /= len(train_loader)
            train_losses.append(avg_train_loss)
            
            with torch.no_grad():
                avg_test_loss = 0.
                for sample in test_loader:
                    inp, out = sample['x'], sample['y']
                    
                    output = neural_net(inp)
                    
                    loss = loss_fn(output, out)
                    
                    avg_test_loss += loss.item()
                
                avg_test_loss /= len(test_loader)
                end_time = time.time()
                test_losses.append(avg_test_loss)
                elapsed = end_time - start_time
                
                msg = f'[*] Epoch: {epoch:04d} - Avg Train Loss: {avg_train_loss:.3f} - Avg Test Loss: {avg_test_loss:.3f} ({elapsed:.1f}s)'
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    torch.save(neural_net.state_dict(), 'saved_models/pitch_state_interpolator.pt')
                    msg += ' - [CHECKPOINT]'
                
            print(msg)
        
        np.savetxt('train_losses.txt', train_losses)
        np.savetxt('test_losses.txt', test_losses)
        exit()
    
    def __interpolate_ball_possessor(self):
        """
        This function finds the ball possessor for the instances at which the possessor is unknwon.
        Such instances usually happen in continuous sequences. For a found sequence, we look at the
        events information and see whether there is a pass event happend before the sequence. If so,
        we interpolate the ball possessor with the information of the pass receiver. Otherwise, we 
        take the simple approach and fill the information of the ball possessor using the concept of
        Voronoi cells; i.e., the ball is assigned to the closest player.
        """
        print('[*] Interpolating ball possession information')
        initial_missings = self.__missing_possession_counter()
        
        for i in tqdm(range(len(self.game_tracks))):
            # get the information of the possessing team, possessing player, and whether the game has started
            possessor_team = self.game_tracks[i]['possession']['group']
            possessor_player = self.game_tracks[i]['possession']['trackable_object']
            game_started_condition = len(self.get_team_instance_info(i)) > 0
            
            # if the ball is outside the pitch the possession is assigned to neither teams
            if self.is_ball_out_of_pitch(i):
                self.game_tracks[i]['possession']['group'] = 'out'
                self.game_tracks[i]['possession']['trackable_object'] = 'out'
                
            # if team possession is unknown but possessing player is known, fill the team possession accordingly
            self.get_team_possession_from_trackable_object(i)
            
            # if team possession is known but possessing player is unknown, assign the possessing player
            # based on their distance to the ball and with prior information that the possessing player
            # must belong to the possessing team
            if possessor_team is not None and possessor_player is None and game_started_condition:
                self.assign_ball_based_on_distance(i, possessor_team)
        
            # if both team possession and possessing player is unknown, assign the possessing team
            # and the possessing player based on their distance to the ball
            if possessor_team is None and possessor_player is None and game_started_condition:
                self.assign_ball_based_on_distance(i)
        
        finial_missing = self.__missing_possession_counter()
        
        if finial_missing == 0:
            print(f'\__[*] Successfully interpolated {initial_missings}/{len(self.game_tracks)} instances with missing possession information.')
    
    def __interpolate_player_position(self, trackable_object, threshold=50, mode='linear'):
        presence_indicator = []
        
        for i in range(len(self.game_tracks)):
            exists, _ = self.player_exists(i, trackable_object)
            if exists:
                presence_indicator.append(1)
            else:
                presence_indicator.append(0)
        
        i = presence_indicator.index(1)
        last_index = len(presence_indicator) - 1 - presence_indicator[::-1].index(1)
        
        while i < last_index:
            if presence_indicator[i] == 1 and presence_indicator[i + 1] == 0:
                j = i + 2
                count = 1
                while  j < last_index and presence_indicator[j] == presence_indicator[i + 1]:
                    count += 1
                    j += 1
                    
                if count < threshold:
                    player_first = self.get_player_coords(i, trackable_object)
                    player_last = self.get_player_coords(j, trackable_object)
                        
                    avg_x = 0.5 * (player_first[0] + player_last[0])
                    avg_y = 0.5 * (player_first[1] + player_last[1])
                    denom_x = player_last[0] - player_first[0]
                    denom_y = player_last[1] - player_first[1]
                    
                    coeff = 1
                    for ind in range(i + 1, j):
                        if mode == 'linear':
                            if denom_x == 0:
                                new_x = player_last[0]
                            else:
                                new_x = (coeff / denom_x) * (player_last[0] - player_first[0]) + player_first[0]
                            
                            if denom_y == 0:
                                new_y = player_last[1]
                            else:
                                new_y = (coeff / denom_y) * (player_last[1] - player_first[1]) + player_first[1]
                            
                            coeff += 1
                        else:
                            new_x = avg_x
                            new_y = avg_y
                            
                        self.game_tracks[ind]['data'].append({'track_id': trackable_object,
                                                              'trackable_object': trackable_object,
                                                              'is_visible': False,
                                                              'x': new_x,
                                                              'y': new_y})
                    
                i = j
            else:
                i += 1
    
    def __pass_detection(self) -> None:
        """
        This method is supposed to extract the passes throughout the game. The speculation is
        that the successfull pass instances are the ones that the team possession is known but
        the possessing player is unknown. The second conjecture is that the intercepted passes
        are the ones that neither the possessing team nor the possessing player are known.
        """
        if os.path.exists('pass_instances.csv'):
            return
        
        pass_instances = pd.DataFrame(columns=['instance', 'outcome', 'possessing team'])
        for i in tqdm(range(len(self.game_tracks))):
            # get the information of the possessing team, possessing player, and whether the game has started
            possessor_team = self.game_tracks[i]['possession']['group']
            possessor_player = self.game_tracks[i]['possession']['trackable_object']
            game_started_condition = len(self.get_team_instance_info(i)) > 0
            
            # if the ball is outside the pitch the possession is assigned to neither teams
            if self.is_ball_out_of_pitch(i):
                self.game_tracks[i]['possession']['group'] = 'out'
                self.game_tracks[i]['possession']['trackable_object'] = 'out'
                
            # if team possession is unknown but possessing player is known, fill the team possession accordingly
            self.get_team_possession_from_trackable_object(i)
            
            # if team possession is known but possessing player is unknown, assign the possessing player
            # based on their distance to the ball and with prior information that the possessing player
            # must belong to the possessing team
            if not self.is_ball_out_of_pitch(i):
                if possessor_team is not None and possessor_player is None and game_started_condition:
                    pass_instances.loc[len(pass_instances.index)] = [i, 'success', possessor_team]
            
                # if both team possession and possessing player is unknown, assign the possessing team
                # and the possessing player based on their distance to the ball
                if possessor_team is None and possessor_player is None and game_started_condition:
                    pass_instances.loc[len(pass_instances.index)] = [i, 'failed', None]
        
        pass_instances.to_csv('pass_instances.csv', index=False)

        
class PlayerTracker:
    """
    This class tracks the location of a target player along with the ball coordinates
    for every instance that the player happens to exist.
    """
    def __init__(self, game: GameLoader, first_name: str, last_name: str, home: bool=True) -> None:
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
        self.first_name = first_name
        self.last_name = last_name
        self.home = home
        self.player_trajectory = None
        self.ball_trajectory = None
        self.team_possession_trajectory = None
        self.track_id = [player['trackable_object'] for player in self.game.get_team_players() 
                         if player['first_name'] == first_name and player['last_name'] == last_name]
        
        if len(self.track_id) == 0:
            raise Exception('[!] No such player exists!')
        else:
            self.track_id = self.track_id[0]
        
        self.present_instances = []
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
        
        print('[*] Tracking the player and the ball...')
        for index in tqdm(range(len(self.game.game_tracks))):
            track = self.game.game_tracks[index]
            # get the information of all players at an instance
            players = track['data']
            # check whether the target player has any record in this frame
            target_player = list(filter(lambda x: x['trackable_object'] == self.track_id, players))
            
            # if the player's record exists in this frame...
            if len(target_player) > 0:
                # find the player's coordinates at this frame
                coords = [target_player[0]['x'], target_player[0]['y']]
                
                # fill in the player's trajectory list
                player_trajectory.append(coords)

                # find the position of the ball
                ball_trajectory.append(self.game.get_ball_coords(index))
                
                # determining which team possesses the ball in this instance
                team_possession_trajectory.append(track['possession'])
                
                # keep the true index of the game instance for future use
                self.present_instances.append(index)
                
        self.player_trajectory = player_trajectory
        self.ball_trajectory = ball_trajectory
        self.team_possession_trajectory = team_possession_trajectory


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
            # find the position of the ball
            ball_trajectory.append(self.game.get_ball_coords(index))
            
            # get the information of all players at an instance
            home_players = self.game.get_team_instance_info(index, True)
            away_players = self.game.get_team_instance_info(index, False)
                
            home_coordinates.append(home_players)
            away_coordinates.append(away_players)
        
        self.ball_trajectory = ball_trajectory
        self.home_coordinates = home_coordinates
        self.away_coordinates = away_coordinates
    
    def get_ball_possessor(self, instance_no):
        trackable_object = self.game.game_tracks[instance_no]['possession']['trackable_object']
        team = self.game.game_tracks[instance_no]['possession']['group']
        
        home = True if team == 'home_team' else False
        
        return home, trackable_object
    
    def get_reachable_teammates(self, instance_no: int, first_name: str, last_name: str, home: bool=True) -> int:
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
        
        trackable_object = self.game.get_trackable_object(first_name, last_name)
        
        all_coordinates = []
        home_players = []
        away_players = []
        
        for i in range(len(home_info)):
            all_coordinates.append([home_info[i]['x'], home_info[i]['y']])
            home_players.append(home_info[i]['trackable_object'])
        for i in range(len(away_info)):
            all_coordinates.append([away_info[i]['x'], away_info[i]['y']])
            away_players.append(away_info[i]['trackable_object'])
        
        voronoi_cells = Voronoi(np.array(all_coordinates))
        
        # regardless of the player belonging to either teams, find the
        # player's index and the teammates indices in the voronoi cells
        if home:
            try:
                player_index = home_players.index(trackable_object)
            except:
                return -1
            teammate_indices = [i for i in range(len(home_players))]
            adjacency_matrix = np.zeros((len(home_players), len(home_players)))
        else:
            try:
                player_index = away_players.index(trackable_object) + len(home_players)
            except:
                return -1
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
    
    def attack_detection(self, home: bool=True, mode: str='penetration'):
        # first, extract the instances where the possession is for the specified team
        group = 'home team' if home else 'away team'
        indices = []
        initial_side = self.game.get_team_side(10, home)
        
        attacking_indices = []
        
        i = 0
        while i < len(self.game.game_tracks):
            if self.game.game_tracks[i]['possession']['group'] == group:
                period = self.game.game_tracks[i]['period']
                start_index = i
                
                j = i
                while j < len(self.game.game_tracks) and self.game.game_tracks[j]['possession']['group'] == group:
                    j += 1
                
                end_index = j - 1
                
                indices.append((start_index, end_index, period))
                
                i = j
            else:
                i += 1
        
        # now implement the attack detection strategy (here our main criteria is penetration)
        for index_pair in indices:
            start_coord = self.game.get_ball_coords(index_pair[0])
            end_coord = self.game.get_ball_coords(index_pair[1])
            period = index_pair[2]
            
            if mode == 'penetration':
                penetration_vector = end_coord[0] - start_coord[0]
                
                if (initial_side == 'right_to_left' and period == 1) or (initial_side == 'left_to_right' and period == 2):
                    penetration_vector = -penetration_vector
                
                if penetration_vector > 45:
                    attacking_indices.append(index_pair)
            elif mode == 'concentration':
                condition1 = initial_side == 'left_to_right' and period == 1 and end_coord[0] > 26.25
                condition2 = initial_side == 'right_to_left' and period == 2 and end_coord[0] < -26.25
                
                if condition1 or condition2:
                    attacking_indices.append(index_pair)
                    
        
        return attacking_indices
    
    def defense_detection(self, home: bool=True, mode: str='retreat'):
         # first, extract the instances where the possession is not for the specified team
        group = 'home team' if home else 'away team'
        indices = []
        initial_side = self.game.get_team_side(10, home)
        
        defense_indices = []
        
        i = 0
        while i < len(self.game.game_tracks):
            if self.game.game_tracks[i]['possession']['group'] != group:
                period = self.game.game_tracks[i]['period']
                start_index = i
                
                j = i
                while j < len(self.game.game_tracks) and self.game.game_tracks[j]['possession']['group'] != group:
                    j += 1
                
                end_index = j - 1
                
                indices.append((start_index, end_index, period))
                
                i = j
            else:
                i += 1
        
        # now implement the attack detection strategy (here our main criteria is penetration)
        for index_pair in indices:
            start_coord = self.game.get_ball_coords(index_pair[0])
            end_coord = self.game.get_ball_coords(index_pair[1])
            period = index_pair[2]
            
            if mode == 'retreat':
                retraction_vector = end_coord[0] - start_coord[0]
                
                if (initial_side == 'left_to_right' and period == 1) or (initial_side == 'right_to_left' and period == 2):
                    retraction_vector = -retraction_vector
                
                if retraction_vector > 45:
                    defense_indices.append(index_pair)
                    
            elif mode == 'concentration':
                condition1 = initial_side == 'left_to_right' and period == 1 and end_coord[0] < -26.25
                condition2 = initial_side == 'right_to_left' and period == 2 and end_coord[0] > 26.25
                
                if condition1 or condition2:
                    defense_indices.append(index_pair)
                    
        
        return defense_indices


class PassAnalyzer:
    def __init__(self, pass_instances_path: str, game: GameLoader, all_tracker: AllTracker, n_cluster: int=2) -> None:
        self.pass_instances = pd.read_csv(pass_instances_path)
        self.game = game
        self.all_tracker = all_tracker
        
        if self.all_tracker.ball_trajectory is None:
            self.all_tracker.track()
            
        print('[*] Passing analysis initiated...')
        self.__preprocessing()    
        
        self.__get_number_of_consecutive_passes()
        self.__get_average_passing_distances()
        self.__get_average_passing_durations()
        self.__get_passing_coverage_areas()
        self.__get_center_of_concentration()
        self.__get_average_ball_height()
        self.__get_period_and_teamsides()
        
        self.__construct_feature_vectors()
        
        self.instance_labels, self.centers = self.kmeans_clustering(n_clusters=n_cluster)
        
        for i, center in enumerate(self.centers):
            print('-----------------------------------------------------------')
            print(f'[*] Cluster {i + 1} with {len(self.instance_labels[i])} instances:\n')
            self.translate_instance(center)
        print('-----------------------------------------------------------')
        
        # self.visualize_clusters()
    
    
    def __preprocessing(self):
        print('\__[*] Preprocessing the data...')
        # discarding the failed passes
        self.pass_instances = self.pass_instances[self.pass_instances['outcome'] != 'failed']
        
        # separating the possessions
        self.home_passes_indices = []
        self.away_passes_indices = []

        
        if self.pass_instances.iloc[0]['possessing team'] == 'home team':
            home_start = 0
        else:
            away_start = 0
            
        for i in range(1, len(self.pass_instances)):
            if self.pass_instances.iloc[i]['possessing team'] != self.pass_instances.iloc[i - 1]['possessing team']:
                if self.pass_instances.iloc[i - 1]['possessing team'] == 'home team':
                    home_end = i - 1
                    away_start = i
                    self.home_passes_indices.append((home_start, home_end))
                else:
                    away_end = i - 1
                    home_start = i
                    self.away_passes_indices.append((away_start, away_end))
                
            if i == len(self.pass_instances) - 1:
                if self.pass_instances.iloc[i - 1]['possessing team'] == 'home team':
                    home_end = i
                    self.home_passes_indices.append((home_start, home_end))
                else:
                    away_end = i
                    self.away_passes_indices.append((away_start, away_end))
                        
        # extracting the pass intervals for each team
        self.home_pass_intervals = []
        for item in self.home_passes_indices:
            interval_start, interval_end = item
            passes = []
            start = self.pass_instances.iloc[interval_start]['instance']
            for index in range(interval_start + 1, interval_end + 1):
                if self.pass_instances.iloc[index]['instance'] - self.pass_instances.iloc[index - 1]['instance'] > 1:
                    end = self.pass_instances.iloc[index - 1]['instance']
                    passes.append((start, end))
                    start = self.pass_instances.iloc[index]['instance']
                if index == interval_end:
                    end = self.pass_instances.iloc[index]['instance']
                    passes.append((start, end))
            self.home_pass_intervals.append(passes)
        
        self.away_pass_intervals = []
        for item in self.away_passes_indices:
            interval_start, interval_end = item
            passes = []
            start = self.pass_instances.iloc[interval_start]['instance']
            for index in range(interval_start + 1, interval_end + 1):
                if self.pass_instances.iloc[index]['instance'] - self.pass_instances.iloc[index - 1]['instance'] > 1:
                    end = self.pass_instances.iloc[index - 1]['instance']
                    passes.append((start, end))
                    start = self.pass_instances.iloc[index]['instance']
                if index == interval_end:
                    end = self.pass_instances.iloc[index]['instance']
                    passes.append((start, end))
            self.away_pass_intervals.append(passes)
    
    def __get_number_of_consecutive_passes(self):
        self.home_consecutive_passes_lengths = []
        self.away_consecutive_passes_lengths = []
        
        print('\__[*] Computing the number of pass streaks...')
        for item in self.home_pass_intervals:
            if len(item) != 0:
                self.home_consecutive_passes_lengths.append(len(item))
        
        for item in self.away_pass_intervals:
            if len(item) != 0:
                self.away_consecutive_passes_lengths.append(len(item))
        
    def __get_average_passing_distances(self):
        self.home_average_passing_distances = []
        self.away_average_passing_distances = []
        
        print('\__[*] Computing the average passing distances...')
        for item in self.home_pass_intervals:
            if len(item) != 0:
                avg_distance = 0
                for start, end in item:
                    end_loc = np.array(self.all_tracker.ball_trajectory[end])
                    start_loc = np.array(self.all_tracker.ball_trajectory[start])
                    distance = np.sqrt(((end_loc - start_loc) ** 2).sum())
                    
                    avg_distance += distance
                avg_distance /= len(item)
                self.home_average_passing_distances.append(avg_distance)
        
        for item in self.away_pass_intervals:
            if len(item) != 0:
                avg_distance = 0
                for start, end in item:
                    end_loc = np.array(self.all_tracker.ball_trajectory[end])
                    start_loc = np.array(self.all_tracker.ball_trajectory[start])
                    distance = np.sqrt(((end_loc - start_loc) ** 2).sum())
                    
                    avg_distance += distance
                avg_distance /= len(item)
                self.away_average_passing_distances.append(avg_distance)
    
    def __get_average_passing_durations(self):
        self.home_average_passing_durations = []
        self.away_average_passing_durations = []
        
        print('\__[*] Computing the average passing duration...')
        for item in self.home_pass_intervals:
            if len(item) != 0:
                avg_duration = 0
                for start, end in item:
                    duration  = (end - start + 1) * 0.1
                    
                    avg_duration += duration
                avg_duration /= len(item)
                self.home_average_passing_durations.append(avg_duration)
        
        for item in self.away_pass_intervals:
            if len(item) != 0:
                avg_duration = 0
                for start, end in item:
                    duration  = (end - start + 1) * 0.1
                    
                    avg_duration += duration
                avg_duration /= len(item)
                self.away_average_passing_durations.append(avg_duration)
    
    def __get_passing_coverage_areas(self):
        self.home_passing_coverage_areas = []
        self.away_passing_coverage_areas = []
        
        print('\__[*] Computing the passing polygon coverage area...')
        for item in self.home_pass_intervals:
            if len(item) != 0:
                ball_coords = []
                for start, end in item:
                    ball_coords.append(self.all_tracker.ball_trajectory[start][:2])
                    ball_coords.append(self.all_tracker.ball_trajectory[end][:2])

                try:
                    convex_hull = ConvexHull(np.array(ball_coords))
                    convex_hull_area = convex_hull.area
                    self.home_passing_coverage_areas.append(convex_hull_area)
                except:
                    self.home_passing_coverage_areas.append(0)
        
        for item in self.away_pass_intervals:
            if len(item) != 0:
                ball_coords = []
                for start, end in item:
                    ball_coords.append(self.all_tracker.ball_trajectory[start][:2])
                    ball_coords.append(self.all_tracker.ball_trajectory[end][:2])

                try:
                    convex_hull = ConvexHull(np.array(ball_coords))
                    convex_hull_area = convex_hull.area
                    self.away_passing_coverage_areas.append(convex_hull_area)
                except:
                    self.away_passing_coverage_areas.append(0)
    
    def __get_center_of_concentration(self):
        self.home_passing_center_of_concentration = []
        self.away_passing_center_of_concentration = []
        
        print('\__[*] Computing the passing center of concentration...')
        for item in self.home_pass_intervals:
            if len(item) != 0:
                ball_coords = []
                for start, end in item:
                    ball_coords.append(self.all_tracker.ball_trajectory[start][:2])
                    ball_coords.append(self.all_tracker.ball_trajectory[end][:2])

                ball_coords = np.array(ball_coords)
                center = np.mean(ball_coords, axis=0)
                self.home_passing_center_of_concentration.append(center)
        
        for item in self.away_pass_intervals:
            if len(item) != 0:
                ball_coords = []
                for start, end in item:
                    ball_coords.append(self.all_tracker.ball_trajectory[start][:2])
                    ball_coords.append(self.all_tracker.ball_trajectory[end][:2])

                ball_coords = np.array(ball_coords)
                center = np.mean(ball_coords, axis=0)
                self.away_passing_center_of_concentration.append(center)
    
    def __get_average_ball_height(self):
        self.home_passing_average_ball_height = []
        self.away_passing_average_ball_height = []
        
        print('\__[*] Computing the average height of the ball during the passing...')
        for item in self.home_pass_intervals:
            if len(item) != 0:
                ball_coords = []
                for start, end in item:
                    ball_coords.append(self.all_tracker.ball_trajectory[start][2])
                    ball_coords.append(self.all_tracker.ball_trajectory[end][2])

                ball_coords = np.array(ball_coords)
                height = np.mean(ball_coords)
                self.home_passing_average_ball_height.append(height)
        
        for item in self.away_pass_intervals:
            if len(item) != 0:
                ball_coords = []
                for start, end in item:
                    ball_coords.append(self.all_tracker.ball_trajectory[start][2])
                    ball_coords.append(self.all_tracker.ball_trajectory[end][2])

                ball_coords = np.array(ball_coords)
                height = np.mean(ball_coords)
                self.away_passing_average_ball_height.append(height)
    
    def __get_period_and_teamsides(self):
        self.home_periods = []
        self.home_sides = []
        self.away_periods = []
        self.away_sides = []
        
        for item in self.home_pass_intervals:
            if len(item) > 0:
                instance = item[0][0]
                try:
                    period = self.game.game_tracks[instance]['period']
                except:
                    if instance / 600 > 45:
                        period = 2
                    else:
                        period = 1
                
                side = self.game.game_info['home_team_side'][period - 1]
                side = 1 if side == 'left_to_right' else 2
                
                self.home_periods.append(period)
                self.home_sides.append(side)
                
        for item in self.away_pass_intervals:
            if len(item) > 0:
                instance = item[0][0]
                try:
                    period = self.game.game_tracks[instance]['period']
                except:
                    if instance / 600 > 45:
                        period = 2
                    else:
                        period = 1
                
                side = self.game.game_info['home_team_side'][2 - period]
                side = 1 if side == 'left_to_right' else 2
                
                self.away_periods.append(period)
                self.away_sides.append(side)      
    
    def __construct_feature_vectors(self):
        self.home_feature_vector = []
        self.away_feature_vector = []
        
        print('\__[*] Constructing the feature vectors...')
        for i in range(len(self.home_average_passing_distances)):
            feature = [self.home_consecutive_passes_lengths[i], 
                       self.home_average_passing_distances[i],
                       self.home_average_passing_durations[i],
                       self.home_passing_coverage_areas[i],
                       self.home_passing_center_of_concentration[i][0],
                       self.home_passing_center_of_concentration[i][1],
                       self.home_passing_average_ball_height[i]]
            self.home_feature_vector.append(feature)
        
        self.home_feature_vector = np.array(self.home_feature_vector)
        
        for i in range(len(self.away_average_passing_distances)):
            feature = [self.away_consecutive_passes_lengths[i], 
                       self.away_average_passing_distances[i],
                       self.away_average_passing_durations[i],
                       self.away_passing_coverage_areas[i],
                       self.away_passing_center_of_concentration[i][0],
                       self.away_passing_center_of_concentration[i][1],
                       self.away_passing_average_ball_height[i]]
            self.away_feature_vector.append(feature)
        
        self.away_feature_vector = np.array(self.away_feature_vector)
    
    def feature_vectors_statistics(self, home=True):
        # stats for the home team features
        consecutive_passes = self.home_consecutive_passes_lengths if home else self.away_consecutive_passes_lengths
        distances = self.home_average_passing_distances if home else self.away_average_passing_distances
        durations = self.home_average_passing_durations if home else self.away_average_passing_durations
        areas = self.home_passing_coverage_areas if home else self.away_passing_coverage_areas
        centers = self.home_passing_center_of_concentration if home else self.away_passing_center_of_concentration
        heights = self.home_passing_average_ball_height if home else self.away_passing_average_ball_height
        
        plt.hist(consecutive_passes, bins='auto')
        plt.xticks(np.arange(min(consecutive_passes), max(consecutive_passes) + 1, 1.0))
        plt.title('Histogram of the number of consecutive passes for the home team')
        plt.xlabel('Num Consecutive Passes')
        plt.ylabel('Count')
        plt.yticks(rotation=90)
        plt.show()
        
        plt.hist(distances, bins='auto')
        plt.xticks(np.arange(min(distances), max(distances) + 1, 1.0), rotation=90)
        plt.title('Histogram of the average passing distance in each passing sequence for the home team')
        plt.xlabel('Avg Distance of Pass')
        plt.ylabel('Count')
        plt.yticks(rotation=90)
        plt.show()
        
        plt.hist(durations, bins='auto')
        plt.xticks(np.arange(min(durations), max(durations) + 1, 1.0))
        plt.title('Histogram of the average duration of a pass in each passing sequence for the home team')
        plt.xlabel('Pass Duration (in seconds)')
        plt.ylabel('Count')
        plt.yticks(rotation=90)
        plt.show()
        
        plt.hist(areas, bins='auto')
        plt.xticks(np.arange(min(areas), max(areas) + 1, 4.0), rotation=90)
        plt.title('Histogram of coverage area of passing lines in each passing sequence for the home team')
        plt.xlabel('Area (in $m^2$)')
        plt.ylabel('Count')
        plt.yticks(rotation=90)
        plt.show()

        plt.hist2d(np.array(centers)[:, 0], np.array(centers)[:, 1], [15, 15])
        plt.show()
        
        plt.hist(heights, bins='auto')
        plt.xticks(np.arange(min(heights), max(heights) + 1, 1.0))
        plt.title('Histogram of the average ball height in a pass in each passing sequence for the home team')
        plt.xlabel('Avg Ball Height (in meters)')
        plt.ylabel('Count')
        plt.yticks(rotation=90)
        plt.show()

    def kmeans_clustering(self, home=True, n_clusters=2):
        print('[*] Clustering different passing styles...')
        features = self.home_feature_vector if home else self.away_feature_vector
        
        # normalize the features for better clustering performance
        scaler = StandardScaler()
        scaler.fit(features)
        
        normalized_features = scaler.transform(features)
        
        clusterer = KMeans(n_clusters=n_clusters)
        
        clusterer.fit(normalized_features)
        
        self.features_labels = clusterer.labels_
        
        rescaled_cluster_centers = scaler.inverse_transform(clusterer.cluster_centers_)
        
        instance_labels = {}
        
        for label in np.unique(clusterer.labels_).tolist():
            instance_labels[label] = []
        
        # we must return the instances belonging to each cluster
        index = 0
        for item in self.home_pass_intervals:
            if len(item) > 0:
                for start, end in item:
                    instance_labels[clusterer.labels_[index]].extend([i for i in range(start, end + 1)])
                index += 1
        
        return instance_labels, rescaled_cluster_centers

    def translate_instance(self, feature_vector):
        translation = 'consecutive passing sequences have\n'
        
        translation += f'\t(an average length of {np.round(feature_vector[0], 2)} passes)\n'
        translation += f'\t(an average distance of {np.round(feature_vector[1], 2)} meters per pass)\n'
        translation += f'\t(an average duration of {np.round(feature_vector[2], 2)} seconds per pass)\n'
        translation += f'\t(an average coverage area of {np.round(feature_vector[3], 2)} meters squared)\n'
        translation += f'\t(an average center of concentration of (x, y) = {np.round(feature_vector[4], 2), np.round(feature_vector[5], 2)})\n'
        translation += f'\t(an average ball height of {np.round(feature_vector[6], 2)} meters per pass)\n'
        
        print(translation)
    
    def visualize_clusters(self, home=True):
        features = self.home_feature_vector if home else self.away_feature_vector
        
        pca = PCA(n_components=3)
        pca.fit(features)
        
        reduced_features = pca.transform(features)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=self.features_labels)
        plt.title('clustering result applied on the data\n(the features are mapped to 3D using PCA)')
        plt.show()


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
    
    def plot_game_snapshot(self, instance_no: int, save: bool=True) -> None:
        """
        This function receives the index of a snapshot of the game and plots the coordinates of the home-team
        players (blue), and away players (red) as well as the ball (yellow) on the pitch.
        
        :param instance_no (int): the index of the snapshot we want to illustrate.
        """
            
        ball_trajectory = self.all_tracker.ball_trajectory
        home_coordinates = self.all_tracker.home_coordinates
        away_coordinates = self.all_tracker.away_coordinates
        
        home_instance_coordinates = [[item['x'], item['y']] for item in home_coordinates[instance_no]]
        away_instance_coordinates = [[item['x'], item['y']] for item in away_coordinates[instance_no]]
        
        # extract the pitch size
        pitch_size = self.game.get_pitch_size()
        
        # define the pitch graphical object
        pitch = mpl.Pitch(pitch_type='skillcorner',
                          pitch_color='grass',
                          pitch_length=pitch_size[0], 
                          pitch_width=pitch_size[1], 
                          axis=True, 
                          label=True)
        _, ax = pitch.draw(figsize=(9, 6))
        
        try:   
            pitch.scatter(np.array(home_instance_coordinates)[:, 0], np.array(home_instance_coordinates)[:, 1], ax=ax, facecolor='blue', s=30, edgecolor='k')
            pitch.scatter(np.array(away_instance_coordinates)[:, 0], np.array(away_instance_coordinates)[:, 1], ax=ax, facecolor='red', s=30, edgecolor='k')
            pitch.scatter([ball_trajectory[instance_no][0]], [ball_trajectory[instance_no][1]], ax=ax, facecolor='yellow', s=20, edgecolor='k')
            
            if save:
                plt.savefig(f'visualized/{instance_no}.png')
            else:
                plt.show()    
        except:
            print('[!] No information found on players coordinates')
            pitch.scatter([ball_trajectory[instance_no][0]], [ball_trajectory[instance_no][1]], ax=ax, facecolor='yellow', s=20, edgecolor='k')

            if save:
                plt.savefig(f'visualized/{instance_no}.png')
            else:
                plt.show()    
            
        plt.close()
    
    def plot_voronoi_cell(self, instance_no: int) -> None:
        """
        This function receives the number of a snapshot of the game and plots the Voronoi cell
        of the players in that snapshot.
        
        :param instance_no (int): an integer specifying the index of the snapshot of the game
        """ 
        ball_trajectory = self.all_tracker.ball_trajectory
        home_coordinates = self.all_tracker.home_coordinates
        away_coordinates = self.all_tracker.away_coordinates
        
        home_instance_coordinates = [[item['x'], item['y']] for item in home_coordinates[instance_no]]
        away_instance_coordinates = [[item['x'], item['y']] for item in away_coordinates[instance_no]]
            
        ball_coordinates = ball_trajectory[instance_no]
        all_coordinates = home_instance_coordinates.copy()
        # all_coordinates.extend(away_instance_coordinates)
        
        voronoi_cells = Voronoi(np.array(all_coordinates))

        
        plt.rcParams["figure.figsize"] = (9,6)
        voronoi_plot_2d(voronoi_cells, show_points=False, show_vertices=False, line_colors='darkgreen')
        plt.scatter([ball_coordinates[0]], [ball_coordinates[1]], c='orange')
        plt.scatter(np.array(home_instance_coordinates)[:, 0], np.array(home_instance_coordinates)[:, 1], s=35, c='blue')
        plt.scatter(np.array(away_instance_coordinates)[:, 0], np.array(away_instance_coordinates)[:, 1], s=35, c='red')
        plt.show()
        
        plt.show()
    
    def plot_presence_bar_plot(self) -> None:
        """
        This method plots the player's presence in each frame as a bar plot.
        The plot is also color coded where when the bar is blue it indicates that
        the possession is for the home team and when it is red, the possession is
        for the away team.
        """
        colors = []
        heights = []
        count1 = 0
        count2 = 0
        for i in range(len(self.game.game_tracks)):
            exists, possession = self.game.player_exists(i, self.player.track_id)
            
            if exists:
                heights.append(1)
                if possession == 'home team':
                    count1 += 1
                    colors.append('blue')
                else:
                    count2 += 1
                    colors.append('red')
            else:
                heights.append(0)
                colors.append('white')
                
        print(f'In possession: {count1}, Out of Possession: {count2}, Total frames: {count1 + count2}')
        plt.bar([i for i in range(len(self.game.game_tracks))], heights, color=colors)
        plt.show()
    
    def plot_presence_histogram(self, multi_resolution=False, resolution=50) -> None:
        """
        This method plots the histogram of the lengths of continuous players' presence.
        """
        
        def first_appearence_index(presence_list):
            return presence_list.index(1)
        
        def last_appearence_index(presence_list):
            return len(presence_list) - 1 - presence_list[::-1].index(1)
        
        presence_indicator = []
        presence_histogram = {}
        absence_histogram = {}
        
        for i in range(len(self.game.game_tracks)):
            exists, _ = self.game.player_exists(i, self.player.track_id)
            if exists:
                presence_indicator.append(1)
            else:
                presence_indicator.append(0)
        
        i = 0
        i = first_appearence_index(presence_indicator)
        last_index = last_appearence_index(presence_indicator)
        
        while i <= last_index:
            if presence_indicator[i] == 1:
                j = i + 1
                count = 0
                while  j < last_index and presence_indicator[j] == presence_indicator[i]:
                    count += 1
                    j += 1
                    
                if count in presence_histogram.keys():
                    presence_histogram[count] += 1
                else:
                    presence_histogram[count] = 1
                    
                i = j
            elif presence_indicator[i] == 0:
                k = i + 1
                count2 = 0
                while k < len(presence_indicator) and presence_indicator[k] == presence_indicator[i]:
                    count2 += 1
                    k += 1
                
                if count2 in absence_histogram.keys():
                    absence_histogram[count2] += 1
                else:
                    absence_histogram[count2] = 1
                
                i = k
        
        # computing and ploting the histograms
        presence_lengths = [i for i in range(max(list(presence_histogram.keys())) + 100)]
        presence_heights = [presence_histogram[i] if i in presence_histogram.keys() else 0 for i in range(max(list(presence_histogram.keys())) + 100)]
        
        absence_lengths = [i for i in range(max(list(absence_histogram.keys())) + 100)]
        absence_heights = [absence_histogram[i] if i in absence_histogram.keys() else 0 for i in range(max(list(absence_histogram.keys())) + 100)]
        
        plt.subplot(2, 1, 1)
        plt.bar(presence_lengths, presence_heights, width=4)
        plt.title(f'Player ({self.player.first_name} {self.player.last_name}) Presence Histogram')
        plt.ylabel('Count')
        plt.subplot(2, 1, 2)
        plt.bar(absence_lengths, absence_heights, width=4)
        plt.title(f'Player ({self.player.first_name} {self.player.last_name}) Absence Histogram')
        plt.xlabel('Lengths (= miliseconds)')
        plt.show()
        
        if multi_resolution:
            # by default we set the resolution scale to 5 seconds (500 ms)
            presence_hist_list = []
            for key, value in presence_histogram.items():
                for _ in range(value):
                    presence_hist_list.append(key)
                    
            absence_hist_list = []
            for key, value in absence_histogram.items():
                for _ in range(value):
                    absence_hist_list.append(key)
            
            new_presence_heights = []
            presence_x_axis = []
            for i in range(int(np.ceil(max(list(presence_histogram.keys())) / resolution)) + 1):
                new_presence_heights.append(len(list(filter(lambda x: i * resolution <= x < (i + 1) * resolution, presence_hist_list))))
                presence_x_axis.append(f'<{(i + 1) * int(resolution / 10)}s')
            
            new_absence_heights = []
            absence_x_axis = []
            for i in range(int(np.ceil(max(list(absence_histogram.keys())) / resolution)) + 1):
                new_absence_heights.append(len(list(filter(lambda x: i * resolution <= x < (i + 1) * resolution, absence_hist_list))))
                absence_x_axis.append(f'[{i * int(resolution / 10)}s, {(i + 1) * int(resolution / 10)}s]')
            
            
            plt.subplot(2, 1, 1)
            plt.bar(presence_x_axis, new_presence_heights)
            plt.title(f'{self.player.first_name} {self.player.last_name} Presence (Up) / Absence (Down) Histogram with Resolution {resolution}ms')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.subplot(2, 1, 2)
            plt.bar(absence_x_axis, new_absence_heights)
            plt.xlabel('Duration (in seconds)')
            plt.xticks(rotation=45)
            plt.show()
            
    def visualize_formulas(self, traces, formulas):
        os.makedirs('visualized', exist_ok=True)
        
        ball_trajectory = self.all_tracker.ball_trajectory
        home_coordinates = self.all_tracker.home_coordinates
        away_coordinates = self.all_tracker.away_coordinates
        
        # extract the pitch size
        pitch_size = self.game.get_pitch_size()
        
        index = 1
        for instance_no in range(len(self.game.game_tracks)):
            if self.game.player_exists(instance_no, self.player.track_id)[0]:
                home_instance_coordinates = [[item['x'], item['y']] for item in home_coordinates[instance_no]]
                away_instance_coordinates = [[item['x'], item['y']] for item in away_coordinates[instance_no]]
            
            
                # define the pitch graphical object
                pitch = mpl.Pitch(pitch_type='skillcorner', 
                                pitch_length=pitch_size[0], 
                                pitch_width=pitch_size[1], 
                                axis=True, 
                                label=True)
                _, ax = pitch.draw(figsize=(15, 10))
              
                pitch.scatter(np.array(home_instance_coordinates)[:, 0], np.array(home_instance_coordinates)[:, 1], ax=ax, facecolor='blue', s=30, edgecolor='k')
                pitch.scatter(np.array(away_instance_coordinates)[:, 0], np.array(away_instance_coordinates)[:, 1], ax=ax, facecolor='red', s=30, edgecolor='k')
                pitch.scatter([ball_trajectory[instance_no][0]], [ball_trajectory[instance_no][1]], ax=ax, facecolor='yellow', s=20, edgecolor='k')
                
                player_coords = self.game.get_player_coords(instance_no, self.player.track_id)
                pitch.scatter([player_coords[0]], [player_coords[1]], ax=ax, facecolor='green', s=50, edgecolor='k')
                
                
                # show the corresponding trace and check which formulas are satisfied
                trace = traces[:index]
                instance = traces[index - 1]
                
                satisfactions = []
                for formula in formulas:
                    satisfies = ltl_utils.doesTraceEntailFormula(formula[0], trace)
                    if satisfies:
                        formula_str = '\n'.join(formula[0].split(','))
                        satisfactions.append(u'\u2713' + ' ' + formula_str)
                
                plt.title('\n'.join(instance[0]), loc='left')
                plt.title('\n'.join(satisfactions), loc='right')
                
                index += 1
                    
                plt.savefig(f'visualized/{instance_no}.png')
                plt.close()
        


if __name__ == '__main__':
    first_name = 'Manuel'
    last_name = 'Akanji'
    
    game = GameLoader('data/matches', '852654-Manchester City-Chelsea')
    
    # player = PlayerTracker(game, first_name, last_name, True)
    # all_tracker = AllTracker(game)
    
    # # pass detection correctness test
    # pass_analyzers = PassAnalyzer('pass_instances.csv', game, all_tracker, n_cluster=2)
    
    # vis = Visualizers(game, player, all_tracker)
    
    # for item in pass_analyzers.home_pass_intervals:
    #     for start, end in item:
    #         for instance in range(start, end):
    #             vis.plot_game_snapshot(instance, save=True)
    
    # exit()
    ################
    
    player = PlayerTracker(game, first_name, last_name, True)
    all_tracker = AllTracker(game)
    
    vis = Visualizers(game, player, all_tracker)
    vis.plot_voronoi_cell(1093)
    exit()
    for i in range(40000, 41000, 10):
        vis.plot_game_snapshot(i, save=False)
    exit()
    # exit()
    
    # print(all_tracker.attack_detection(True, 'concentration'))
    # print(all_tracker.defense_detection(True, 'concentration'))
    
    vis.plot_presence_histogram(multi_resolution=True, resolution=50)
    exit()
    game = GameLoader('data/matches', '852654-Manchester City-Chelsea', name=(first_name, last_name))
    
    count2 = 0
    for i in range(len(game.game_tracks)):
        if game.player_exists(i, game.get_trackable_object(first_name, last_name))[0]:
            count2 += 1
    
    print(count2)
    
    print(count2 - count1, (count2 - count1) / count1)
    player = PlayerTracker(game, first_name, last_name, True)
    all_tracker = AllTracker(game)
    
    vis = Visualizers(game, player, all_tracker)
    vis.plot_presence_histogram(multi_resolution=True, resolution=50)
    exit()
    # home, track_obj = all_tracker.get_ball_possessor(150)
    # print(home, track_obj)
    # vis.plot_voronoi_cell(1093)
    # print(all_tracker.get_reachable_teammates(1093, first_name, last_name, True))
    # vis.plot_trajectory(start_time=None, end_time=200)
    vis.plot_presence_heat_map()
    # print(game.get_team_instance_info(1112, True))
    for i in range(6400, 6700):
        # print(all_tracker.game.game_tracks[i]['timestamp'])
        vis.plot_game_snapshot(i)
        ball_coords = game.get_ball_coords(i)
        print(i, game.game_tracks[i]['timestamp'], game.game_tracks[i]['data'], np.linalg.norm(ball_coords))
        # print(game.get_ball_coords(i))
