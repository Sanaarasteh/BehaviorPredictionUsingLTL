import requests
import json
import os

os.makedirs('data', exist_ok=True)
os.makedirs('data/matches', exist_ok=True)

authorization = ('21sa145@queensu.ca', 'M4t!nS4n4')

print('[*] Downloading all the available matches in the English Premier League...')

competitions = requests.get('https://skillcorner.com/api/competitions/?user=false', auth=authorization)
competitions = json.loads(competitions.text)

with open('data/competitions.json', 'w') as handle:
    json.dump(competitions, handle)


matches = requests.get('https://skillcorner.com/api/matches/?competition=1', auth=authorization)
matches = json.loads(matches.text)

with open('data/matches.json', 'w') as handle:
    json.dump(matches, handle)

for i, match in enumerate(matches['results']):
    home =  match['home_team']['short_name']
    away = match['away_team']['short_name']
    match_id = match['id']
    date = match['date_time'].split('T')[0]
    print(f'{i + 1} - Downloading the tracking data of the match "{home}" vs "{away}" on {date}')
    tracking = requests.get(f'https://skillcorner.com/api/match/{match_id}/tracking/?file_format=jsonl&data_type=tracking-extrapolated', auth=authorization).text.split('\n')
    json_data = []
    for track in tracking:
         try:
             json_data.append(json.loads(track))
         except:
             continue

    with open(f'data/matches/{match_id}-{home}-{away}.json', 'w') as handle:
        json.dump(json_data, handle)
    
    match_info = requests.get(f'https://skillcorner.com/api/match/{match_id}/', auth=authorization).text
    match_info = json.loads(match_info)
    
    with open(f'data/matches/{match_id}-{home}-{away}-match-info.json', 'w') as handle:
        json.dump(match_info, handle)
        

print('--done--')
