import json

import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint


def predicate_parser(whole_data, predicate_type):
    general_predicates = {}
    for key in whole_data.keys():
        if key.startswith(predicate_type):
            new_keys = key.split(':')[1].strip()
            new_keys = new_keys.split(',')
            
            for new_key in new_keys:
                new_key = new_key[1:-1]
                if new_key.startswith('(('):
                    new_key = new_key[1:]
                
                if new_key in general_predicates.keys():
                    general_predicates[new_key]['total_count'] += whole_data[key]['count']
                    general_predicates[new_key]['total_score'] += whole_data[key]['score']
                else:
                    general_predicates[new_key] = {}
                    general_predicates[new_key]['total_count'] = whole_data[key]['count']
                    general_predicates[new_key]['total_score'] = whole_data[key]['score']

    counts = list(map(lambda x: x['total_count'], list(general_predicates.values())))
    scores = list(map(lambda x: x['total_score'] / x['total_count'], list(general_predicates.values())))
    
    plt.bar(list(general_predicates.keys()), counts)
    plt.title(f'Distribution of "{predicate_type}" LTL predicate types after 10 experiments (200 generated formula)')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.show()
    
    plt.bar(list(general_predicates.keys()), scores)
    plt.title(f'Average scores of "{predicate_type}" LTL predicate types after 10 experiments (200 generated formula)')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.show()

def predicate_parser_with_implication(whole_data, predicate_type):
    general_predicates = {}
    for key in whole_data.keys():
        if key.startswith(predicate_type):
            new_keys = key.split(':')[1].strip()
            if len(new_keys.split(',')) == 2:
                antecedent, consequent = new_keys.split(',')
                antecedent = antecedent.strip()[1:]
                consequent = consequent.strip()[:-1]
                
                new_key = f'{antecedent}=>\n{consequent}'
                
                if new_key in general_predicates.keys():
                    general_predicates[new_key]['total_count'] += whole_data[key]['count']
                    general_predicates[new_key]['total_score'] += whole_data[key]['score']
                else:
                    general_predicates[new_key] = {}
                    general_predicates[new_key]['total_count'] = whole_data[key]['count']
                    general_predicates[new_key]['total_score'] = whole_data[key]['score']
            else:
                pairs = new_keys.split(')), ((')
                
                for i, pair in enumerate(pairs):
                    antecedent, consequent = pair.split(',')
                    
                    if i == 0:
                        antecedent = antecedent.strip()[1:]
                        consequent = consequent.strip() + ')'
                    elif i == len(pairs) - 1:
                        antecedent = '(' + antecedent.strip()
                        consequent = consequent.strip()[:-1]
                    else:
                        antecedent = '(' + antecedent.strip()
                        consequent = consequent.strip() + ')'
                    
                    new_key = f'{antecedent}=>\n{consequent}'
                
                    if new_key in general_predicates.keys():
                        general_predicates[new_key]['total_count'] += whole_data[key]['count']
                        general_predicates[new_key]['total_score'] += whole_data[key]['score']
                    else:
                        general_predicates[new_key] = {}
                        general_predicates[new_key]['total_count'] = whole_data[key]['count']
                        general_predicates[new_key]['total_score'] = whole_data[key]['score']

    counts = list(map(lambda x: x['total_count'], list(general_predicates.values())))
    scores = list(map(lambda x: x['total_score'] / x['total_count'], list(general_predicates.values())))
    
    plt.bar(list(general_predicates.keys()), counts)
    plt.title(f'Distribution of "{predicate_type}" LTL predicate types after 10 experiments (200 generated formula)')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.show()
    
    plt.bar(list(general_predicates.keys()), scores)
    plt.title(f'Average scores of "{predicate_type}" LTL predicate types after 10 experiments (200 generated formula)')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.show()


def feature_importance_analyzer():
    file_names = ['features_effect_Kevin_De Bruyne_Manchester City_Leicester.json', 
                  'features_effect_Kyle_Walker_Manchester City_Chelsea.json', 
                  'features_effect_Manuel_Akanji_Manchester City_Chelsea.json']
    
    for file_name in file_names:
        with open(f'results/{file_name}', 'r') as handle:
            data = json.load(handle)
        print(f'\n\n ############### {file_name} ###############')
        for key in data.keys():
            info = {'sometime_before': 0, 'response': 0, 'stability': 0, 'global': 0, 'eventual': 0}
            print(f'********** {key} **********')
            parse1 = data[key]
            max_score = parse1[0][-1]
            for item in parse1:
                if 'sometime_before' in item[0]:
                    info['sometime_before'] += 1
                elif 'response' in item[0]:
                    info['response'] += 1
                elif 'stability' in item[0]:
                    info['stability'] += 1
                elif 'global' in item[0]:
                    info['global'] += 1
                elif 'eventual' in item[0]:
                    info['eventual'] += 1
            
            pprint(info)
            print(max_score)
            print(info['response'] + info['sometime_before'])
            print('****************************************')
        print(f'\n\n ###########################################')

if __name__ == '__main__':
    feature_importance_analyzer()
    exit()
    with open('results/formula_distribution_Kevin_De Bruyne_Manchester City_Leicester.json', 'r') as handle:
        data = json.load(handle)

    unique_predicate_types = {}

    for key in data.keys():
        predecessor = key.split(':')[0].strip()
        if predecessor in unique_predicate_types.keys():
            unique_predicate_types[predecessor] += data[key]['count']
        else:
            unique_predicate_types[predecessor] = data[key]['count']

    plt.bar(list(unique_predicate_types.keys()), list(unique_predicate_types.values()))
    plt.title('Distribution of LTL predicate types after 10 experiments (200 generated formula)')
    plt.show()

    # distribution of global predicates
    predicate_parser(data, 'global')

    # distribution of eventual predicates
    predicate_parser(data, 'eventual')
    
    # distribution of stability predicates
    predicate_parser(data, 'stability')
    
    # distribution of response predicates
    predicate_parser_with_implication(data, 'response')
    
    # distribution of sometime_before predicates
    predicate_parser_with_implication(data, 'sometime_before')
    
