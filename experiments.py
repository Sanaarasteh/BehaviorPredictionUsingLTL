import os
import json

from sc_features import *
from sc_pipeline import Pipeline


def formula_distribution(pipeline: Pipeline, num_experiments: int=10):
    os.makedirs('results', exist_ok=True)
    pipeline.make_propositions(on_ball_possession=False)
    formula_props = {}
    for exp in range(num_experiments):
        print(f'\n\n****** Experiment {exp + 1}/{num_experiments} ******\n\n')
        formulas = pipeline.make_inference()
        
        for form, score in formulas:
            if form not in formula_props.keys():
                formula_props[form] = {'count': 1, 'score': score}
            else:
                formula_props[form]['count'] += 1
                formula_props[form]['score'] += score
    
    with open(f'results/formula_distribution_{first_name}_{last_name}_{"_".join(game_name.split("-")[1:])}.json', 'w') as handle:
        json.dump(formula_props, handle)


def feature_exclusion(pipeline: Pipeline):
    np.random.seed(2)
    os.makedirs('results', exist_ok=True)
    
    feat_formulas = {}
        
    pipeline.make_propositions(on_ball_possession=False)
    print(f'\n\n****** All Features Included ******')
    
    feat_formulas['all'] = pipeline.make_inference()
    
    features_back_up = pipeline.feature_makers.copy()
    
    for i in range(len(features_back_up)):
        print(f'\n\n****** Feature {i + 1} ({features_back_up[i].__class__.__name__}) Excluded ******')
        pipeline.feature_makers = features_back_up.copy()
        pipeline.feature_makers.pop(i)
        
        pipeline.make_propositions(on_ball_possession=False)
        
        feat_formulas[f'feat{i + 1}_removed'] = pipeline.make_inference()
    
    with open(f'results/features_effect_{first_name}_{last_name}_{"_".join(game_name.split("-")[1:])}.json', 'w') as handle:
        json.dump(feat_formulas, handle)

    
    


if __name__ == '__main__':
    base_path = 'data/matches'
    # game_name = '775627-Manchester City-Leicester'
    # first_name = 'Kevin'
    # last_name = 'De Bruyne'
    game_name = '852654-Manchester City-Chelsea'
    first_name = 'Kyle'
    last_name = 'Walker'
    home = True

    # constructing the pipeline
    pipeline = Pipeline(base_path, game_name, first_name, last_name, home)

    # constructing the 
    feat1 = PlayerLocFeature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker)
    feat2 = TeammateDensityFeature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker)
    feat3 = OpponentsDensityFeature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker)
    feat4 = HasBallFeature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker)
    feat5 = CanPassToFeature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker)
    feat6 = ProgressivePassingLane(pipeline.game, pipeline.player_tracker, pipeline.all_tracker, threshold='low')
    feat7 = ProgressivePassingLane(pipeline.game, pipeline.player_tracker, pipeline.all_tracker, threshold='medium')
    feat8 = ProgressivePassingLane(pipeline.game, pipeline.player_tracker, pipeline.all_tracker, threshold='high')
    feat9 = BackwardPassingLane(pipeline.game, pipeline.player_tracker, pipeline.all_tracker, threshold='low')
    feat10 = BackwardPassingLane(pipeline.game, pipeline.player_tracker, pipeline.all_tracker, threshold='medium')
    feat11 = BackwardPassingLane(pipeline.game, pipeline.player_tracker, pipeline.all_tracker, threshold='high')
    feat12 = OpponentPressureFeature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker)
    feat13 = VelocityFeature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker, interval=20)
    feat14 = ExpansionContractionFeature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker, interval=20, k=3)
    feat15 = PlayerLoc2Feature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker)
    feat16 = PositionSpectralFeature(pipeline.game, pipeline.player_tracker, pipeline.all_tracker, interval=20)

    features = [feat1, feat2, feat3, feat4, feat5, feat6, 
                feat7, feat8, feat9, feat10, feat11, feat12,
                feat13, feat14, feat15, feat16]

    pipeline.add_features(features)
    
    
    # # Exp1: extracting the statistical distribution of the 
    # # formulas in repeated runs
    # formula_distribution(pipeline, 10)
    
    # Exp2: extracting the effect of each feature on the 
    # LTL formulas as scores
    feature_exclusion(pipeline)
    