import argparse
from itertools import permutations
import json
from operator import itemgetter
import time
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import json
import statistics

# Local imports
import BayesLTL.ltlfunc as ltlfunc
import BayesLTL.interestingness as interestingness

#############################################################


def run_ltl_inference(data, output_filepath=None, sample_count=10):

    # Vocabulary (lowercased, unique)
    vocab = [s.lower() for s in data['vocab']]
    vocab = list(set(vocab))

    # Traces - organize both pos and neg clusters
    cluster_A = []
    for i, trace in enumerate(data['traces_pos']):
        trace = [[v.lower() for v in s] for s in trace]  # lowercase
        temp = dict()
        temp['name'] = 'a' + str(i)  # Create a name id
        temp['trace'] = tuple(trace)  # Use tuple
        cluster_A.append(temp)

    cluster_B = []
    for i, trace in enumerate(data['traces_neg']):
        trace = [[v.lower() for v in s] for s in trace]
        temp = dict()
        temp['name'] = 'b' + str(i)
        temp['trace'] = tuple(trace)
        cluster_B.append(temp)
    # X = (cluster_A, cluster_B)  # Evidence

    # Parameters
    inference = data['params']['inference']
    iterations = 500#data['params']['iterations']
    conjoin = data['params']['conjoin']
    ltl_sample_cnt = data['params'].get('ltl_sample_cnt', sample_count)
    run_reversed_inference = data['params'].get('reversed_inference', True)
    verbose = False#data['params'].get('verbose', False)

    # Default inference parameters
    params = dict()
    params['alpha'] = data['params'].get('alpha', 0.01)
    params['beta'] = data['params'].get('beta', 0.01)
    params['lambda'] = data['params'].get('lambda', 0.60)
    params['epsilon'] = data['params'].get('epsilon', 0.2)

    # Get LTL templates
    if 'probs_templates' in data:
        probs_templates = data['probs_templates']
    else:
        probs_templates = None
    templates = ltlfunc.getLTLtemplates(user_probs=probs_templates)

    # Get permutation tables
    perm_table = dict()
    perm_table[1] = [list(i) for i in permutations(vocab, 1)]
    perm_table[2] = [list(i) for i in permutations(vocab, 2)]

    ltl_rundata = [
        {'X': (cluster_A, cluster_B), 'reversed': False}
    ]

    if run_reversed_inference:
        ltl_rundata.append(
            {'X': (cluster_B, cluster_A), 'reversed': True}
        )

    # Preparing json output
    output_inference = list()

    for data_X in ltl_rundata:
        X = data_X['X']
        reversed = data_X['reversed']

        cluster_A_inf, cluster_B_inf = X

        output = list()

        #######################################################
        # RUN INFERENCE
        #
        # 1) Metropolis-Hastings Sampling
        if inference == 'mh':

            # Initial guess
            ltl_initial = ltlfunc.samplePrior(templates, vocab, perm_table, params['lambda'], conjoin)
            st = time.time()

            # Preparation
            burn_in_mh = 500
            num_iter_mh = iterations + burn_in_mh
            memory = dict()
            cache = dict()

            # Run MH Sampler
            sampler = ltlfunc.MH_sampler(ltl_initial, X, vocab, templates, params, perm_table, memory, cache, conjoin)
            sampler.runMH(num_iter_mh, burn_in_mh, verbose=verbose)
            memory = sampler.memory

        
            ranked = sorted(sampler.posterior_dict, key=sampler.posterior_dict.get, reverse=True)
            i = 0

            for r in ranked:
                cscore = sampler.cscore_dict[r]
                cscore1, cscore2 = memory[r]
                cscore2 = 1 - cscore2
      
                try:
                    ltl_meaning = sampler.ltl_str_meanings[r]['meaning']
                    ltl = sampler.ltl_log[r]
                    ltl_name = ltl['name']
                    ltl_props = ltl['props_list'] if conjoin else [ltl['props']]
                except:
                    #ltl_meaning = np.nan
                    pass
                

                # Positive set support
                positive_support = interestingness.compute_support(cluster_A_inf, ltl_name, ltl_props, vocab)

                if positive_support == 0:
                    continue

                i += 1

                if i >= ltl_sample_cnt:
                    break

                # Adding to output
                try:
                    temp = dict()
                    temp['formula'] = r
                    temp['meaning'] = sampler.ltl_str_meanings[r]
                    temp['accuracy'] = cscore
                    temp['cscores_individual'] = (cscore1, cscore2)
                    temp['interestingness'] = positive_support
                    temp['reversed'] = reversed
                    output.append(temp)
                except:
                    pass



        # 2) Brute force search (delimited enumeration)
        elif inference == 'brute':
            st = time.time()
            if conjoin:
                # Brute force random sampler (b/c pre-enumerating everything is intractable)
                ltl_full = []
                history = []
                num_brute_force = iterations

                # Collection loop
                while len(history) < num_brute_force:
                    s = ltlfunc.samplePrior(templates, vocab, perm_table, conjoin=conjoin, doRandom=True)
                    ltl_str = s['str_friendly']
                    if ltl_str not in history:
                        ltl_full.append(s)
                        history.append(ltl_str)


            else:
                # If not using conjunction, then obtain a full brute force list
                ltl_full = []
                for template in templates:
                    results = ltlfunc.instantiateLTLvariablePermutate(template, vocab)
                    ltl_full += results

            # Exact inference on collection
            memory = dict()
            cache = dict()
            for ltl_instance in ltl_full:
                log_posterior, cscore, memory = ltlfunc.computePosterior(ltl_instance, X, vocab, params, memory, cache,
                                                                         conjoin)
                ltl_instance['posterior'] = log_posterior
                ltl_instance['cscore'] = cscore
                ltl_instance['cscores_individual'] = memory[ltl_instance['str_friendly']]
                

            # Rank posterior and print top-10 samples
            ranked = sorted(ltl_full, key=itemgetter('posterior'), reverse=True)
            i = 0

            for r in ranked:
                cscore1, cscore2 = r['cscores_individual']
                cscore2 = 1 - cscore2

                ltl_name, ltl_props = r['name'], r['props_list']

                # Positive set support
                positive_support = interestingness.compute_support(cluster_A_inf, ltl_name, ltl_props, vocab)

                if positive_support == 0:
                    continue

                i += 1

                if i >= ltl_sample_cnt:
                    break

                # Adding to output
                temp = dict()
                temp['formula'] = r['str_friendly']
                temp['meaning'] = r['str_meaning']
                temp['accuracy'] = r['cscore']
                temp['cscores_individual'] = (cscore1, cscore2)
                temp['interestingness'] = positive_support
                temp['reversed'] = reversed
                output.append(temp)

        else:
            raise AttributeError("Wrong inference mode specified.")

        #######################################################
        # END OF INFERENCE
        #######################################################

        # Append local ltl order inference output to global output list
        output_inference.extend(output)
        output_inference = sorted(output_inference, key=lambda x: x['accuracy'], reverse=True)[:ltl_sample_cnt]


    return output_inference
