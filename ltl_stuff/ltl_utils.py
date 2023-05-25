import random
import copy

import BayesLTL.py_inference as py_inference
import BayesLTL.ltlfunc as ltlfunc

# Creates list of propositions from formula
def get_props(formula):
    return [formula.split(': ')[1].split(', ')[x][1:-1].split(',') for x in range(len(formula.split(': ')[1].split(', ')))]


# Takes arbitrary nested list of lists and returns flattened iterable
def flatten_list(container):
        x = []
        for i in container:
            if isinstance(i, (list,tuple)):
                for j in flatten_list(i):
                    yield j
            else:
                yield i
                
                
# Takes formula and translates to list of unique elements
def formulaToVocab(formula):
    return list(set(flatten_list(get_props(formula))))


# Takes traces and translates to list of unique elements
def tracesToVocab(traces):
    return list(set(flatten_list(traces)))


# Takes traces and formula and return all unique elements as list
def tracesAndFormulaToVocab(traces, formula):
    return list(set(tracesToVocab(traces) + formulaToVocab(formula)))


# Returns input file ready for BayesLTL
def createBayesLTLInput(traces_pos=[], traces_neg=[]):
    return {'vocab':tracesToVocab(traces_pos + traces_neg),
            'params':{'conjoin': True,
                      'inference': 'mh',
                      'iterations': 2000},
            'probs_templates':{'eventual': 2.0,
                               'global': 1.0,
                               'until': 0.0,
                               'response': 2.0,
                               'stability': 1.0,
                               'atmostonce': 1.0,
                               'sometime_before': 1.0},
            'traces_pos':traces_pos,
            'traces_neg':traces_neg
           }


# Randomizes traces and splits into 2 balanced groups
def splitIntoTwoRandomizedGroups(traces):
    all_traces = traces.copy()
    random.shuffle(all_traces)
    return all_traces[:len(all_traces)//2], all_traces[len(all_traces)//2:]


# Runs BayesLTL and returns list of top LTL specification formulas
# Quantity can be adjusted in py_inference module. Currently gives 20.
def bayesLtlExplanationsList(group_A, group_B, sample_count=10):
    input_file = createBayesLTLInput(traces_pos=group_A, traces_neg=group_B)
    while True:
        try:
            formulas = [x['formula'] for x in py_inference.run_ltl_inference(input_file, sample_count=sample_count)]
        except UnboundLocalError:
            print('BayesLTL Error, trying again')
            continue
        break
    return formulas


# Runs BayesLTL and returns top formula
def bayesLtlBestExplanation(group_A, group_B):
    return bayesLtlExplanationsList(group_A, group_B)[0]


# Takes string formula and converts to ltl-format to keep BayesLTL happy
def LTLfromFormula(formula, vocab):
    template = ltlfunc.getLTLtemplates(formula.split(':'))[0]
    props = get_props(formula)
    LTLconjunct = ltlfunc.getLTLconjunct(template, vocab, props)
    return LTLconjunct


# Check whether single trace entails formula and returns boolean
def doesTraceEntailFormula(formula, trace):
    vocab = tracesAndFormulaToVocab(trace, formula)
    cluster_A = [{'name':'a0',
                  'trace': trace}]
    cluster_B = [{'name':'a1',
                  'trace': [[]]}]
    cscore_A, cscore_B  = ltlfunc.checkContrastiveValidity(LTLfromFormula(formula, vocab), cluster_A, cluster_B, vocab)
    if cscore_A == 1.0:
        return True
    else:
        return False

    
# Takes multiple traces and a formula. Splits into positive and negative groups
# of traces based on whether each single trace entails the formula or not
def split_traces_by_formula_entailment(traces, formula):
    positive = []
    negative = []
    for trace in traces:
        if doesTraceEntailFormula(formula, trace):
            positive.append(trace)
        else:
            negative.append(trace)
    return positive, negative


# Calculates information gain given 2 resulting sets of a split
def info_gain(set_a, set_b):
    return 1-abs(len(set_a)-len(set_b))/((len(set_a))+len(set_b))


# Sample and Randomize traces, split and run BayesLTL, check balance, repeat max_iter times returning 2 groups from best split and formula
def monte_carlo_split(all_traces, max_iter=10, sample_prop=1, min_sample=10, ig_threshold=0.8):
    best_info_gain = -100000
    best_formula = None
    for i in range(max_iter):
        sample_size = min(len(all_traces), max(int(len(all_traces)*sample_prop), min_sample))
        group_A, group_B = splitIntoTwoRandomizedGroups(random.sample(all_traces,sample_size))
        formula = bayesLtlBestExplanation(group_A, group_B)
        temp_A, temp_B = split_traces_by_formula_entailment(all_traces, formula)
        ig = info_gain(temp_A, temp_B)
        if ig > best_info_gain:
            best_info_gain = ig
            best_formula = formula
        if best_info_gain >= ig_threshold:
            break
    split = split_traces_by_formula_entailment(all_traces, best_formula)
    return {'positive': split[0],
            'negative': split[1],
            'formula':best_formula}
        
    
    
# There is easy room to speed these up by stopping when False
def get_static_fluents_single_trace(trace):
    static_fluents = []
    for v in tracesToVocab(trace):
        is_vocab_in_bitvec = []
        for y in trace:
            is_vocab_in_bitvec.append(v in y)  
        if all(is_vocab_in_bitvec):
            static_fluents.append(v)
    return static_fluents

def get_static_fluents_multiple_traces(traces):
    static_fluents = []
    for v in tracesToVocab(traces):
        is_vocab_static_fluent_in_trace = []
        for y in traces:
            is_vocab_static_fluent_in_trace.append(v in get_static_fluents_single_trace(y))
        if all(is_vocab_static_fluent_in_trace):
            static_fluents.append(v)
    return static_fluents

def filter_static_fluents(traces):
    filtered_traces = copy.deepcopy(traces)
    static_fluents = get_static_fluents_multiple_traces(traces)
    for trace in filtered_traces:
        for bitvec in trace:
            for fluent in static_fluents:
                if fluent in bitvec:
                    bitvec.remove(fluent)
    return filtered_traces

