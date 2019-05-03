import numpy as np

from transformations import all_transforms as transformations

# Global parameters
N_SUBPOL = 5                        # Number of subpolicies
N_OPS    = 2                        # Number of operations within each subpolicy
N_TYPES  = len(transformations)     # Number of different types of transformations
N_PROBS  = 11                       # Number of different discretized probabilities
N_MAG    = 11                       # Number of different discretized magnitudes

class Operation:
    def __init__(self, type_softmax, probability_softmax, magnitude_softmax, greedy = False):
        if greedy:
            self.type = type_softmax.argmax()                               # Transformation type
            t         = transformations[self.type]                          # t <- dict containing 0: function, 1: minval, 2: maxval
            self.prob = probability_softmax.argmax() / (N_PROBS - 1)        # Normalizes prob
            m         = magnitude_softmax.argmax() / (N_MAG - 1)            # Normalization factor
            self.magn = m * (t['maxval'] - t['minval']) + t['minval']       # Normalizes magnitude between [minval, maxval]
        else:
            self.type = np.random.choice(N_TYPES, p = type_softmax)
            t         = transformations[self.type]
            self.prob = np.random.choice(np.linspace(0, 1, N_PROBS), p = probability_softmax)
            self.magn = np.random.choice(np.linspace(t['minval'], t['maxval'], N_MAG), p = magnitude_softmax)
        self.transformation = t['func']                                           # Transformation function
        self.name           = t['name']                                           # Transformation name

    def __call__(self, input_batch, reference_batch):
        input_batch_t     = np.copy(input_batch)
        reference_batch_t = np.copy(reference_batch)
        if np.random.rand() < self.prob:
            input_batch_t, reference_batch_t = self.transformation(input_batch, reference_batch, self.magn)
        return input_batch_t, reference_batch_t

    def __str__(self):
        return "Operation: {} | Probability: {} | Magnitude: {}".format(self.name, self.prob, self.magn)

class Subpolicy:
    def __init__(self, *ops):
        self.operations = ops           # List of operations to apply

    def __call__(self, input_batch, reference_batch):      # Apply each single operation in the list
        input_batch_t     = np.copy(input_batch)
        reference_batch_t = np.copy(reference_batch)

        for op in self.operations:
            input_batch_t, reference_batch_t = op(input_batch_t, reference_batch_t)    
        
        return input_batch_t, reference_batch_t

    def __str__(self):
        ret = ''
        for i, op in enumerate(self.operations):
            ret += str(op)
            if i < len(self.operations) - 1:
                ret += '\n'
        return ret

def json2policy(policy_json, verbose = 1):
    policy = []
    transform_names = np.array([t['name'] for t in transformations])
    transform_probs = np.linspace(0, 1, N_PROBS)

    for subpolicy in policy_json.keys():
        if subpolicy == "Reward":
            break
        operations = []
        subpolicy_dict = policy_json[subpolicy]
        for transformation in subpolicy_dict.keys():
            # Get operation values
            op_type = transformation
            op_prob = policy_json[subpolicy][transformation]["Probability"]
            op_magn = policy_json[subpolicy][transformation]["Magnitude"]
            if verbose:
                print("Transform information:")
                print("Type: {}".format(op_type))
                print("Prob: {}".format(op_prob))
                print("Magn: {}".format(op_magn))
            # Mapping values => softmaxes
            which_transform = np.where(transform_names == op_type)[0][0]
            transform_magns = np.linspace(transformations[which_transform]['minval'], transformations[which_transform]['maxval'], N_MAG)
            if verbose:
                print("Prob array: {}".format(transform_probs))
                print(np.isclose(transform_probs, op_prob))
                print("Magn array: {}".format(transform_magns))
                print(np.isclose(transform_magns, op_magn))
            which_prob      = np.where(np.isclose(transform_probs, op_prob))[0][0]
            which_magn      = np.where(np.isclose(transform_magns, op_magn))[0][0]

            # Softmaxes (dirac deltas)
            softmax_type = np.zeros([len(transformations),]); softmax_type[which_transform] = 1
            softmax_prob = np.zeros([N_PROBS, ]); softmax_prob[which_prob] = 1
            softmax_magn = np.zeros([N_MAG, ]); softmax_magn[which_magn] = 1

            # Operation
            op = Operation(softmax_type, softmax_prob, softmax_magn, greedy = True)

            operations.append(op)
        policy.append(Subpolicy(*operations))

    return policy
