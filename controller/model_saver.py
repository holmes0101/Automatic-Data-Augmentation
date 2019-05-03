import os
import json
import numpy as np

LOGDIR = "./logs/autoaugment_runs"

class ModelSaver:
    def __init__(self, n):
        self.N_TOKEEP = n                # Num of models to save in memory
        self.N_SAVED  = 1                # Num of models currently in memory
        self.KEPT_POL = []               # Policies kept in memory
        self.KEPT_REW = []               # Policies rewards

    def __call__(self, child_model, subpolicy_list, reward):
        def create_dict(subpolicy_list, reward, index):
            """
            This function creates and saves a dict containing the subpolicy
            in memory
            """
            policy_dict = {
                'Subpolicy {}'.format(i): {
                    operation.name: {
                        'Probability': operation.prob,
                        'Magnitude': operation.magn
                    } for operation in subpolicy.operations 
                } for i, subpolicy in enumerate(subpolicy_list)
            }
            policy_dict['Reward'] = reward

            with open(os.path.join(LOGDIR, "policies_to_keep/model_{}_policy.json".format(index)), 'w') as fp:
                json.dump(policy_dict, fp)
            return policy_dict

        # Managing models in memory
        if self.N_SAVED <= self.N_TOKEEP:
            # Saving Subpolicy
            self.KEPT_REW.append(reward)
            self.KEPT_POL.append(create_dict(subpolicy_list, reward, self.N_SAVED))
            # Saving model
            model_save_path = os.path.join(LOGDIR, "models_to_keep/")
            child_model.save_weights(os.path.join(model_save_path, "best-model_{}.h5".format(self.N_SAVED)))
            self.N_SAVED += 1
        elif reward > np.min(self.KEPT_REW):
            i_min = np.argmin(self.KEPT_REW)
            # Saving Subpolicy
            self.KEPT_REW[i_min] = reward
            self.KEPT_POL[i_min] = create_dict(subpolicy_list, reward, i_min)
            # Saving model
            model_save_path = os.path.join(LOGDIR, "models_to_keep/")
            child_model.save_weights(os.path.join(model_save_path, "best-model_{}.h5".format(self.N_SAVED)))
            
