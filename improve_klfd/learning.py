from improve_klfd.utils import ls_dir_idxs
import os
import numpy as np
from sklearn.decomposition import PCA
from improve_klfd.goal_model import HMMGoalModel
from improve_klfd.rl import HMMES
from improve_klfd.s2d import Sparse2Dense


class KFLFD(object):
    def __init__(self, data_dir, n_offspring, perception_dim=8):
        '''
        :param data_dir: Directory of the demonstrations. Each demo should have names 1,2,3 etc. and each demo folder
        must contain keyframes.csv, perception.csv and ee_poses_wrt_obj.csv files.
        '''

        self.data_dir = data_dir
        self.n_offspring = n_offspring
        self.perception_dim = perception_dim

        self.demo_dirs = ls_dir_idxs(data_dir)
        self.pca = PCA(n_components=perception_dim)

        self.perception_kf_data = []
        self.ee_kf_data = []

        self.durations = []

        for ddir in self.demo_dirs:
            per_dir = os.path.join(ddir, 'perception.csv')
            ee_dir = os.path.join(ddir, 'ee_poses_wrt_obj.csv')
            keyframe_dir = os.path.join(ddir, 'keyframes.csv')

            keyframe_times = np.loadtxt(keyframe_dir, delimiter=',')
            perception_data = np.loadtxt(per_dir, delimiter=',')
            ee_data = np.loadtxt(ee_dir, delimiter=',')

            self.durations.append(ee_data[-1]-ee_data[0])

            # -1 since first dimension is time
            perception_kf = np.zeros((len(keyframe_times), perception_data.shape[-1]-1))
            ee_kf = np.zeros((len(keyframe_times), ee_data.shape[-1]-1))

            first_perception = perception_data[0,1:].reshape(1,-1)
            last_perception = perception_data[-1,1:].reshape(1,-1)

            first_ee = ee_data[0,1:].reshape(1,-1)
            last_ee = ee_data[-1,1:].reshape(1,-1)

            # Get closest ee poses and perception to a given kf time
            for i, kf_t in enumerate(keyframe_times):
                ee_kf_idx = np.argmin(np.abs(ee_data[:, 0]-kf_t))
                perception_kf_idx = np.argmin(np.abs(perception_data[:, 0]-kf_t))

                # [1:] since we don't want time here
                perception_kf[i] = perception_data[perception_kf_idx][1:]
                ee_kf[i] = ee_data[ee_kf_idx][1:]

            perception_kf = np.concatenate((first_perception, perception_kf, last_perception), axis=0)
            ee_kf = np.concatenate((first_ee, ee_kf, last_ee), axis=0)

            self.perception_kf_data.append(perception_kf)
            self.ee_kf_data.append(ee_kf)

        # Learn pca from data
        self.pca.fit(np.concatenate(self.perception_kf_data, axis=0))
        print "PCA Explained variance:", np.sum(self.pca.explained_variance_ratio_)

        self.latent_perception_kf = list(map(self.pca.transform, self.perception_kf_data))

        self.n_kf = np.floor(np.mean(map(len, self.ee_kf_data)))
        print "Avg. # of KF: ", self.n_kf

        # Learn goal and action models
        self.goal_model = HMMGoalModel(self.latent_perception_kf)
        self.action_model = HMMES(self.ee_kf_data, self.n_kf, self.n_offspring)

        # Learn dense rewards
        self.s2d = Sparse2Dense(self.goal_model)

        self.avg_dur = np.floor(np.mean(self.durations))
        print "Average motion duration: ", self.avg_dur

    def generate_rollout(self):
        return self.action_model.generate_rollout(1.0, duration=self.avg_dur)

    def remove_rollout(self):
        self.action_model.remove_rollout()

    def update(self, per_seq):
        if per_seq.shape[-1] != self.perception_dim:
            per_seq = self.pca.transform(per_seq)

        ret = self.s2d.get_expected_return(per_seq)
        self.action_model.update(ret)
        is_success = self.goal_model.is_success(per_seq)

        return is_success, ret
