from improve_klfd.utils import ls_dir_idxs
import os
import numpy as np
from sklearn.decomposition import PCA
from improve_klfd.goal_model import HMMGoalModel
from improve_klfd.rl import HMMES
from improve_klfd.s2d import Sparse2Dense
import pickle


class KFLFD(object):
    def __init__(self, gamma, n_offspring):
        self.n_offspring = n_offspring
        self.gamma = gamma
        self.perception_kf_data = []
        self.ee_kf_data = []

        self.durations = []
        self.success = []

    def learn_models(self):
        self.n_kf = int(np.floor(np.mean(map(len, self.ee_kf_data))))
        print "Avg. # of KF: ", self.n_kf

        # Learn goal and action models
        self.goal_model = HMMGoalModel(self.latent_perception_kf, n_states=self.n_kf)
        self.action_model = HMMES(self.ee_kf_data, self.n_kf, self.n_offspring, self.gamma)

        # Learn dense rewards
        self.s2d = Sparse2Dense(self.goal_model)

        self.avg_dur = np.floor(np.mean(self.durations))
        print "Average motion duration: ", self.avg_dur

    def from_loaded_data(self, action_kfs, goal_kfs, pca):
        self.pca = pca
        self.perception_dim = pca.n_components
        self.latent_perception_kf = goal_kfs[:, :, 1:]
        self.ee_kf_data = action_kfs[:, :, 1:]
        self.durations = action_kfs[:, -1, 0]
        self.learn_models()

    def load_from_dir(self, data_dir, perception_dim=8):
        self.demo_dirs = ls_dir_idxs(data_dir)
        self.pca = PCA(n_components=perception_dim)
        self.perception_dim = perception_dim

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
        self.learn_models()

    def generate_rollout(self):
            return self.action_model.generate_rollout(duration=self.avg_dur)

    def generate_motion(self):
        return self.action_model.generate_motion(duration=self.avg_dur)

    def remove_rollout(self):
        self.action_model.remove_rollout()

    def update(self, per_seq):
        if per_seq.shape[-1] != self.perception_dim:
            per_seq = self.pca.transform(per_seq)

        ret = self.s2d.get_expected_return(per_seq)
        updated = self.action_model.update(ret)
        is_success = self.goal_model.is_success(per_seq)
        self.success.append(is_success)

        if updated:
            self.success = []

        return is_success, ret

    def get_goal_info(self, per_seq):
        ret = self.s2d.get_expected_return(per_seq)
        is_success = self.goal_model.is_success(per_seq)

        return is_success, ret

    def save_models(self, dir):
        action_model_dir = os.path.join(dir, 'action_model.pk')
        goal_model_dir = os.path.join(dir, 'goal_model.pk')
        s2d_dir = os.path.join(dir, 's2d.pk')
        pca_dir = os.path.join(dir, 'pca.pk')

        pickle.dump(self.action_model, open(action_model_dir, 'wb'))
        pickle.dump(self.goal_model, open(goal_model_dir, 'wb'))
        pickle.dump(self.s2d, open(s2d_dir, 'wb'))
        pickle.dump(self.pca, open(pca_dir, 'wb'))

    def save_log(self, dir, episode):
        means_dir = os.path.join(dir, 'means_{}.csv'.format(episode))
        exp_means_dir = os.path.join(dir, 'exp_means_{}'.format(episode))
        covars_dir = os.path.join(dir, 'covars_{}.csv'.format(episode))
        rewards_dir = os.path.join(dir, 'rewards_{}.csv'.format(episode))
        success_dir = os.path.join(dir, 'success_{}'.format(episode))

        np.savetxt(rewards_dir, self.action_model.rewards, delimiter=',')
        np.save(exp_means_dir, self.action_model.exp_means)
        np.savetxt(means_dir, self.action_model.hmm.means, delimiter=',')
        np.save(covars_dir, self.action_model.hmm.covars)
        np.savetxt(success_dir, self.success, delimiter=',')