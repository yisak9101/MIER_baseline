

import numpy as np
import random
from gym.envs.mujoco import AntEnv as AntEnv
from . import register_env



@register_env('ant-dir-2개')
class AntDir2Env(AntEnv):
    def __init__(self, n_train_tasks, n_eval_tasks, n_indistribution_tasks, eval_tasks_list, indistribution_tasks_list, tsne_tasks_list):
        self._task = {}
        self.tasks = self.sample_tasks(n_train_tasks, eval_tasks_list, indistribution_tasks_list, tsne_tasks_list)
        print("all tasks : ", self.tasks)
        self._goal = self.tasks[0]['goal']

        # xml_path = 'ant.xml'
        # super().__init__(xml_path, frame_skip=5, automatically_set_obs_and_action_space=True)
        super(AntDir2Env, self).__init__()

    def get_obs_dim(self):
        return int(np.prod(self._get_obs().shape))

    def sample_tasks(self, n_train_tasks, eval_tasks_list, indistribution_tasks_list, tsne_tasks_list):

        train_goal_dir = [0.0, 0.5 * np.pi]  # 2개
        eval_goal_dir  = [3 * np.pi / 4,    7 * np.pi / 4]   # 2개
        indis_goal_dir = [1 * np.pi / 4]
        tsne_goal_dir = [0.0, 1 * np.pi / 4, 0.5 * np.pi, 3 * np.pi / 4, 7 * np.pi / 4]
        goal_dirs = train_goal_dir + eval_goal_dir + indis_goal_dir + tsne_goal_dir

        tasks = [{'goal': goal_dir} for goal_dir in goal_dirs]

        return tasks

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))

        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2 ] /self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum( np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0

        # if self.done_flase:
        #     done = False  # not notdone
        # else:
        done = not notdone

        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )


    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat,
    #         self.sim.data.qvel.flat,
    #         # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
    #     ])
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector
        self.reset()