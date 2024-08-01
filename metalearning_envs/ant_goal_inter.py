import numpy as np
from .ant_multitask_base import MultitaskAntEnv
from . import register_env
import random

# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal-inter')
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, n_tasks=2, restricted_train_set=False,task_mode = 'forward_backward'):
        self.restricted_train_set = restricted_train_set
        self.task_mode = task_mode
        super(AntGoalEnv, self).__init__({}, n_tasks)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        goal_train1 = []
        for i in range(num_tasks):
            prob = random.random()  # np.random.uniform()
            if prob < 4.0 / 15.0:
                r = random.random() ** 0.5  # [0, 1]
            else:
                # r = random.random() * 0.5 + 2.5  # [2.5, 3.0]
                r = (random.random() * 2.75 + 6.25) ** 0.5
            theta = random.random() * 2 * np.pi  # [0.0, 2pi]
            goal_train1.append([r * np.cos(theta), r * np.sin(theta)])

        goal_train2 = [[1.75, 0], [0, 1.75], [-1.75, 0], [0, -1.75]]

        goals = goal_train1 + goal_train2


        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    @staticmethod
    def termination_fn(obs, act, next_obs):

        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        done = np.array([False]).repeat(len(obs))
        done = done[:, None]
        return done