from rand_param_envs.gym.core import Env
from rand_param_envs.gym.envs.mujoco import MujocoEnv
import numpy as np
import random

class MetaEnv(Env):
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def log_diagnostics(self, paths, prefix):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        pass


class RandomEnv(MetaEnv, MujocoEnv):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """
    RAND_PARAMS = ['body_mass']
    # RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, log_scale_limit, file_name, *args, rand_params=RAND_PARAMS, **kwargs):
        MujocoEnv.__init__(self, file_name, 4)
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        self.log_scale_limit = log_scale_limit
        self.rand_params = rand_params
        self.save_parameters()

    def get_one_rand_params(self, eval_mode='train', value=0):
        new_params = {}

        mass_size_ = np.prod(self.model.body_mass.shape)

        if eval_mode == "train":
            prob = random.random()
            if prob >= 0.5:
                body_mass_multiplyers_ = random.uniform(0, 0.5)
            else:
                body_mass_multiplyers_ = random.uniform(3.0, 3.5)  # 3.0 - 0.5 = 2.5
            # print("body_mass_multiplyers_ in base2", body_mass_multiplyers_)
        elif eval_mode == "eval":
            body_mass_multiplyers_ = value

        else:
            body_mass_multiplyers_ = None

        body_mass_multiplyers = np.array([body_mass_multiplyers_ for _ in range(mass_size_)])
        body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
        body_mass_multiplyers = np.array(body_mass_multiplyers).reshape(self.model.body_mass.shape)
        new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

        return new_params, body_mass_multiplyers_

    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets_train = []
        """train_task"""  # [0,0.5] + [3,3.5]
        for _ in range(n_tasks):  #
            new_params, body_mass_multiplyers_ = self.get_one_rand_params(eval_mode='train')
            param_sets_train.append(new_params)

        param_sets_test = []
        test_task = list(np.linspace(0.75, 2.75, 5))
        """train_task"""  # [0,0.5] + [3,3.5]
        for task in test_task:  #
            new_params, body_mass_multiplyers_ = self.get_one_rand_params(eval_mode='eval', value=task)
            param_sets_test.append(new_params)

        param_sets = param_sets_train + param_sets_test
        return param_sets

    def set_task(self, task):
        for param, param_val in task.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            setattr(self.model, param, param_val)
        self.cur_params = task

    def get_task(self):
        return self.cur_params

    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.model.body_mass

        # body_inertia
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = self.model.dof_damping

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = self.model.geom_friction
        self.cur_params = self.init_params
