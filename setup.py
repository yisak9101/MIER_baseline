import tensorflow as tf
from metalearning_envs import ENVS
from metalearning_envs.wrappers import NormalizedBoxEnv
from softlearning.value_functions import vanilla
from softlearning.value_functions.utils import create_double_value_function
from softlearning.policies.gaussian_policy import FeedforwardGaussianPolicy
from softlearning.policies.uniform_policy import UniformPolicy
from softlearning.misc.utils import set_seed, initialize_tf_variables
from softlearning_sac import SAC

from models.bnn_contextual import BNN as BNN_contextual
from models.fake_env_contextual import FakeEnv as FakeEnvContextual

from softlearning_sampler import ContextConditionedSimpleSampler as SoftlearningSAC_ContextConditionedSimpleSampler
from data import ReplayBuffer,  MultiTaskReplayBuffer
from misc_utils import TensorBoardLogger, parse_network_arch, load_data, get_sep_model_hyperparams, set_random_seed

import metaworld
import numpy as np

def setup_sess(self):

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)
    self.sess = tf.keras.backend.get_session()

def setup(self):

    set_random_seed(self.seed)

    if "ml" in self.env_name:
        if "reach" in self.env_name:
            ml = metaworld.ML1(seed=self.seed, env_name='reach-v2')
            goal_low = np.array([-0.10, 0.80, 0.05])
            goal_high = np.array([0.10, 0.90, 0.30])
        elif "push" in self.env_name:
            ml = metaworld.ML1(seed=self.seed, env_name='push-v2')
            goal_low = np.array([-0.10, 0.80, 0.01])
            goal_high = np.array([0.10, 0.90, 0.02])
        
        train_env_name_list = [name for name, _ in ml.train_classes.items()]
        train_env_cls_list = [env_cls() for _, env_cls in ml.train_classes.items()]
        test_env_name_list = [name for name, _ in ml.test_classes.items()]
        test_env_cls_list = [env_cls() for _, env_cls in ml.test_classes.items()]

        ml_train_tasks = ml.train_tasks
        ml_test_tasks = ml.test_tasks
        ml_total_tasks = ml_train_tasks + ml_test_tasks

        x_distance, y_distance, z_distance = np.round(np.abs(goal_high - goal_low), 4) / 5  # 0.04 0.02 0.05
        inner1_bound_low = goal_low + 2 * np.array([x_distance, y_distance, z_distance])
        inner1_bound_high = goal_high - 2 * np.array([x_distance, y_distance, z_distance])
        inner2_bound_low = goal_low + np.array([x_distance, y_distance, z_distance])
        inner2_bound_high = goal_high - np.array([x_distance, y_distance, z_distance])

        size = inner2_bound_high - inner2_bound_low
        part_size = size / 3
        centers = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    center = inner2_bound_low + (np.array([i, j, k]) * part_size) + (part_size / 2)
                    centers.append(center)
        centers = np.stack(centers)

        """total_area"""
        total_area = np.random.uniform(low=goal_low, high=goal_high, size=(20000, 3))

        """inner1"""
        inner1_area_indices = (
                (total_area[:, 0] < inner1_bound_high[0]) & (total_area[:, 0] > inner1_bound_low[0]) &
                (total_area[:, 1] < inner1_bound_high[1]) & (total_area[:, 1] > inner1_bound_low[1]) &
                (total_area[:, 2] < inner1_bound_high[2]) & (total_area[:, 2] > inner1_bound_low[2])
        )

        inner1_center_indices = (
                (centers[:, 0] < inner1_bound_high[0]) & (centers[:, 0] > inner1_bound_low[0]) &
                (centers[:, 1] < inner1_bound_high[1]) & (centers[:, 1] > inner1_bound_low[1]) &
                (centers[:, 2] < inner1_bound_high[2]) & (centers[:, 2] > inner1_bound_low[2])
        )
        centers2 = centers[[not a for a in inner1_center_indices]]

        """ inner2 """
        inner2_area_indices = (
                (total_area[:, 0] < inner2_bound_low[0]) | (total_area[:, 0] > inner2_bound_high[0]) |
                (total_area[:, 1] < inner2_bound_low[1]) | (total_area[:, 1] > inner2_bound_high[1]) |
                (total_area[:, 2] < inner2_bound_low[2]) | (total_area[:, 2] > inner2_bound_high[2])
        )
        ml1_inter_tasks_without_center = total_area[inner2_area_indices][:self.n_train_tasks, :]
        ml1_inter_test_points_without_center = centers.copy()

        ml1_inter_tasks_with_center_in1 = total_area[inner1_area_indices][:5, :]
        ml1_inter_tasks_with_center_in2 = total_area[inner2_area_indices][:self.n_train_tasks - 5, :]
        ml1_inter_tasks_with_center = np.concatenate([ml1_inter_tasks_with_center_in1, ml1_inter_tasks_with_center_in2])
        ml1_inter_test_points_with_center = centers2.copy()


        """ extra """
        extra_area_indices = (
                (total_area[:, 0] < inner2_bound_high[0]) & (total_area[:, 0] > inner2_bound_low[0]) &
                (total_area[:, 1] < inner2_bound_high[1]) & (total_area[:, 1] > inner2_bound_low[1]) &
                (total_area[:, 2] < inner2_bound_high[2]) & (total_area[:, 2] > inner2_bound_low[2])
        )
        ml1_extra_tasks = total_area[extra_area_indices][:self.n_train_tasks, :]

        x_centers = np.linspace(goal_low[0] + x_distance / 2, goal_high[0] - x_distance / 2, 5)
        y_centers = np.linspace(goal_low[1] + y_distance / 2, goal_high[1] - y_distance / 2, 5)
        z_centers = np.linspace(goal_low[2] + z_distance / 2, goal_high[2] - z_distance / 2, 5)
        total_centers = np.array(np.meshgrid(x_centers, y_centers, z_centers)).T.reshape(-1, 3)
        filtered_centers = total_centers[
            (total_centers[:, 0] < inner2_bound_low[0]) | (total_centers[:, 0] > inner2_bound_high[0]) |
            (total_centers[:, 1] < inner2_bound_low[1]) | (total_centers[:, 1] > inner2_bound_high[1]) |
            (total_centers[:, 2] < inner2_bound_low[2]) | (total_centers[:, 2] > inner2_bound_high[2])
            ]
        ml1_extra_test_points = filtered_centers.copy()
        """"""



        if 'ood' not in self.env_name:
            train_envs = []
            for i in range(len(train_env_name_list)):
                train_env_name = train_env_name_list[i]
                env_cls = train_env_cls_list[i]
                for j in range(50):  # 50
                    train_envs.append({"ml_env_name": train_env_name,
                                        "env_cls": env_cls,
                                        "sub_task_idx": j,
                                        "target_pos": None} )
            eval_envs = []
            for i in range(len(test_env_name_list)):  # 1
                eval_env_name = test_env_name_list[i]
                env_cls = test_env_cls_list[i]
                for j in range(50):  # 7
                    eval_envs.append({"ml_env_name": eval_env_name,
                                        "env_cls": env_cls,
                                        "sub_task_idx": j + 50,
                                        "target_pos": None })
        elif "ood-inter-without-center" in self.env_name:
            train_envs = []
            for i in range(len(train_env_name_list)):
                train_env_name = train_env_name_list[i]
                env_cls = train_env_cls_list[i]
                for j in range(self.n_train_tasks):  # 20
                    train_envs.append({"ml_env_name": train_env_name,
                                    "env_cls": env_cls,
                                    "sub_task_idx": j,
                                    "target_pos": ml1_inter_tasks_without_center[j]})
            eval_envs = []
            for i in range(len(test_env_name_list)):  # 1
                eval_env_name = test_env_name_list[i]
                env_cls = test_env_cls_list[i]
                for j in range(self.n_val_tasks):  # 7
                    eval_envs.append({"ml_env_name": eval_env_name,
                                    "env_cls": env_cls,
                                    "sub_task_idx": j + 50,
                                    "target_pos": ml1_inter_test_points_without_center[j] })
        elif "ood-inter-with-center" in self.env_name:
            train_envs = []
            for i in range(len(train_env_name_list)):
                train_env_name = train_env_name_list[i]
                env_cls = train_env_cls_list[i]
                for j in range(self.n_train_tasks):  # 20
                    train_envs.append({"ml_env_name": train_env_name,
                                    "env_cls": env_cls,
                                    "sub_task_idx": j,
                                    "target_pos": ml1_inter_tasks_with_center[j]})
            eval_envs = []
            for i in range(len(test_env_name_list)):  # 1
                eval_env_name = test_env_name_list[i]
                env_cls = test_env_cls_list[i]
                for j in range(self.n_val_tasks):  # 7
                    eval_envs.append({"ml_env_name": eval_env_name,
                                    "env_cls": env_cls,
                                    "sub_task_idx": j + 50,
                                    "target_pos": ml1_inter_test_points_with_center[j] })
        elif "ood-extra" in self.env_name:
            train_envs = []
            for i in range(len(train_env_name_list)):
                train_env_name = train_env_name_list[i]
                env_cls = train_env_cls_list[i]
                for j in range(self.n_train_tasks):  # 20
                    train_envs.append({"ml_env_name": train_env_name,
                                    "env_cls": env_cls,
                                    "sub_task_idx": j,
                                    "target_pos": ml1_extra_tasks[j]})
            eval_envs = []
            for i in range(len(test_env_name_list)):  # 1
                eval_env_name = test_env_name_list[i]
                env_cls = test_env_cls_list[i]
                for j in range(self.n_val_tasks):  # 7
                    eval_envs.append({"ml_env_name": eval_env_name,
                                    "env_cls": env_cls,
                                    "sub_task_idx": j,
                                    "target_pos": ml1_extra_test_points[j] }) 

        total_envs = train_envs + eval_envs
        self.ml_env_infos = [
                train_env_name_list,
                train_env_cls_list,
                test_env_name_list,
                test_env_cls_list,
                ml_total_tasks,
                total_envs,
            ]

        init_task_idx = 0
        _env_name = total_envs[init_task_idx]["ml_env_name"]
        _subtask_idx = total_envs[init_task_idx]["sub_task_idx"]
        self.env = total_envs[init_task_idx]["env_cls"]
        self.env.set_task([_task for _task in ml_total_tasks if _task.env_name == _env_name][_subtask_idx])
        # tasks, total_tasks_dict_list = list(range(len(total_envs))), None
        self.env_params['n_tasks'] = self.n_train_tasks + self.n_val_tasks



    else:
        self.env_params['n_tasks'] = self.n_train_tasks + self.n_val_tasks
        self.env = NormalizedBoxEnv(ENVS[self.env_name](**self.env_params))
        self.ml_env_infos = None
    
    print("$$"*100)
    print("max_path_length", self.max_path_length)
    print("$$"*100)

    #set_gpu_mode(self.device == 'cuda')
    self.obs_dim = obs_dim = int(self.env.observation_space.shape[0])
    self.act_dim = act_dim = int(self.env.action_space.shape[0])
    context_dim = self.context_dim

    setup_sess(self)
    self.logger = TensorBoardLogger(self.log_dir)

    ############## end to end load path #########################
    if self.load_path_prefix !=None and \
        ((self.run_mode == 'train' and self.continue_training_from_loaded_model) or self.run_mode == 'extrapolate'):
        load_path = self.load_path_prefix + "seed-"+str(self.seed) + "/Itr_"+str(self.load_model_itr)+"/"

        if self.joint_state_reward_model:
            self.model_load_path = load_path +'model.pkl'
        else:
            self.state_model_load_path = load_path + 'state_model.pkl'
            self.reward_model_load_path = load_path + 'reward_model.pkl'

        self.sac_load_path = load_path+'sac.pkl'

        self.replay_buffer_load_path = load_path + 'replay_buffer.pkl'
        #self.pre_adapt_replay_buffer_load_path = load_path + 'pre_adapt_replay_buffer.pkl'

        if self.run_mode == "extrapolate":
            self.cross_task_data_load_path = load_path + 'replay_buffer.pkl'

    ############################## Setup Model #############################

    if self.joint_state_reward_model:

        self.model = BNN_contextual(self.sess, obs_dim, act_dim, self.context_dim,
                                        model_hyperparams=self.model_hyperparams)
        if self.model_load_path:
            self.model.load_model(self.model_load_path)

        termination_fn = self.env.termination_fn if hasattr(self.env, 'termination_fn') else None
        self.fake_env = FakeEnvContextual({'model': self.model}, termination_fn)

    else:
        assert self.meta_learn_state_dynamics == False
        state_model_hyperparams, reward_model_hyperparams = get_sep_model_hyperparams(
                                    self.model_hyperparams, self.meta_learn_state_dynamics, self.meta_learn_reward)


        self.state_model = BNN_contextual(self.sess, obs_dim, act_dim, self.context_dim,
                                   model_hyperparams= state_model_hyperparams)

        self.model = self.reward_model = BNN_contextual(self.sess, obs_dim, act_dim, self.context_dim,
                                    model_hyperparams= reward_model_hyperparams)

        if self.state_model_load_path:
            self.state_model.load_model(self.state_model_load_path)

        if self.reward_model_load_path:
            self.reward_model.load_model(self.reward_model_load_path)

        termination_fn = self.env.termination_fn if hasattr(self.env, 'termination_fn') else None
        self.fake_env = FakeEnvContextual({'state_model': self.state_model, 'reward_model': self.reward_model},
                               termination_fn, joint_state_reward_model=False)

    ############### Setup SAC #############################

    with tf.variable_scope("softlearning"):
        Qs = create_double_value_function(
            vanilla.create_feedforward_Q_function,
            observation_shape=(obs_dim + context_dim,),
            action_shape=(act_dim,),
            hidden_layer_sizes=parse_network_arch(self.critic_nn_arch)
        )

        policy = FeedforwardGaussianPolicy(
            input_shapes=((obs_dim + context_dim,),),
            output_shape=(act_dim,),
            hidden_layer_sizes=parse_network_arch(self.actor_nn_arch),
            squash=True
        )

        initial_exploration_policy = UniformPolicy(
            input_shapes=((obs_dim + context_dim,),),
            output_shape=(act_dim,))

        self.sac_trainer = SAC(
            observation_shape=(obs_dim + context_dim,),
            action_shape=(act_dim,),
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            session=self.sess,
            discount=self.sac_hyperparams["discount_factor"],
            tau=self.sac_hyperparams["target_update_rate"],
            reward_scale=self.sac_hyperparams["sac_reward_scale"],
            reparameterize=True,
            lr=self.sac_hyperparams["actor_learning_rate"],
            target_update_interval=self.sac_hyperparams["target_update_interval"],
            target_entropy=self.sac_hyperparams["target_entropy"]
        )

        initialize_tf_variables(self.sess, only_uninitialized=True)
        if self.sac_load_path:

            self.sac_trainer.load_model(self.sac_load_path)

        self.sampler = SoftlearningSAC_ContextConditionedSimpleSampler(
            env=self.env,
            max_path_length=self.max_path_length,
            policy=self.sac_trainer._policy,
            exploration_policy=initial_exploration_policy,
            ml_env_infos=self.ml_env_infos
        )

    ######################## DataSet, Buffer Setup #####################################################
    if self.run_mode == "train":

        self.replay_buffer = MultiTaskReplayBuffer(self.replay_buffer_max_sample_size, self.n_train_tasks)
        self.pre_adapt_replay_buffer = MultiTaskReplayBuffer(self.replay_buffer_max_sample_size, self.n_train_tasks)
        self.model_buffer = MultiTaskReplayBuffer(self.replay_buffer_max_sample_size, self.n_train_tasks)

        if self.replay_buffer_load_path != None:
            self.replay_buffer.load_data(self.replay_buffer_load_path)

        if self.pre_adapt_replay_buffer_load_path != None:
            self.pre_adapt_replay_buffer.load_data(self.pre_adapt_replay_buffer_load_path)

    elif self.run_mode == "extrapolate":

        if self.cross_task_data_load_path != None:
            self.cross_task_data, self.cross_task_data_size = load_data(self.cross_task_data_load_path, True)

        self.replay_buffer = ReplayBuffer(self.replay_buffer_max_sample_size)
        self.model_buffer = ReplayBuffer(self.replay_buffer_max_sample_size)
