import time
import os
import inspect
import glob
import yaml
import importlib

import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
from gym.envs.registration import load

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.policies import FeedForwardPolicy as BasePolicy, CnnPolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.bench import Monitor
from stable_baselines import logger
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, DDPG, TRPO, SAC, TD3
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, \
    VecFrameStack, SubprocVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import set_global_seeds
import utils.mobilenet_v1 as mobilenet_v1

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm, ortho_init
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.input import observation_input
import utils.channel_net_ops as ops
slim = tf.contrib.slim


init_scale=1.0
def separable_conv(input_tensor, scope, *, n_filters, filter_size, stride,
         pad='VALID', channel_multiplier=1, init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 2d convolutional layer for TensorFlow
    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
    or the height and width of kernel filter if the input is a list or tuple
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_input, n_filters]
    with tf.variable_scope(scope):
        depthwise_filter = tf.get_variable(shape=(filter_height, filter_width, n_input, channel_multiplier), name="deptwise_filter", initializer=ortho_init(init_scale))
        pointwise_filter = tf.get_variable(shape=[1, 1, channel_multiplier * n_input, n_filters], name="pointwise_filter", initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)

        output = tf.nn.separable_conv2d(
            input_tensor,
            depthwise_filter,
            pointwise_filter,
            strides=strides,
            padding=pad,
            data_format=data_format
        )
        return bias + output 



def Depthwise(x, in_channels, stride=1, is_training=True, scope='depthwise', kernel_size=3):
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', (kernel_size, kernel_size, in_channels, 1),
                            initializer=ortho_init(init_scale))
        x = tf.nn.depthwise_conv2d(x, w, (1, 1, stride, stride), 'SAME',
                                   data_format='NCHW')
        x = tf.nn.relu(x)
        return x


def get_mask(in_channels, kernel_size=3):
    in_channels = int(in_channels)
    mask = np.zeros((kernel_size, kernel_size, in_channels, in_channels))
    for _ in range(in_channels):
        mask[:, :, _, _] = 1.
    return mask


def Conv3x3(x, in_channels, out_channels, stride=1, is_training=True,
        scope='convolution', kernel_size=3):
    with tf.variable_scope(scope):
        w = tf.get_variable('weight', shape=(kernel_size, kernel_size, in_channels, out_channels),
                            initializer=ortho_init(init_scale))
        x = tf.nn.conv2d(x, w, (1, 1, stride, stride), 'SAME',
                         data_format='NCHW')
        x = tf.nn.relu(x)
        return x

def DiagonalwiseRefactorization(x, in_channels, stride=1, groups=4,
        is_training=True, scope='depthwise', kernel_size=3):
    with tf.variable_scope(scope):
        channels = int(in_channels / groups)
        mask = tf.constant(get_mask(channels, kernel_size).tolist(), dtype=tf.float32,
                           shape=(kernel_size, kernel_size, channels, channels))
        splitw = [
            tf.get_variable('weights_%d' % _, (kernel_size, kernel_size, channels, channels),
                            initializer=ortho_init(init_scale))
            for _ in range(groups)
        ]
        splitw = [tf.multiply(w, mask) for w in splitw]
        splitx = tf.split(x, groups, 1)
        splitx = [tf.nn.conv2d(x, w, (1, 1, stride, stride), 'SAME',
                               data_format='NCHW')
                  for x, w in zip(splitx, splitw)]
        x = tf.concat(splitx, 1)
        x = tf.nn.relu(x)
        return x

def Pointwise(x, in_channels, out_channels, is_training=True,
        scope='pointwise'):
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', (1, 1, in_channels, out_channels),
                            initializer=ortho_init(init_scale))
        x = tf.nn.conv2d(x, w, (1, 1, 1, 1), 'SAME', data_format='NCHW')
        x = tf.nn.relu(x)
        return x

def Separable(x, in_channels, out_channels, stride=1, is_training=True,
        scope='separable', kernel_size=3):
    with tf.variable_scope(scope):
        # Diagonalwise Refactorization
        # groups = in_channels
        # groups = 16
        groups = int(max(in_channels / 32, 1))
        x = DiagonalwiseRefactorization(x, in_channels, stride, groups,
                                        is_training, 'depthwise', kernel_size=kernel_size)

        # Specialized Kernel
        # x = Depthwise(x, in_channels, stride, is_training, 'depthwise')

        # Standard Convolution
        # x = Conv3x3(x, in_channels, in_channels, stride, is_training,
                    # 'convolution')
        x = Pointwise(x, in_channels, out_channels, is_training, 'pointwise')
        return x

def lower_flop_nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    

def mobilenet_cnn(images, **kwargs):
    activ = tf.nn.relu
    #TODO: try this: https://github.com/HongyangGao/ChannelNets/blob/master/model.py
    cur_out_num = 32
    data_format = kwargs.get('data_format', 'NHWC')
    outs = ops.conv2d(images, cur_out_num, (3, 3), 'conv_s', train=False,
            stride=2, act_fn=None, data_format=data_format)
    cur_outs = ops.dw_block(  # 112 * 112 * 64
        outs, cur_out_num, 1, 'conv_1_0', 1.0,
        False, data_format=data_format)
    outs = tf.concat([outs, cur_outs], axis=-1, name='add0')
    cur_out_num *= 2
    outs = ops.dw_block(  # 56 * 56 * 128
        outs, cur_out_num, 2, 'conv_1_1', 1.0,
        False, data_format=data_format)
    cur_outs = ops.dw_block(  # 56 * 56 * 128
        outs, cur_out_num, 1, 'conv_1_2', 1.0,
        False, data_format=data_format)
    outs = tf.concat([outs, cur_outs], axis=-1, name='add1')
    outs = ops.dw_block(  # 7 * 7 * 1024
            outs, cur_out_num, 1, 'conv_3_1', 1.0,
            False, 
            data_format=data_format)
    layer_3 = conv_to_fc(outs)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

    

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3
}


# ================== Custom Policies =================

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                              layers=[64],
                                              layer_norm=True,
                                              feature_extraction="mlp")



class CustomLowerFlopCnnPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomLowerFlopCnnPolicy, self).__init__(*args, **kwargs, cnn_extractor=lower_flop_nature_cnn)


class SmallMobileNetCnnPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super(SmallMobileNetCnnPolicy, self).__init__(*args, **kwargs, cnn_extractor=mobilenet_cnn)


class CustomMlpPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs,
                                              layers=[16],
                                              feature_extraction="mlp")


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256],
                                              feature_extraction="mlp")


register_policy('CustomSACPolicy', CustomSACPolicy)
register_policy('CustomDQNPolicy', CustomDQNPolicy)
register_policy('SmallMobileNetCnnPolicy', SmallMobileNetCnnPolicy)
register_policy('CustomLowerFlopCnnPolicy', CustomLowerFlopCnnPolicy)
register_policy('CustomMlpPolicy', CustomMlpPolicy)


def flatten_dict_observations(env):
    assert isinstance(env.observation_space, gym.spaces.Dict)
    keys = env.observation_space.spaces.keys()
    return gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))


def get_wrapper_class(hyperparams):
    """
    Get a Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    :param hyperparams: (dict)
    :return: a subclass of gym.Wrapper (class object) you can use to
             create another Gym env giving an original env.
    """

    def get_module_name(fullname):
        return '.'.join(wrapper_name.split('.')[:-1])

    def get_class_name(fullname):
        return wrapper_name.split('.')[-1]

    if 'env_wrapper' in hyperparams.keys():
        wrapper_name = hyperparams.get('env_wrapper')
        wrapper_module = importlib.import_module(get_module_name(wrapper_name))
        return getattr(wrapper_module, get_class_name(wrapper_name))
    else:
        return None


def make_env(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    :param wrapper: (type) a subclass of gym.Wrapper to wrap the original
                    env with
    """
    if log_dir is None and log_dir != '':
        log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        set_global_seeds(seed + rank)
        env = gym.make(env_id)

        # Dict observation space is currently not supported.
        # https://github.com/hill-a/stable-baselines/issues/321
        # We allow a Gym env wrapper (a subclass of gym.Wrapper)
        if wrapper_class:
            env = wrapper_class(env)

        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        return env

    return _init


def create_test_env(env_id, n_envs=1, is_atari=False,
                    stats_path=None, seed=0,
                    log_dir='', should_render=True, hyperparams=None):
    """
    Create environment for testing a trained agent

    :param env_id: (str)
    :param n_envs: (int) number of processes
    :param is_atari: (bool)
    :param stats_path: (str) path to folder containing saved running averaged
    :param seed: (int) Seed for random number generator
    :param log_dir: (str) Where to log rewards
    :param should_render: (bool) For Pybullet env, display the GUI
    :param env_wrapper: (type) A subclass of gym.Wrapper to wrap the original
                        env with
    :param hyperparams: (dict) Additional hyperparams (ex: n_stack)
    :return: (gym.Env)
    """
    # HACK to save logs
    if log_dir is not None:
        os.environ["OPENAI_LOG_FORMAT"] = 'csv'
        os.environ["OPENAI_LOGDIR"] = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        logger.configure()

    # Create the environment and wrap it if necessary
    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    if is_atari:
        print("Using Atari wrapper")
        env = make_atari_env(env_id, num_env=n_envs, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif n_envs > 1:
        # start_method = 'spawn' for thread safe
        env = SubprocVecEnv([make_env(env_id, i, seed, log_dir, wrapper_class=env_wrapper) for i in range(n_envs)])
    # Pybullet envs does not follow gym.render() interface
    elif "Bullet" in env_id:
        spec = gym.envs.registry.env_specs[env_id]
        try:
            class_ = load(spec.entry_point)
        except AttributeError:
            # Backward compatibility with gym
            class_ = load(spec._entry_point)
        # HACK: force SubprocVecEnv for Bullet env that does not
        # have a render argument
        render_name = None
        use_subproc = 'renders' not in inspect.getfullargspec(class_.__init__).args
        if not use_subproc:
            render_name = 'renders'
        # Dev branch of pybullet
        # use_subproc = use_subproc and 'render' not in inspect.getfullargspec(class_.__init__).args
        # if not use_subproc and render_name is None:
        #     render_name = 'render'

        # Create the env, with the original kwargs, and the new ones overriding them if needed
        def _init():
            # TODO: fix for pybullet locomotion envs
            env = class_(**{**spec._kwargs}, **{render_name: should_render})
            env.seed(0)
            if log_dir is not None:
                env = Monitor(env, os.path.join(log_dir, "0"), allow_early_resets=True)
            return env

        if use_subproc:
            env = SubprocVecEnv([make_env(env_id, 0, seed, log_dir, wrapper_class=env_wrapper)])
        else:
            env = DummyVecEnv([_init])
    else:
        env = DummyVecEnv([make_env(env_id, 0, seed, log_dir, wrapper_class=env_wrapper)])

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams['normalize']:
            print("Loading running average")
            print("with params: {}".format(hyperparams['normalize_kwargs']))
            env = VecNormalize(env, training=False, **hyperparams['normalize_kwargs'])
            env.load_running_average(stats_path)

        n_stack = hyperparams.get('frame_stack', 0)
        if n_stack > 0:
            print("Stacking {} frames".format(n_stack))
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def get_trained_models(log_folder):
    """
    :param log_folder: (str) Root log folder
    :return: (dict) Dict representing the trained agent
    """
    algos = os.listdir(log_folder)
    trained_models = {}
    for algo in algos:
        for env_id in glob.glob('{}/{}/*.pkl'.format(log_folder, algo)):
            # Retrieve env name
            env_id = env_id.split('/')[-1].split('.pkl')[0]
            trained_models['{}-{}'.format(algo, env_id)] = (algo, env_id)
    return trained_models


def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_saved_hyperparams(stats_path, norm_reward=False, test_mode=False):
    """
    :param stats_path: (str)
    :param norm_reward: (bool)
    :param test_mode: (bool)
    :return: (dict, str)
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, 'config.yml')
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, 'config.yml'), 'r') as f:
                hyperparams = yaml.load(f)
            hyperparams['normalize'] = hyperparams.get('normalize', False)
        else:
            obs_rms_path = os.path.join(stats_path, 'obs_rms.pkl')
            hyperparams['normalize'] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams['normalize']:
            if isinstance(hyperparams['normalize'], str):
                normalize_kwargs = eval(hyperparams['normalize'])
                if test_mode:
                    normalize_kwargs['norm_reward'] = norm_reward
            else:
                normalize_kwargs = {'norm_obs': hyperparams['normalize'], 'norm_reward': norm_reward}
            hyperparams['normalize_kwargs'] = normalize_kwargs
    return hyperparams, stats_path
