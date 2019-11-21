import argparse
from stable_baselines import logger
import difflib
import os
from collections import OrderedDict
from pprint import pprint
import warnings
import importlib

# For pybullet envs
warnings.filterwarnings("ignore")
import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
import yaml
try:
    import highway_env
except ImportError:
    highway_env = None
from mpi4py import MPI

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn

from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class
from utils.hyperparams_opt import hyperparam_optimization
from utils.noise import LinearNormalActionNoise
from experiment_impact_tracker.compute_tracker import ImpactTracker
from experiment_impact_tracker.utils import get_flop_count_tensorflow
from experiment_impact_tracker.cpu.common import assert_cpus_by_attributes
from experiment_impact_tracker.gpu.nvidia import assert_gpus_by_attributes 
import json
import time
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, nargs='+', default=["CartPole-v1"], help='environment ID(s)')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                        type=int)
    parser.add_argument('--evaluate-interval', help='Override log interval (default: -1, no change)', default=250000,
                        type=int)
    parser.add_argument('--hparam_file', type=str, help="the hyperparam file spec")
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
                        default='skopt', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
                        default='none', choices=['halving', 'median', 'none'])
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--cpu-only', action="store_true", default=False)
    parser.add_argument('--ignore-hardware', action="store_true", default=False)
    args = parser.parse_args()

    if not args.ignore_hardware:
        if not args.cpu_only:
            assert_gpus_by_attributes({ "name" : "GeForce GTX TITAN X"})
        assert_cpus_by_attributes({ "brand": "Intel(R) Xeon(R) CPU E5-2640 v3 @ 2.60GHz" }) 
   
    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_ids = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    for env_id in env_ids:
        # If the environment is not found, suggest the closest match
        if env_id not in registered_envs:
            try:
                closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
            except IndexError:
                closest_match = "'no close match found...'"
            raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .pkl file"

    rank = 0
    if MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            args.verbose = 0
            args.tensorboard_log = ''

    for env_id in env_ids:
        tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, env_id)
        os.environ["OPENAI_LOG_FORMAT"] = 'csv'
        os.environ["OPENAI_LOGDIR"] = os.path.abspath(tensorboard_log)
        logger.configure()

        tracker = ImpactTracker(tensorboard_log)

        tracker.launch_impact_monitor()

        is_atari = False
        if 'NoFrameskip' in env_id:
            is_atari = True

        print("=" * 10, env_id, "=" * 10)

        # Load hyperparameters from yaml file
        if args.hparam_file:
            hparam_file_name = args.hparam_file
        else:
            hparam_file_name = 'hyperparams/{}.yml'.format(args.algo)
        with open(hparam_file_name, 'r') as f:
            hyperparams_dict = yaml.load(f)
            if env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[env_id]
            elif is_atari:
                hyperparams = hyperparams_dict['atari']
            else:
                raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        algo_ = args.algo
        # HER is only a wrapper around an algo
        if args.algo == 'her':
            algo_ = saved_hyperparams['model_class']
            assert algo_ in {'sac', 'ddpg', 'dqn', 'td3'}, "{} is not compatible with HER".format(algo_)
            # Retrieve the model class
            hyperparams['model_class'] = ALGOS[saved_hyperparams['model_class']]

        if args.verbose > 0:
            pprint(saved_hyperparams)

        n_envs = hyperparams.get('n_envs', 1)

        if args.verbose > 0:
            print("Using {} environments".format(n_envs))

        # Create learning rate schedules for ppo2 and sac
        if algo_ in ["ppo2", "sac", "td3"]:
            for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
                if key not in hyperparams:
                    continue
                if isinstance(hyperparams[key], str):
                    schedule, initial_value = hyperparams[key].split('_')
                    initial_value = float(initial_value)
                    hyperparams[key] = linear_schedule(initial_value)
                elif isinstance(hyperparams[key], (float, int)):
                    # Negative value: ignore (ex: for clipping)
                    if hyperparams[key] < 0:
                        continue
                    hyperparams[key] = constfn(float(hyperparams[key]))
                else:
                    raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))

        # Should we overwrite the number of timesteps?
        if args.n_timesteps > 0:
            if args.verbose:
                print("Overwriting n_timesteps with n={}".format(args.n_timesteps))
            n_timesteps = args.n_timesteps
        else:
            n_timesteps = int(hyperparams['n_timesteps'])

        normalize = False
        normalize_kwargs = {}
        if 'normalize' in hyperparams.keys():
            normalize = hyperparams['normalize']
            if isinstance(normalize, str):
                normalize_kwargs = eval(normalize)
                normalize = True
            del hyperparams['normalize']

        if 'policy_kwargs' in hyperparams.keys():
            hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

        # Delete keys so the dict can be pass to the model constructor
        if 'n_envs' in hyperparams.keys():
            del hyperparams['n_envs']
        del hyperparams['n_timesteps']

        # obtain a class object from a wrapper name string in hyperparams
        # and delete the entry
        env_wrapper = get_wrapper_class(hyperparams)
        if 'env_wrapper' in hyperparams.keys():
            del hyperparams['env_wrapper']

        def create_env(n_envs, test=False):
            """
            Create the environment and wrap it if necessary
            :param n_envs: (int)
            :return: (gym.Env)
            """
            global hyperparams

            if is_atari:
                if args.verbose > 0:
                    print("Using Atari wrapper")
                env = make_atari_env(env_id, num_env=n_envs, seed=args.seed, wrapper_kwargs=dict(clip_rewards=(not test), episode_life=(not test)))
                # Frame-stacking with 4 frames
                env = VecFrameStack(env, n_stack=4)
            elif algo_ in ['dqn', 'ddpg']:
                if hyperparams.get('normalize', False):
                    print("WARNING: normalization not supported yet for DDPG/DQN")
                env = gym.make(env_id)
                env.seed(args.seed)
                if env_wrapper is not None:
                    env = env_wrapper(env)
            else:
                if n_envs == 1:
                    env = DummyVecEnv([make_env(env_id, 0, args.seed, wrapper_class=env_wrapper)])
                else:
                    # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
                    # On most env, SubprocVecEnv does not help and is quite memory hungry
                    env = DummyVecEnv([make_env(env_id, i, args.seed, wrapper_class=env_wrapper) for i in range(n_envs)])
                if normalize:
                    if args.verbose > 0:
                        if len(normalize_kwargs) > 0:
                            print("Normalization activated: {}".format(normalize_kwargs))
                        else:
                            print("Normalizing input and reward")
                    env = VecNormalize(env, **normalize_kwargs)
            # Optional Frame-stacking
            if hyperparams.get('frame_stack', False):
                n_stack = hyperparams['frame_stack']
                env = VecFrameStack(env, n_stack)
                print("Stacking {} frames".format(n_stack))
                del hyperparams['frame_stack']
            return env


        env = create_env(n_envs)
        # Stop env processes to free memory
        if args.optimize_hyperparameters and n_envs > 1:
            env.close()

        # Parse noise string for DDPG and SAC
        if algo_ in ['ddpg', 'sac', 'td3'] and hyperparams.get('noise_type') is not None:
            noise_type = hyperparams['noise_type'].strip()
            noise_std = hyperparams['noise_std']
            n_actions = env.action_space.shape[0]
            if 'adaptive-param' in noise_type:
                assert algo_ == 'ddpg', 'Parameter is not supported by SAC'
                hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                    desired_action_stddev=noise_std)
            elif 'normal' in noise_type:
                if 'lin' in noise_type:
                    hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                          sigma=noise_std * np.ones(n_actions),
                                                                          final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                          max_steps=n_timesteps)
                else:
                    hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                    sigma=noise_std * np.ones(n_actions))
            elif 'ornstein-uhlenbeck' in noise_type:
                hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                           sigma=noise_std * np.ones(n_actions))
            else:
                raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
            print("Applying {} noise with std {}".format(noise_type, noise_std))
            del hyperparams['noise_type']
            del hyperparams['noise_std']
            if 'noise_std_final' in hyperparams:
                del hyperparams['noise_std_final']

        if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
            # Continue training
            print("Loading pretrained agent")
            # Policy should not be changed
            del hyperparams['policy']

            model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                          tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

            exp_folder = args.trained_agent.split('.pkl')[0]
            if normalize:
                print("Loading saved running average")
                env.load_running_average(exp_folder)

        elif args.optimize_hyperparameters:

            if args.verbose > 0:
                print("Optimizing hyperparameters")


            def create_model(*_args, **kwargs):
                """
                Helper to create a model with different hyperparameters
                """
                return ALGOS[args.algo](env=create_env(n_envs), tensorboard_log=tensorboard_log,
                                        verbose=0, **kwargs)


            data_frame = hyperparam_optimization(args.algo, create_model, create_env, n_trials=args.n_trials,
                                                 n_timesteps=n_timesteps, hyperparams=hyperparams,
                                                 n_jobs=args.n_jobs, seed=args.seed,
                                                 sampler_method=args.sampler, pruner_method=args.pruner,
                                                 verbose=args.verbose)

            report_name = "report_{}_{}-trials-{}-{}-{}.csv".format(env_id, args.n_trials, n_timesteps,
                                                                    args.sampler, args.pruner)

            log_path = os.path.join(args.log_folder, args.algo, report_name)

            if args.verbose:
                print("Writing report to {}".format(log_path))

            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            data_frame.to_csv(log_path)
            exit()
        else:
            # Train an agent from scratch
            model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

        print("FLOP count {}".format(get_flop_count_tensorflow(model.graph)))

        model.test_env = create_env(25, test=True)

        kwargs = {}
        if args.log_interval > -1:
            kwargs = {'log_interval': args.log_interval}


        eval_output_filename = os.path.join(tensorboard_log, 'eval_test.csv')
        eval_csv_file = open(eval_output_filename, 'w', newline='')
        eval_csv_file.write(json.dumps(saved_hyperparams))
        eval_csv_file.write('\n')
        eval_csv_writer = csv.writer(eval_csv_file, delimiter=',')
        eval_csv_writer.writerow(['frames','total_time',
                                      'rmean','rmedian','rmin','rmax','rstd',
                                      'lmean','lmedian','lmin','lmax','lstd'])
        start_time = time.time()
        model.last_time_evaluated = 0

        def callback(_locals, _globals):
            self_ = _locals['self']
            # if we've reached the max timesteps run an evaluation no matter what, otherwise every n steps
            if "update" in _locals:
                final_update = _locals["update"] == (_locals["total_timesteps"] // self_.n_batch)
            else:
                final_update = self_.num_timesteps == (_locals["total_timesteps"] - 1)
            print("final update {}".format(final_update))
            if not final_update:
                if (self_.num_timesteps - self_.last_time_evaluated) < args.evaluate_interval:
                    return True

            self_.last_time_evaluated = self_.num_timesteps
            tracker.get_latest_info_and_check_for_errors()

            total_time = time.time() - start_time
            episode_returns = []
            lengths = []
            n_test_episodes = 25 
            n_episodes, episode_length, reward_sum = 0, 0, 0.0

            # Sync the obs rms if using vecnormalize
            # NOTE: this does not cover all the possible cases
            if isinstance(self_.test_env, VecNormalize):
                self_.test_env.obs_rms = deepcopy(self_.env.obs_rms)
                # Do not normalize reward
                self_.test_env.norm_reward = False

            width, height = 84, 84
            num_ales = n_test_episodes

            obs = self_.test_env.reset()

            lengths = np.zeros(num_ales, dtype=np.int32)
            rewards = np.zeros(num_ales, dtype=np.int32)
            all_done = np.zeros(num_ales, dtype=np.bool)
            not_done = np.ones(num_ales, dtype=np.int32)

            while not all_done.all():
                actions, _ = self_.predict(obs)

                obs, reward, done, info = self_.test_env.step(actions)
                
                done = np.array(done, dtype=np.bool_)

                obs = np.array(obs, dtype=np.float32)

                # update episodic reward counters
                lengths += not_done
                rewards += np.array(reward, dtype=np.int32) * not_done

                all_done |= done
                not_done[:] = np.array(all_done == False, dtype=np.int32)

            returns = rewards
            rmean = np.mean(returns)
            rmin = np.min(returns)
            rmax = np.max(returns)
            rstd = np.std(returns)
            lmean = np.mean(lengths)
            lmin = np.min(lengths)
            lstd = np.std(lengths)
            lmax = np.max(lengths)
            lmedian = np.median(lengths)
            rmedian = np.median(returns)

            eval_csv_writer.writerow([self_.num_timesteps, total_time, rmean, rmedian, rmin, rmax, rstd, lmean, lmedian, lmin, lmax, lstd])
            eval_csv_file.flush()

        model.learn(n_timesteps, **kwargs, callback=callback)

        # Save trained model
        log_path = "{}/{}/".format(args.log_folder, args.algo)
        save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id) + 1))
        params_path = "{}/{}".format(save_path, env_id)
        os.makedirs(params_path, exist_ok=True)

        # Only save worker of rank 0 when using mpi
        if rank == 0:
            print("Saving to {}".format(save_path))

            model.save("{}/{}".format(save_path, env_id))
            # Save hyperparams
            with open(os.path.join(params_path, 'config.yml'), 'w') as f:
                yaml.dump(saved_hyperparams, f)

            if normalize:
                # Unwrap
                if isinstance(env, VecFrameStack):
                    env = env.venv
                # Important: save the running average, for testing the agent we need that normalization
                env.save_running_average(params_path)




