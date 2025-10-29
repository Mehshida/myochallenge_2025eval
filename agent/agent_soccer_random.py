import os
import pickle
import time

import copy
import numpy as np

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym
from stable_baselines3 import PPO

def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)



class EnvShell:

    def __init__(self, stub):

        action_len = unpack_for_grpc(
            stub.get_action_space(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
            ).SerializedEntity
        )

        obs_len = unpack_for_grpc(
            stub.get_observation_space(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
            ).SerializedEntity
        )
        self.observation_space = gym.spaces.Box(shape=(obs_len,), high=1e6, low=-1e6)
        self.action_space = gym.spaces.Box(shape=(action_len,), high=1.0, low=0.0)
        print("Action Space", self.action_space)
        print("Observation Space", self.observation_space)
        # TODO case for remapping of [-1 1] -> [0 1]


class Policy:

    def __init__(self, env, model_path="ppo_myo_soccer.zip"):
        self.action_space = env.action_space
        self.model = PPO.load(model_path)
        print(f"Loaded PPO policy from {model_path}")

    def __call__(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action

custom_obs_keys = [      
    'internal_qpos',
    'internal_qvel',
    'grf',
    'torso_angle',
    'pelvis_angle',
    'model_root_pos',
    'model_root_vel',
    'muscle_length',
    'muscle_velocity',
    'muscle_force',
    'ball_pos',
    'goal_bounds',
]


time.sleep(60) # DO NOT REMOVE. Required for EvalAI processing

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    channel = grpc.insecure_channel("environment:8086")
else:
    channel = grpc.insecure_channel("localhost:8086")

stub = evaluation_pb2_grpc.EnvironmentStub(channel)
env_shell = EnvShell(stub)
# policy = deprl.load_baseline(env_shell)
policy = Policy(env_shell)

# Preparing the dictionary of environment keys
custom_environment_varibles = {'obs_keys':custom_obs_keys, 'normalize_act':True}

# Setting the keys to the environment
stub.set_environment_keys(
    evaluation_pb2.Package(SerializedEntity=pack_for_grpc(custom_environment_varibles))
).SerializedEntity

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"LOCO-SOCCER: Start Resetting the environment and get 1st obs of iter {trial}")

    obs = unpack_for_grpc(
        stub.reset(
            evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
        ).SerializedEntity
    )

    # Handle new Gymnasium API that returns (obs, info) tuple
    if isinstance(obs, tuple):
        obs = obs[0]

    counter = 0

    while not flag_trial:
    #for t in range(1000):
        print(
            f"Trial: {trial}, Iteration: {counter} flag_trial: {flag_trial} flat_completed: {flat_completed}"
        )

        action = policy(obs)
        base = unpack_for_grpc(
            stub.act_on_environment(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(action))
            ).SerializedEntity
        )
        # print(f" \t \t after step: {base['feedback'][1:4]}")
        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]
        # print(
        #     f" \t \t after step: flag_trial: {flag_trial} flat_completed: {flat_completed}"
        # )
        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
