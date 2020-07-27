import argparse
import gym
import numpy as np

from agent import AgentDDQN
from atari_wrappers import make_atari, wrap_deepmind


def parse():
    """
        Command parser, choose:
            - Game
            - Training or testing
            - Rendering or not
            - Using prioritized replay memory or not
            - Using Dueling architecture or not
    """
    parser = argparse.ArgumentParser(description="Dueling DDQN Atari Game")
    parser.add_argument('--game', type=str, default='SpaceInvadersNoFrameskip-v4', help='Game environment name?')
    parser.add_argument('--mode', type=str, default='Training', help='Training/Testing mode?')
    parser.add_argument('--testing_path', type=str, default='model/game_4000000.h5', help='Where is stored the model?')
    parser.add_argument('--PER', action='store_true', help='Using Prioritized memory or not, True/False?')
    parser.add_argument('--dueling', action='store_true', help='Using Dueling architecture or not, True/False?')
    parser.add_argument('--render', action='store_true', help='Rendering the game or not, True/False?')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def start(args):
    # Frames will be preprocessed by the Atari Wrappers
    if args.mode == 'Training':
        env_name = args.game
        # Create the environment game with stacked frame
        env = make_atari(env_name)
        if "SpaceInvaders" in env_name:
            env = wrap_deepmind(env, frame_stack=True, scale=True, number_of_frame=3)
        else:
            env = wrap_deepmind(env, frame_stack=True, scale=True, number_of_frame=4)    
        env.seed(58)
        # Create DQQN agent to training it
        myagent = AgentDDQN(env, args)
        myagent.train_loop()

    #TODO add testing mode
    if args.mode == 'Testing':
        env_name = args.game
        # Create the environment game with stacked frame
        env = make_atari(env_name)
        if "SpaceInvaders" in env_name:
            env = wrap_deepmind(env, frame_stack=True, scale=True, number_of_frame=3)
        else:
            env = wrap_deepmind(env, frame_stack=True, scale=True, number_of_frame=4)  
        env.seed(58)

        # Create DQQN agent to training it
        myagent = AgentDDQN(env, args)

        myagent.test_loop()

if __name__ == '__main__':
    args = parse()
    start(args)
