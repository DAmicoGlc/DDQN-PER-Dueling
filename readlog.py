import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def parse():
    """
        Command parser, choose:
            - Generate a plot
            - Compare two file
    """
    parser = argparse.ArgumentParser(description="Dueling DDQN Atari Game")
    parser.add_argument('--input', type=str, default='doubleQNresult/game.log', help='Loading folder, if there are multiple file use the - to divide them')
    parser.add_argument('--output', type=str, default='plotting', help='Saving folder')
    parser.add_argument('--compare', action='store_true', help='Compare more than 1 logs')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def compare_plot(filename_list, output):
    data = []
    log_read = 0
    for filename in filename_list:
        data.append({"Timestep":[], "Avg_reward": []})
        with open(filename, 'r') as f:
            for line in f:
                items = line.split("/")

                data[log_read]["Timestep"].append([int(s) for s in items[1].split(" ") if is_number(s)])
                data[log_read]["Avg_reward"].append([float(s) for s in items[4].split(" ") if is_number(s)])

        log_read += 1

    plt.figure()
    plt.title("Average Reward over Timestep")
    for dat in data:
        plt.plot(dat["Timestep"], dat["Avg_reward"])
        
    plt.xlabel('TimeStep')
    plt.ylabel('Avg Clipped Reward')
    plt.savefig(output + "/plotcompare.png")
    plt.show()

def generate_plot(filename, output):
    data_1 = {"Timestep":[], "Avg_reward": []}
    with open(filename, 'r') as f:
        
        data_read = 0
        for line in f:
            items = line.split("/")

            data_1["Timestep"].append([int(s) for s in items[1].split(" ") if is_number(s)])
            data_1["Avg_reward"].append([float(s) for s in items[4].split(" ") if is_number(s)])

    plt.figure()
    plt.title("Average Reward over Timestep")
    plt.plot(data_1["Timestep"], data_1["Avg_reward"])
    plt.xlabel('TimeStep')
    plt.ylabel('Avg Clipped Reward')
    plt.savefig(output + "/plot.png")
    plt.show()

if __name__ == '__main__':
    args = parse()
    
    if args.compare:
        filename = args.input.split('-')
        for file in filename:
            if not os.path.exists(file):
                sys.exit(file + " does not exists!")

        if not os.path.exists(args.output):
            os.makedirs(args.output)

        compare_plot(filename, args.output)
    else:
        if not os.path.exists(args.input):
            sys.exit(args.input + " does not exists!")

        if not os.path.exists(args.output):
            os.makedirs(args.output)

        generate_plot(args.input, args.output)