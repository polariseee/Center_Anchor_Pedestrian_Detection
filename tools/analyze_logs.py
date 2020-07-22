import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import pdb


def plot_curve(args):
    xs = []
    ys = []
    num = 0
    with open(args.json_file, 'r', encoding='utf-8') as load_f:
        lines = load_f.readlines()
    for line in lines:
        if num == 1500:
            break
        line = line.strip('\n')
        datas = json.loads(line)
        xs.append(np.array(datas['iteration']))
        ys.append(np.array(datas['loss_csp']))
        num += 1

    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.ylim(0.0, 0.25)
    plt.plot(xs, ys, label='loss_csp', linewidth=0.5)
    plt.legend(loc='upper right')
    print(f'save curve to: {args.out}')
    plt.savefig(args.out)
    plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Log')
    parser.add_argument("json_file", help="json file", type=str)
    parser.add_argument("out", help="out", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot_curve(args)


if __name__ == '__main__':
    main()
