from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--model', type=str, default='ngcf')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

run_experiment(f"config_files/{args.model}_{args.dataset}.yml")
