from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='allrecipes')
parser.add_argument('--hop', type=int, default=0)
args = parser.parse_args()

run_experiment(f"config_files/{args.dataset}_quartiles_{args.hop}.yml")
