import argparse
from utils import transform_monitor

# This should happen automatically when run.py finishes, but it won't if run.py
# is killed early. In that case, run this standalone script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Gym environment name')
    parser.add_argument('--dir', help='The directory of the monitor files')
    args = parser.parse_args()

    transform_monitor(args.dir, args.env)