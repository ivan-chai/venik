import argparse

from .utils import SweepDB


def parse_arguments():
    parser = argparse.ArgumentParser("List sweeps. Use environment variables for locating Optuna.")
    args = parser.parse_args()
    return args


def list_sweeps(args):
    db = SweepDB()
    for r in sorted(db.get_sweeps_list(), key=lambda r: r["sweep_id"]):
        print(r["sweep_id"])


if __name__ == "__main__":
    args = parse_arguments()
    list_sweeps(args)
