import json
import argparse


def load_from_json(path):
    with open(path, 'rb') as f:
        file_config = json.load(f)

    assert type(file_config) == dict
    return file_config


def load_from_argparse(args):
    cmd_config = {}
    for arg in vars(args):
        cmd_config[arg] = getattr(args, arg)

    return cmd_config
