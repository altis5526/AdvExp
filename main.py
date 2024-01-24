import argparse
import yaml
import pprint
from AdvExp import train


def load_params(fname):
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, help='name of config file to load', default='configs.yaml')

    args = parser.parse_args()
    params = load_params(args.fname)

    train(params)


    