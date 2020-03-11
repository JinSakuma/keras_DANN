import argparse
from configs.cfg import Config
from dann import DANN


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', dest='output', default='output')
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()
    config = Config()
    DANN = DANN(n_classes=10, config=config, output_name=args.output)
    DANN.train_source_only(batch_size=64, nSteps=1000, nEpochs=1)
    DANN.train(batch_size=64, train_Steps=100, val_Steps=5, nEpochs=20)
