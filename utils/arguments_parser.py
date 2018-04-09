"""
Custom arguments parser module.

Author: bbarbosa
Date: 02-04-2018
"""

import argparse

TRUE_CASES = ['true', 't', 'yes', '1']
CUSTOM_FORMATTER_CLASS = lambda prog: argparse.HelpFormatter(prog, max_help_position=2000)

def training_parser():
    """
    Training parser.
    """
    parser = argparse.ArgumentParser(
    description="High level Tensorflow and TFLearn training script.",
    prefix_chars='-',
    formatter_class=CUSTOM_FORMATTER_CLASS)

    # required arguments
    parser.add_argument(
        "--train_dir",
        required=True,
        help="<REQUIRED> directory to the training data",
        type=str)
    parser.add_argument(
        "--arch", required=True, help="<REQUIRED> architecture name", type=str)
    parser.add_argument(
        "--model_name",
        required=True,
        help="<REQUIRED> Model name / Path to trained model",
        type=str)

    # optional arguments
    parser.add_argument(
        "--bsize",
        required=False,
        help="batch size (default=16)",
        default=16,
        type=int)
    parser.add_argument(
        "--test_dir",
        required=False,
        help="directory to the testing data (default=None)",
        type=str)
    parser.add_argument(
        "--height",
        required=False,
        help="images height (default=64)",
        default=64,
        type=int)
    parser.add_argument(
        "--width",
        required=False,
        help="images width (default=64)",
        default=64,
        type=int)
    parser.add_argument(
        "--val_set",
        required=False,
        help="percentage of training data to validation (default=0.3)",
        default=0.3,
        type=float)
    parser.add_argument(
        "--gray",
        required=False,
        help="convert images to grayscale (default=False)",
        default=False,
        type=lambda s: s.lower() in TRUE_CASES)
    parser.add_argument(
        "--freeze",
        required=False,
        help="freeze graph (not for retraining) (default=False)",
        default=False,
        type=lambda s: s.lower() in TRUE_CASES)
    parser.add_argument(
        "--snap",
        required=False,
        help="evaluate training frequency (default=5)",
        default=5,
        type=int)
    parser.add_argument(
        "--pproc",
        required=False,
        help="enable/disable pre-processing (default=True)",
        default=True,
        type=lambda s: s.lower() in TRUE_CASES)
    parser.add_argument(
        "--aug",
        required=False,
        nargs="+",
        help="enable data augmentation (default=[])",
        default=[])
    parser.add_argument(
        "--n_epochs",
        required=False,
        help="maximum number of training epochs (default=1000)",
        default=1000,
        type=int)
    parser.add_argument(
        "--eval_crit",
        required=False,
        help="classification confidence threshold (default=0.75)",
        default=0.75,
        type=float)
    parser.add_argument(
        "--cspace",
        required=False,
        help="convert images color space (default=None)",
        default=None,
        type=str)
    parser.add_argument(
        "--param",
        required=False,
        help="versatile extra parameter (default=None)",
        default=None)
    parser.add_argument(
        "--show",
        required=False,
        help="show test model & PDF report (default=False)",
        default=False,
        type=lambda s: s.lower() in TRUE_CASES)

    # parse arguments
    args = parser.parse_args()

    print(args, "\n")

    return args

