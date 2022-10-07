import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'    # silence tensorflow messages
"""Main function for model inference."""
import argparse

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Train GNIME model for the specified target data and model explanation type.')
    parser.add_argument('--target', type=str, required=True, choices=['celeba', 'mnist', 'cifar10'],
                        help='target data type')
    parser.add_argument('--xai', type=str, required=True, choices=['grad', 'gradcam', 'lrp'],
                        help='xai type')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number for DNN training (default: 0))')
    
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments.
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    fname = f'scripts.train_{args.target}_{args.xai}'
    __import__(fname)
    

if __name__ == '__main__':
    main()