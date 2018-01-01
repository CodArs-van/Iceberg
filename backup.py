import utils
import argparse

parser = argparse.ArgumentParser(description='copy source dir to destination dir every given seconds')
parser.add_argument('-s', '--src', default='', type=str, metavar='PATH', help='path to source dir (default: "")')
parser.add_argument('-d', '--dst', default='', type=str, metavar='PATH', help='path to destination dir (default: "")')
parser.add_argument('-t', '--time', default=600, type=int, metavar='N', help='interval seconds to backup')

def ignore(src, dst):
    return ['train.json', 'train.json.7z', 'test.json', 'test.json.7z', 'stinkbug.png', 'sample_submission.csv', 'sample_submission.csv.7z']

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.src, 'source dir is none'
    assert args.dst, 'distination dir is none'
    assert args.time > 0, 'interval should not be negative'
    
    utils.backup(args.src, args.dst, False, ignore, args.time)