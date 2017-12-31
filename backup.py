import utils

def ignore(src, dst):
    return ['train.json', 'train.json.7z', 'test.json', 'test.json.7z', 'stinkbug.png', 'sample_submission.csv', 'sample_submission.csv.7z']

if __name__ == '__main__':
    utils.backup('/home/iceberg/data', '/var/storage/shared/pnrsy/sys/jobs/application_1513627406021_6970/data/iceberg', False, ignore, 600)