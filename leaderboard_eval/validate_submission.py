'''
checks the validity of the submission

python validate_submission.py test_set_predictions.zip
'''
import argparse
import zipfile
import numpy as np
import io


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_zip',
        type=str
    )

    return parser.parse_args()

def check_array(val):
    assert val.dtype in [np.float32, np.float64], 'Only float32 or float64 supported'
    assert np.sum(np.isnan(val)) == 0, 'Your array contains a NaN.'
    

def main():
    args = parse_args()

    expected_shapes = {
        'comparison': 4719,
        'localization': 2365256,
        'retrieval_0': 1000000,
        'retrieval_1': 978121,
        'retrieval_2': 992016,
        'retrieval_3': 1008016,
        'retrieval_4': 964324,
        'retrieval_5': 978121,
        'retrieval_6': 992016,
        'retrieval_7': 982081,
        'retrieval_8': 976144,
        'retrieval_9': 984064,
        'retrieval_10': 984064,
        'retrieval_11': 1000000,
        'retrieval_12': 994009,
        'retrieval_13': 980100,
        'retrieval_14': 976144,
        'retrieval_15': 974169,
        'retrieval_16': 996004,
        'retrieval_17': 966289,
        'retrieval_18': 992016,
        'retrieval_19': 966289,
        'retrieval_20': 970225,
        'retrieval_21': 984064,
        'retrieval_22': 996004
    }

    if args.input_zip.split('/')[-1] != 'test_set_predictions.zip':
        print('please name your zip file "test_set_predictions.zip"')
        quit()

    archive = zipfile.ZipFile(args.input_zip)

    print('checking for comparison')
    
    comparison = np.load(io.BytesIO(archive.read('test_set_predictions/comparison.npy')))
    check_array(comparison)
    print('okay!')

    print('checking for localization')
    doing_localization = True
    try:
        localization = archive.read('test_set_predictions/localization.npy')
    except:
        print('localization.npy not found, skipping that task.')
        doing_localization = False

    if doing_localization:
        localization = np.load(io.BytesIO(localization))
        check_array(localization)
        assert localization.shape[0] == expected_shapes['localization'], 'Localization shape {} (expected {})'.format(
            localization.shape[0], expected_shapes['localization'])
        print('okay!')

    print('checking for retrieval')
    missing_splits = []
    for split_idx in range(23):
        try:
            retrieval = archive.read('test_set_predictions/retrieval_{}.npy'.format(split_idx))
        except:
            missing_splits.append(split_idx)
            continue
        retrieval = np.load(io.BytesIO(retrieval))
        check_array(retrieval)
        assert retrieval.shape[0] == expected_shapes['retrieval_{}'.format(split_idx)], 'Retrieval shape {} (expected {})'.format(
            retrieval.shape[0], expected_shapes['retrieval_{}'.format(split_idx)])

    if len(missing_splits) == 0:
        print('okay!')
    else:
        print('Missing retrieval splits with indices: {}'.format(missing_splits))
        
    
    
if __name__ == '__main__':
    main()
