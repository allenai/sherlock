'''
Dummy predictor, outputs random numbers for predictions.

python dummy_predictor.py val_localization/val_localization_instances.json validation_set_predictions/localization.npy
'''
import argparse
import json
import numpy as np
import pprint

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_instances'
    )

    parser.add_argument(
        'output_npy'
    )

    parser.add_argument(
        '--constant_prediction',
        default=0,
        help='do constant prediction instead of random.'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(1)
    with open(args.input_instances) as f:
        data = json.load(f)

    predictions = {}
    for d in data:
        predictions[d['test_id']] = 0 if args.constant_prediction else float(np.random.random())

    sorted_scores = np.array(
        [predictions[k] for k in sorted(predictions.keys())]).astype(np.float32)

    print(sorted_scores.shape)
    
    np.save(args.output_npy, sorted_scores)

if __name__ == '__main__':
    main()
