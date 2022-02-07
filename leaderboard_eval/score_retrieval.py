'''
Compute retrieval metrics.

python score_retrieval.py val_retrieval/val_retrieval_0_instances_random_predictions.json val_retrieval/val_retrieval_0_answer_key.json
'''
import argparse
import numpy as np
import json
import scipy.stats
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions')
    parser.add_argument('answer_key')
    parser.add_argument(
        '--instance_ids',
        default=None,
        type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(1)

    if '.json' in args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)
    elif '.npy' in args.predictions:
        pred_vec = np.load(args.predictions)
        with open(args.instance_ids) as f:
            pred_vec_ids = json.load(f)
        predictions = dict(zip(pred_vec_ids, pred_vec))

    with open(args.answer_key) as f:
        answer_key = json.load(f)

    instance_ids = list(set([x[0] for x in answer_key.values()]))
    decoded_predictions = {tuple(answer_key[test_instance_id]): prediction for test_instance_id, prediction in predictions.items()}

    # image 2 txt
    sim_mat = np.zeros((len(instance_ids), len(instance_ids)))

    for idx1, instance_id1 in enumerate(instance_ids):
        for idx2, instance_id2 in enumerate(instance_ids):
            sim_mat[idx1, idx2] = decoded_predictions[(instance_id1, instance_id2)]

    im2text_ranks = np.diagonal(scipy.stats.rankdata(-sim_mat, axis=1))
    text2im_ranks = np.diagonal(scipy.stats.rankdata(-sim_mat, axis=0))
    p_at_1 = float(100*np.mean(im2text_ranks == 1.0))
    print('im2txt: {:.3f}'.format(np.mean(im2text_ranks)))
    print('txt2im: {:.3f}'.format(np.mean(text2im_ranks)))
    print('p_at_1: {:.3f}'.format(np.mean(p_at_1)))


if __name__ == '__main__':
    main()
