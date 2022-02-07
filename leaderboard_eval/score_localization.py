'''
Evaluate localization task.

python score_localization.py val_localization/val_localization_instances_random_predictions.json val_localization/val_localization_answer_key.json
'''
import argparse
import json
import pprint
import numpy as np
import lapjv
import collections
import math
import tqdm


def get_region_sol(text_region_sim):
    '''little solver for gt instead of argmax'''
    text_region_sim = np.array(text_region_sim)
    sol, _, _ = lapjv.lapjv(-text_region_sim)
    i_s = np.arange(len(sol))
    j_s = sol[i_s]
    return j_s


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions')
    parser.add_argument('answer_key')
    parser.add_argument(
        '--iou_thresh',
        default=.5,
        type=float
    )
    parser.add_argument(
        '--instance_ids',
        default=None,
        type=str)

    return parser.parse_args()


def main():
    args = parse_args()

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

    # recollect each image: eval metrics are macro means over images.
    im2gt_ans = collections.defaultdict(list)
    im2auto_ans = collections.defaultdict(list)

    for a in answer_key.items():
        if a[1]['type'] == 'gt':
            im2gt_ans[a[1]['image']].append(a)
        elif a[1]['type'] == 'auto':
            im2auto_ans[a[1]['image']].append(a)

    auto_accs, oracle_auto_accs = [], []
    for image, anses in tqdm.tqdm(im2auto_ans.items()):
        per_inst = collections.defaultdict(list)
        for test_id, ans in anses:
            ans['score'] = predictions[test_id]
            per_inst[ans['inst_id']].append(ans)

        corr = []
        corr_oracle = []
        for inst, preds in per_inst.items():
            top_pred_idx = np.argmax([p['score'] for p in preds])
            oracle_pred_idx = np.argmax([p['IoU'] for p in preds])
            corr.append(float(preds[top_pred_idx]['IoU'] > args.iou_thresh))
            corr_oracle.append(float(preds[oracle_pred_idx]['IoU'] > args.iou_thresh))
        auto_accs.append(np.mean(corr))
        oracle_auto_accs.append(np.mean(corr_oracle))

    gt_accs = []
    error = 0
    for image, anses in tqdm.tqdm(im2gt_ans.items()):
        n_infs = math.isqrt(len(anses))
        assert len(anses) == n_infs ** 2, (n_infs, len(anses)) # assert perfect square

        inst_id_map = {}
        for a in anses:
            if a[1]['correct']:
                inst_id_map[a[1]['inst_id']] = a[1]['bbox_idx']

        sim_mat = np.zeros((n_infs, n_infs))
        for a in anses:
            bbox_idx = a[1]['bbox_idx']
            inf_idx = inst_id_map[a[1]['inst_id']]
            sim_mat[bbox_idx, inf_idx] = predictions[a[0]]

        gt_solve = get_region_sol(sim_mat)
        gt_accs.append(np.mean(np.arange(n_infs)==gt_solve))

    print('acc gt bbox = {:.4f} (n={}), acc auto bbox = {:.4f} (n={}), acc oracle bbox = {:.4f} (n={})'.format(
        np.mean(gt_accs) * 100, len(gt_accs), np.mean(auto_accs) * 100, len(auto_accs), np.mean(oracle_auto_accs) * 100, len(oracle_auto_accs)))


if __name__ == '__main__':
    main()
