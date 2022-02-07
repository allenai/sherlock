'''
Evaluate comparison task.

python score_comparison.py val_comparison/val_comparison_instances_random_predictions.json val_comparison/val_comparison_answer_key.json

also works with compressed numpy versions:

python score_comparison.py ../../clip_leaderboard_inference/comparison_test_predictions.npy test_comparison/test_comparison_answer_key.json --instance_ids test_comparison/test_comparison_instance_ids.json
'''
import argparse
import json
import pprint
import numpy as np

def pairwise_acc(model_pred, label, seed=1):
    #noise tiebreak vector
    np.random.seed(seed)
    tiebreak_preds = np.random.random(size=10)/1E9
    tiebroken_preds = model_pred + tiebreak_preds[:len(model_pred)]
    correct, total = 0, 0
    for idx1 in range(len(label)):
        for idx2 in range(idx1 + 1, len(label)):
            if label[idx1] == label[idx2]: continue
            total += 1
            correct += int((label[idx1] < label[idx2]) == (tiebroken_preds[idx1] < tiebroken_preds[idx2]))

    if total > 0 :
        return (correct / total - .5) * 2
    else:
        return 0.0


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
    np.random.seed(1) # just for random estimate, random seed is set later, so this is not that important

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

    accuracy = {'human': [], 'model': [], 'random': [], 'oracle': []}
    decoded_preds = {}
    for pred_id, pred_score in predictions.items():
        decoded_instance = answer_key['test_id_map'][pred_id]
        input_cand = (decoded_instance['Input_iid'], decoded_instance['candidate'])
        decoded_preds[input_cand] = pred_score

    for a in answer_key['annotations']:

        ann1_preds = np.array([float(c['annot1']) for c in a['candidates']])
        ann2_preds = np.array([float(c['annot2']) for c in a['candidates']])
        model_preds = [float(decoded_preds[(a['Input_iid'], c['source_iid'])]) for c in a['candidates']]

        human, model, random, oracle = [], [], [], []
        human_and_label = [(ann1_preds, ann2_preds), (ann2_preds, ann1_preds)]
        oracle_preds = np.mean(np.vstack([x[0] for x in human_and_label]), axis=0)

        for human_pred, label in human_and_label:
            human.append(pairwise_acc(human_pred, label))
            model.append(pairwise_acc(model_preds, label))
            random.append(pairwise_acc(np.random.random(label.shape), label))
            oracle.append(pairwise_acc(oracle_preds, label))

        accuracy['human'].append(np.mean(human))
        accuracy['model'].append(np.mean(model))
        accuracy['random'].append(np.mean(random))
        accuracy['oracle'].append(np.mean(oracle))

    print('Human corr: {:.2f}'.format(np.mean(accuracy['human'])*100))
    print('Model corr: {:.2f} (N={})'.format(np.mean(accuracy['model'])*100, len(accuracy['human'])))
    print('Random corr: {:.2f} (N={})'.format(np.mean(accuracy['random'])*100, len(accuracy['random'])))
    print('Oracle corr: {:.2f} (N={})'.format(np.mean(accuracy['oracle'])*100, len(accuracy['oracle'])))

if __name__ == '__main__':
    main()
