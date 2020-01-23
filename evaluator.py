import time
from typing import List, Dict
import numpy as np
from numpy import ndarray
import pandas as pd
import logging

from data_generator import DataGenerator
from dataset import Dataset
from trial_data import TrialData, TrialMetaData
from approver import Approver

def score_model_batch(
        model,
        dataset: Dataset,
        batch_index: int,
        score_func):
    pred_y = model.predict(dataset.x, batch_index)
    score_dicts = score_func(pred_y, dataset.y)
    for k, v in score_dicts.items():
        score_dicts[k] = np.mean(v)
    return score_dicts

def get_true_model_scores(
        seed,
        data_generator: DataGenerator,
        trial_meta: TrialMetaData,
        num_test_obs: int,
        approved_models: List):
    """
    @return Dict: key = endpoint, value = matrix of true model scores over time
    """
    print("Evaluating...")
    np.random.seed(seed)
    true_model_score_dict = {}
    is_const_dist = "constant" in data_generator.support_sim_settings.min_func_name
    if is_const_dist:
        batch_data = data_generator.create_data(num_test_obs * 2, 0)

    for batch_index in range(trial_meta.num_batches):
        if not is_const_dist:
            batch_data = data_generator.create_data(num_test_obs, batch_index)

        true_model_scores = [
            score_model_batch(model, batch_data, batch_index, trial_meta.score_func)
            for model in approved_models]
        for k in true_model_scores[0].keys():
            if k not in true_model_score_dict:
                true_model_score_dict[k] = np.array([
                    [score_dict[k]] for score_dict in true_model_scores])
            else:
                new_scores = np.array([
                    [score_dict[k]] for score_dict in true_model_scores])
                true_model_score_dict[k] = np.concatenate([
                        true_model_score_dict[k],
                        new_scores
                    ], axis=1)

    print("Done evaluating")
    return true_model_score_dict

def _get_selected_scores(selected_model_hist, true_model_scores):
    t_idx = np.arange(len(selected_model_hist)) + 1
    return true_model_scores[selected_model_hist, t_idx]

def get_final_score(
        approver: Approver,
        true_scores_dict: Dict[str, ndarray]):
    all_final_scores = {}
    for k in true_scores_dict.keys():
        approved_scores = _get_selected_scores(
                approver.index_approved,
                true_scores_dict[k])
        all_final_scores[k] = approved_scores[-1]
    return all_final_scores

def count_bad_approval(
        window_length: int,
        ni_margin: float,
        approver: Approver,
        true_scores_dict: Dict[str, ndarray],
        leeway: float = 0.005):
    """
    @return ndarray with bad approval history per window
    """
    did_approval = np.array([0] + [m is not None for m in approver.approved_models[1:]])
    key_list = list(true_scores_dict.keys())
    bad_approval_hist = np.zeros(len(approver.approved_models))
    for t in range(bad_approval_hist.size):
        if not did_approval[t]:
            continue
        approved_score = np.array([true_scores_dict[k][approver.index_approved[t], t] for k in key_list]).reshape((-1,1))
        prev_approved_models = np.unique(approver.index_approved[:t])
        past_scores = np.array([true_scores_dict[k][prev_approved_models, t] for k in key_list])
        all_diffs = approved_score - past_scores
        is_strict_inferior = np.all(all_diffs < -leeway, axis=0)
        is_not_ni = np.any(all_diffs < -ni_margin - leeway, axis=0)
        bad_approval_hist[t] = np.any(is_strict_inferior + is_not_ni)

    window_bad_approval = np.zeros(bad_approval_hist.size)
    for j in range(window_bad_approval.size):
        window_bad_approval[j] = np.sum(bad_approval_hist[max(0, j + 1 - window_length): j + 1])
    return window_bad_approval

def count_bad_standard(
        window_length: int,
        ni_margin: float,
        approver: Approver,
        true_scores_dict: Dict[str, ndarray],
        leeway: float = 0.005):
    """
    @return ndarray with bad approval history per window
    """
    made_std = np.array([0] + [m is not None for m in approver.standard_models[1:]])
    key_list = list(true_scores_dict.keys())
    bad_std_hist = np.zeros(len(approver.standard_models))
    for t in range(bad_std_hist.size):
        if not made_std[t]:
            continue
        approved_score = np.array([true_scores_dict[k][approver.index_standard[t], t] for k in key_list]).reshape((-1,1))
        prev_approved_models = np.unique(approver.index_standard[:t])
        past_scores = np.array([true_scores_dict[k][prev_approved_models, t] for k in key_list])
        all_diffs = approved_score - past_scores
        is_any_worse = np.any(all_diffs < -leeway, axis=0)
        bad_std_hist[t] = np.any(is_any_worse)

    window_bad_std = np.zeros(bad_std_hist.size)
    for j in range(window_bad_std.size):
        window_bad_std[j] = np.sum(bad_std_hist[max(0, j + 1 - window_length): j + 1])
    return window_bad_std

def get_num_window_approvals(
        window_length: int,
        approver: Approver):
    did_approval = np.array([m is not None for m in approver.approved_models])

    standard_hist= np.array([
        np.sum(did_approval[max(0, i + 1 - window_length): i+1])
        for i in range(did_approval.size)])
    return standard_hist

def get_num_window_standards(
        window_length: int,
        approver: Approver):
    made_std = np.array([m is not None for m in approver.standard_models])

    standard_hist= np.array([
        np.sum(made_std[max(0, i + 1 - window_length): i+1])
        for i in range(made_std.size)])
    return standard_hist

def get_efficiency(
        approver: Approver,
        true_scores_dict: Dict[str, ndarray]):
    all_efficiency = {}
    for k in true_scores_dict.keys():
        approved_scores = _get_selected_scores(
                approver.index_approved,
                true_scores_dict[k])
        all_efficiency[k] = np.mean(approved_scores)
    return all_efficiency

def get_approver_scores(
        approver: Approver,
        true_scores_dict: Dict[str, ndarray]):
    endpoints = []
    times = []
    scores = np.array([])
    for k, v in true_scores_dict.items():
        logging.info("Endpoint %s", k)
        endpoint_scores = _get_selected_scores(approver.index_approved, v)
        endpoints += [k] * endpoint_scores.size
        times = np.concatenate([times, np.arange(endpoint_scores.size)])
        scores = np.concatenate([scores, endpoint_scores])
    scores_dict = {
            "val": scores,
            "endpoint": endpoints,
            "time": times}
    scores_df = pd.DataFrame(scores_dict)
    return scores_df
