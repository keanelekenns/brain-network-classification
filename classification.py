from __future__ import annotations
from typing import Any
import json
from datetime import datetime, timedelta
from statistics import mean, stdev
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from iidaka_transformer import IidakaTransformer
from pipeline import Pipeline
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

def calculate_mean_time(times: list[timedelta]) -> timedelta:
    # Can't do normal mean with timedelta objects
    time_sum = timedelta()
    for time in times:
        time_sum += time
    return time_sum / len(times)
    

def nested_grid_search_cv(X, y, pipeline_steps, step_param_grids, outer_cv = None, inner_cv = None, random_state = None, plot_prefix=None):
    
    outer_cv = outer_cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    inner_cv = inner_cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results_and_params = []
    
    i = 0
    tune_times = []
    train_times = []
    predict_times = []
    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        tune_start_time = datetime.now()
        results, pipeline = grid_search_cv(X=X_train, y=y_train, pipeline_steps=pipeline_steps, step_param_grids=step_param_grids, cv=inner_cv, random_state=random_state)

        tune_end_time = datetime.now()

        y_pred = pipeline.predict(X=X_test)
        predict_end_time = datetime.now()

        tune_time = tune_end_time - tune_start_time
        predict_time = predict_end_time - tune_end_time

        tune_times.append(tune_time)
        train_times.append(results["mean_train_time"])
        predict_times.append(predict_time)

        results_and_params.append({
            "results": {
                "report": classification_report(y_true=y_test, y_pred=y_pred, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_true=y_test, y_pred=y_pred)
            },
            "params": pipeline.params,
            "trained_pipeline_steps": pipeline.steps,
            "parameter_tuning_time": tune_time,
            "mean_train_time": results["mean_train_time"],
            "predict_time": predict_time
        })
        
        if plot_prefix:
            if pipeline.steps[0].__class__ == IidakaTransformer:
                transformer = pipeline.steps[0]
                masked = np.zeros_like(transformer.cohens_d)
                np.add(masked, transformer.cohens_d, out=masked, where=transformer.cohens_d > transformer.effect_size_threshold)
                plt.clf()
                sns.heatmap(masked)
                plt.title(f"{np.count_nonzero(masked)} features chosen, ES threshold = {transformer.effect_size_threshold}")
                plt.savefig(f"plots/{plot_prefix}-{i}-iidaka")
            elif pipeline.plot_prefix:
                pipeline.plot_prefix = f"{plot_prefix}-{i}-{pipeline.plot_prefix}"
            else:
                pipeline.plot_prefix = f"{plot_prefix}-{i}"
            pipeline.train_labels = y_train
            pipeline.test_labels = y_test
            if pipeline.is_plottable():
                pipeline.plot()
        
        i += 1

    accuracies = [ result["results"]["report"]["accuracy"] for result in results_and_params ]
    accuracy_mean = mean(accuracies)
    accuracy_stdev = stdev(accuracies, xbar=accuracy_mean)
    mean_tuning_time = calculate_mean_time(tune_times)
    mean_train_time = calculate_mean_time(train_times)
    mean_predict_time = calculate_mean_time(predict_times)

    summary = {
        "accuracy_mean": accuracy_mean,
        "accuracy_stdev": accuracy_stdev,
        "mean_tuning_time": mean_tuning_time,
        "mean_train_time": mean_train_time,
        "mean_predict_time": mean_predict_time
    }

    return results_and_params, summary




def grid_search_cv(X, y, pipeline_steps, step_param_grids, cv = None, random_state = None):
    
    cv = cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    param_dict = {}
    for step in pipeline_steps:
        if step_param_grids.get(step.__name__):
            param_list = list(ParameterGrid(step_param_grids[step.__name__]))
            param_dict[step.__name__] = param_list
    
    params_list = list(ParameterGrid(param_dict))
    
    train_times = []
    best_accuracy = 0
    best_params = None
    best_results = None
    best_summary = None
    for params in params_list:
        pipeline_inner = Pipeline(steps=pipeline_steps, params=params)

        results, summary = cross_validate(X=X, y=y, pipeline=pipeline_inner, cv=cv, random_state=random_state)
        avg_accuracy = summary["accuracy_mean"]

        if params == params_list[0]:
            print(f"grid_search_cv will take aproximately {(summary['mean_train_time'] + summary['mean_predict_time'])*cv.get_n_splits()*len(params_list)}")

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = params
            best_results = results
            best_summary = summary

        train_times.append(summary['mean_train_time'])
    
    pipeline = Pipeline(steps=pipeline_steps, params=best_params)
    pipeline.fit(X_train=X, y_train=y)

    mean_train_time = calculate_mean_time(train_times)

    results = {
        "summary": best_summary,
        "best_results": best_results,
        "mean_train_time": mean_train_time
    }

    return results, pipeline



def cross_validate(X, y, pipeline: Pipeline, cv = None, random_state = None, plot_prefix=None):
    
    cv = cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results = []

    i = 0
    train_times = []
    predict_times = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_start_time = datetime.now()
        pipeline.fit(X_train=X_train, y_train=y_train)

        train_end_time = datetime.now()

        y_pred = pipeline.predict(X=X_test)
        predict_end_time = datetime.now()

        train_time = train_end_time - train_start_time
        predict_time = predict_end_time - train_end_time

        train_times.append(train_time)
        predict_times.append(predict_time)

        results.append({
            "report": classification_report(y_true=y_test, y_pred=y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true=y_test, y_pred=y_pred),
            "train_time": train_time,
            "predict_time": predict_time
        })

        if plot_prefix:
            if pipeline.plot_prefix:
                pipeline.plot_prefix = f"{plot_prefix}-{i}-{pipeline.plot_prefix}"
            else:
                pipeline.plot_prefix = f"{plot_prefix}-{i}"
            pipeline.train_labels = y_train
            pipeline.test_labels = y_test
            if pipeline.is_plottable():
                pipeline.plot()
        
        i += 1
    
    accuracies = [ result["report"]["accuracy"] for result in results ]
    accuracy_mean = mean(accuracies)
    accuracy_stdev = stdev(accuracies, xbar=accuracy_mean)
    mean_train_time = calculate_mean_time(train_times)
    mean_predict_time = calculate_mean_time(predict_times)

    summary = {
        "accuracy_mean": accuracy_mean,
        "accuracy_stdev": accuracy_stdev,
        "mean_train_time": mean_train_time,
        "mean_predict_time": mean_predict_time,
        "params": pipeline.params
    }
    return results, summary


def write_results_to_file(filename: str, summary: dict[str, Any], results: list, parameter_grid: dict[str, Any], asd_count: int, td_count: int):
    with open(filename, 'a') as fp:
        fp.write("\n============================================================\n")
        fp.write(f"Date: {datetime.now().isoformat()}\n")
        fp.write(f"Counts: {asd_count} ASD subjects, {td_count} TD subjects.\n")
        fp.write("Summary:\n" + json.dumps(summary, default=str) + "\n")
        for result in results:
            fp.write(json.dumps(result, default=str) + "\n")
        fp.write("parameter grid:\n" + json.dumps(parameter_grid, default=str) + "\n")
        fp.write("\n============================================================\n")