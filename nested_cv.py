from random import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from utils import plot_points

def pipeline_init(pipeline_steps, params):
    pipeline = []
    for step in pipeline_steps:
        if not params.get(step.__name__):
            pipeline.append(step())
        else:
            pipeline.append(step(**params[step.__name__]))
    
    return pipeline

def pipeline_pred(pipeline, X_train, y_train, X_test, return_transformed=False):
    y_pred = None
    for step in pipeline:
        if step != pipeline[-1]:
            step.fit(X_train, y_train)
            X_train = step.transform(X_train)
            X_test = step.transform(X_test)
        else:
            # The last step must be a classifier
            step.fit(X_train, y_train)
            y_pred = step.predict(X_test)

    if return_transformed:
        return y_pred, X_train, X_test

    return y_pred
    

def classify(X, y, pipeline_steps, step_params, outer_cv = None, inner_cv = None, random_state = None, plot_prefix=None, axes_labels=None, a_label="", b_label=""):
    
    outer_cv = outer_cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    inner_cv = inner_cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results_and_params = []

    param_dict = {}
    for step in pipeline_steps:
        if step_params.get(step.__name__):
            param_list = list(ParameterGrid(step_params[step.__name__]))
            param_dict[step.__name__] = param_list
    
    n_splits = inner_cv.get_n_splits()
    bucketed_param_lists = [ [] for _ in range(n_splits) ]
    param_list = list(ParameterGrid(param_dict))
    shuffle(param_list)

    i = 0
    for param in param_list:
        bucketed_param_lists[i % n_splits].append(param)
        i += 1
    
    i = 0
    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        j = 0
        chosen_params = None
        best_accuracy = 0
        for inner_train_index, inner_test_index in inner_cv.split(X_train, y_train):
            X_train_inner, X_test_inner = X_train[inner_train_index], X_train[inner_test_index]
            y_train_inner, y_test_inner = y_train[inner_train_index], y_train[inner_test_index]
            
            for param_set in bucketed_param_lists[j]:
                pipeline_inner = pipeline_init(pipeline_steps=pipeline_steps, params=param_set)
                y_pred_inner = pipeline_pred(pipeline=pipeline_inner, X_train=X_train_inner, y_train=y_train_inner, X_test=X_test_inner)

                report = classification_report(y_true=y_test_inner, y_pred=y_pred_inner, output_dict=True, zero_division=0)
                if report["accuracy"] > best_accuracy:
                    best_accuracy = report["accuracy"]
                    chosen_params = param_set

            j += 1

        pipeline = pipeline_init(pipeline_steps=pipeline_steps, params=chosen_params)
        y_pred, X_train_transformed, X_test_transformed = pipeline_pred(pipeline=pipeline, X_train=X_train, y_train=y_train, X_test=X_test, return_transformed=True)

        results_and_params.append({
            "results": {
                "report": classification_report(y_true=y_test, y_pred=y_pred, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_true=y_test, y_pred=y_pred)
            },
            "params": chosen_params,
            "trained_pipeline": pipeline,
        })
        
        #PLOT
        if plot_prefix:
            plot_points(X_train_transformed, y_train,
                        "plots/{}-{}-train-labels".format(plot_prefix,i), axes_labels=axes_labels, a_label=a_label, b_label=b_label)
            plot_points(X_test_transformed, y_pred,
                        "plots/{}-{}-test-predictions".format(plot_prefix,i), axes_labels=axes_labels, a_label=a_label, b_label=b_label)
            plot_points(X_test_transformed, y_test,
                        "plots/{}-{}-test-labels".format(plot_prefix,i), axes_labels=axes_labels, a_label=a_label, b_label=b_label)
        i += 1

    return results_and_params
