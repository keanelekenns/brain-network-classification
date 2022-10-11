from random import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, ParameterGrid


def classify(X, y, pipeline_steps, step_params, outer_cv = None, inner_cv = None, random_state = None):
    
    outer_cv = outer_cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    inner_cv = inner_cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results_and_params = []

    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        param_dict = {}
        num_combinations = 1
        for step in pipeline_steps:
            param_list = list(ParameterGrid(step_params[step.__name__]))
            param_dict[step.__name__] = param_list
            num_combinations *= len(param_list)
        
        n_splits = inner_cv.get_n_splits()
        bucketed_param_lists = [ [] for _ in range(n_splits) ]
        param_list = list(ParameterGrid(param_dict))
        shuffle(param_list)

        i = 0
        for param in param_list:
            bucketed_param_lists[i % n_splits].append(param)
            i += 1

        i = 0
        chosen_params = None
        best_accuracy = 0
        for inner_train_index, inner_test_index in inner_cv.split(X_train, y_train):
            X_train_inner, X_test_inner = X_train[inner_train_index], X_train[inner_test_index]
            y_train_inner, y_test_inner = y_train[inner_train_index], y_train[inner_test_index]
            
            for param_set in bucketed_param_lists[i]:
                pipeline = []
                for step in pipeline_steps:
                    pipeline.append(step(**param_set[step.__name__]))

                X_train_inner_transformed = X_train_inner
                X_test_inner_transformed = X_test_inner

                for step in pipeline:
                    # The last step must be a classifier
                    if step == pipeline[-1]:
                        step.fit(X_train_inner_transformed, y_train_inner)
                        y_pred = step.predict(X_test_inner_transformed)
                        report = classification_report(y_true=y_test_inner, y_pred=y_pred, output_dict=True)
                        if report["accuracy"] > best_accuracy:
                            best_accuracy = report["accuracy"]
                            chosen_params = param_set
                    else:
                        step.fit(X_train_inner_transformed, y_train_inner)
                        X_train_inner_transformed = step.transform(X_train_inner_transformed)
                        X_test_inner_transformed = step.transform(X_test_inner_transformed)

            i += 1

        pipeline = []
        for step in pipeline_steps:
            pipeline.append(step(**chosen_params[step.__name__]))

        X_train_transformed = X_train
        X_test_transformed = X_test

        for step in pipeline:
            # The last step must be a classifier
            if step == pipeline[-1]:
                step.fit(X_train_transformed, y_train)
                y_pred = step.predict(X_test_transformed)
                report = classification_report(y_true=y_test_inner, y_pred=y_pred, output_dict=True)
                results_and_params.append({"results": report, "params": chosen_params, "trained_pipeline": pipeline})
            else:
                step.fit(X_train_transformed, y_train)
                X_train_transformed = step.transform(X_train_transformed)
                X_test_transformed = step.transform(X_test_transformed)

    return results_and_params
