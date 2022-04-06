import numpy as np
import utils
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def classify(inputs, labels, inputs_to_points, num_folds=5,
                leave_one_out=False, plot=False,
                plot_prefix="", random_state=None, **kwargs):
    # Cumulative confusion matrix is used to report on classifier metrics over all of the k folds.
    cumulative_confusion_matrix = np.zeros((2,2))
    classifier = LinearSVC(random_state=random_state)
    splitter = None

    if leave_one_out:
        splitter = LeaveOneOut()
        loo_points = []
        loo_labels = []
        loo_predictions = []
        print("\nPerforming Leave-One-Out cross validation...")
    else:
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        print("\nPerforming {}-fold cross validation...".format(num_folds))

    i = 0
    for train_index, test_index in splitter.split(inputs, labels):
        train_inputs, test_inputs = inputs[train_index], inputs[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Call a custom function that translates whatever inputs are given
        # into points of a desired dimension (likely 2D)
        train_points, test_points = inputs_to_points(train_inputs, train_labels, test_inputs, **kwargs)

        # Scale the returned points
        points = np.concatenate((train_points, test_points))
        points = StandardScaler().fit_transform(points)
        train_points = points[:train_points.shape[0]]
        test_points = points[train_points.shape[0]:]

        classifier.fit(train_points, train_labels)
        test_pred = classifier.predict(test_points)

        if leave_one_out:
            loo_points += [test_points[0]]
            loo_labels += [test_labels[0]]
            loo_predictions += [test_pred[0]]
        else:
            if plot:
                utils.plot_points(train_points, train_labels,
                            "plots/{}-{}-train-labels".format(plot_prefix,i))
                utils.plot_points(test_points, test_pred,
                            "plots/{}-{}-test-predictions".format(plot_prefix,i))
                utils.plot_points(test_points, test_labels,
                            "plots/{}-{}-test-labels".format(plot_prefix,i))

            cumulative_confusion_matrix += confusion_matrix(test_labels, test_pred)
        
        i += 1
    
    if leave_one_out:
        loo_points = np.array(loo_points)
        loo_predictions = np.array(loo_predictions)
        loo_labels = np.array(loo_labels)
        utils.plot_points(loo_points, loo_predictions,
                    "plots/{}-{}-LOO-predictions".format(plot_prefix,i))
        utils.plot_points(loo_points, loo_labels,
                    "plots/{}-{}-LOO-labels".format(plot_prefix,i))
        cumulative_confusion_matrix = confusion_matrix(loo_labels, loo_predictions)

    print("\nMetrics using cumulative confusion matrix:")
    print(cumulative_confusion_matrix)
    print("Accuracy: {}\nPrecision: {}\nRecall: {}\nF1: {}"
            .format(*utils.evaluate_classifier(cumulative_confusion_matrix)))

#==================== TEST EXAMPLE ======================#
    
def to_points(train, test, bob, bill=10, judy=19):
    print(bob, bill, judy)
    print(test, train)
    return train, test

def main():
    print("testing...")
    inputs = np.array([[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]])
    labels = np.array([0, 1, 0, 0, 1, 0])
    #outdated call
    #classify(inputs, labels, 6, to_points, plot=False,bob=34, judy=12)
    

if __name__ == "__main__":
    main()