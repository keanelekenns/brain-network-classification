import numpy as np
import argparse
import utils
from sklearn.model_selection import train_test_split

def evaluate_classifier(predictions_A, predictions_B):
    """
    Inputs:
        predictions_A - A 1D numpy array holding prediction values for members of class A.
        predictions_B - A 1D numpy array holding prediction values for members of class B.
        The prediction values are as follows: Class A = 0, Class B = 1.
    Returns:
        Accuracy, Precision, Recall, F1 - Classifier metrics as defined by 
        https://towardsdatascience.com/classification-performance-metrics-69c69ab03f17
    """
    FP = np.sum(predictions_A)
    TN = len(predictions_A) - FP

    TP = np.sum(predictions_B)
    FN = len(predictions_B) - TP

    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*precision*recall/(precision + recall)

    return accuracy, precision, recall, f1

def cs_p1(cs_a_b, cs_b_a, test_A, test_B):
    """
    Inputs:
        cs_a_b - A 1D numpy array representing the contrast subgraph generated from
        finding a dense subgraph in the difference graph summary_A - summary_B.
        cs_b_a - Similar to cs_a_b, only the difference graph used to generate it was
        summary_B - summary_A.
        test_A - A 3D numpy array representing a group of brain graphs in class A.
        test_B - A 3D numpy array representing a group of brain graphs in class B.
    Returns:
        predictions_A - A 1D numpy array holding prediction values for each graph in test_A.
        predictions_B - A 1D numpy array holding prediction values for each graph in test_B.
        Note that these arrays have shape (len(test_A)) and (len(test_B)) respectively.
        The prediction values are as follows: Class A = 0, Class B = 1.
    """
    predictions_A = np.zeros(len(test_A))
    i = 0
    for graph in test_A:
        a_similarity = utils.contrast_subgraph_overlap(graph, cs_a_b)
        b_similarity = utils.contrast_subgraph_overlap(graph, cs_b_a)
        if b_similarity > a_similarity:
            predictions_A[i] = 1
        i += 1

    predictions_B = np.zeros(len(test_B))
    i = 0
    for graph in test_B:
        a_similarity = utils.contrast_subgraph_overlap(graph, cs_a_b)
        b_similarity = utils.contrast_subgraph_overlap(graph, cs_b_a)
        if b_similarity > a_similarity:
            predictions_B[i] = 1
        i += 1
    
    return predictions_A, predictions_B

def main():
    parser = argparse.ArgumentParser(description='Graph Classification via Contrast Subgraphs')
    parser.add_argument('A_dir', help='Filepath to class A directory containing brain network files', type=str)
    parser.add_argument('B_dir', help='Filepath to class B directory containing brain network files', type=str)
    parser.add_argument('alpha', help='Penalty value for contrast subgraph size (varies from 0 to 1)', type=float)
    parser.add_argument('-p', help='Problem Forumlation (default: 1)', type=int, default = 1, choices={1,2})

    args = parser.parse_args()
    if(args.alpha < 0 or args.alpha > 1):
        raise ValueError("alpha should be between 0 and 1 inclusive.")
    
    # Read brain graph files into numpy arrays
    graphs_A = utils.get_graphs_from_files(args.A_dir)
    graphs_B = utils.get_graphs_from_files(args.B_dir)
    # Split into training and testing samples
    train_A, test_A = train_test_split(graphs_A, test_size=0.2, random_state=23)
    train_B, test_B = train_test_split(graphs_B, test_size=0.2, random_state=23)
    # Create and Write Summary Graphs
    summary_A = utils.summary_graph(train_A)
    summary_B = utils.summary_graph(train_B)

    # Get the difference network between the edge weights in group A and B
    if args.p == 1:
        diff_a_b = summary_A - summary_B
        diff_b_a = summary_B - summary_A
        # Call function(s) for generating contrast subgraphs
        # TEMPORARILY HARDCODED
        # Male TD-ASD alpha = 0.8
        cs_a_b = np.array([64, 35, 3, 24, 31])
        # Male ASD-TD alpha = 0.8
        cs_b_a = np.array([88, 54, 39])

        pred_A, pred_B = cs_p1(cs_a_b, cs_b_a, test_A, test_B)
        print(evaluate_classifier(pred_A, pred_B))
    else:
        diff = abs(summary_A - summary_B)
        # Call function(s) for generating contrast subgraph
        # TEMPORARILY HARDCODED

if __name__ == "__main__":
    main()