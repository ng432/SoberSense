# %%
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch as t

# %%


def calc_prec_recall_f1(model, dataloader, threshold=0.5, num_rep=100, device="mps"):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predictions = []

    with t.no_grad():
        for _ in range(num_rep):
            for _, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                outputs = model(X)
                predicted_probs = outputs.squeeze().cpu()
                predicted_labels = (predicted_probs >= threshold).float()

                true_labels.append(y.cpu().numpy().flatten())
                predictions.append(predicted_labels.numpy().flatten())

    true_labels = np.concatenate(true_labels)
    predictions = np.concatenate(predictions)

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return precision, recall, f1


def calc_prf1_majority_vote(
    model, dataloader, threshold=0.5, num_rep=101, device="mps"
):
    # repeats model labelling multiple times over each sample
    # will then take the majority voted label for that sample as the predicted label

    model.eval()  # Set the model to evaluation mode

    collected_predictions = []

    with t.no_grad():
        for _ in range(num_rep):
            true_labels = []
            predictions = []

            for _, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                outputs = model(X)

                predicted_probs = outputs.squeeze().cpu()
                predicted_labels = (predicted_probs >= threshold).float()

                predictions.extend(predicted_labels.numpy().flatten())
                true_labels.extend(y.cpu().numpy().flatten())

            collected_predictions.append(predictions)

    stacked_predictions = np.stack(collected_predictions)
    # doing a 'majority' voting
    # as the array contains 0 or 1s, the median is equivalent to the mode
    collapsed_predictions = np.median(stacked_predictions, axis=0)

    true_labels = np.array(true_labels)

    precision = precision_score(true_labels, collapsed_predictions)
    recall = recall_score(true_labels, collapsed_predictions)
    f1 = f1_score(true_labels, collapsed_predictions)

    return precision, recall, f1, true_labels, collapsed_predictions


# %%
