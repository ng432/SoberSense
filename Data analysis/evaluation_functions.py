
#%%
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch as t

#%%

def calc_model_prec_recall_f1(model, dataloader, threshold = 0.5, num_rep = 100, device = 'mps'):
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
                true_labels.append(y.cpu())
                predictions.append(predicted_labels)

    true_labels = np.concatenate(true_labels)
    predictions = np.concatenate(predictions)

    print(len(true_labels))

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return precision, recall, f1
# %%
