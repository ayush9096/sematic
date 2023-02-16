# Standard Library
from typing import List, Optional

# Third Party
import numpy as np
import pandas
import plotly.express as px
import torch
import torch.nn as nn
from plotly.graph_objs import Figure, Heatmap
from sklearn.metrics import confusion_matrix
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import PrecisionRecallCurve  # type: ignore


class SpamClassifier(nn.Module):

    def __init__(self, bert) -> None:
       super(SpamClassifier, self).__init__()

       self.bert = bert
       self.dropout = nn.Dropout(0.1)
       self.relu = nn.ReLU()
       self.fc1 = nn.Linear(768, 512)  # Dense Layer 1
       self.fc2 = nn.Linear(512, 2)    # Dense Layer 2
       self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, sent_id, mask):

        # Pass the inputs to BERT model
        out = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(out['pooler_output'])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Applying softmax 
        out = self.softmax(x) 

        return out


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    cross_entropy: nn.NLLLoss,
    epoch: int,
    log_interval: int,
    dry_run: bool,
):
    model.train()

    total_loss = 0.0

    # List to save predictions
    total_preds = []

    for batch_idx, batch in enumerate(train_loader):

        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        # Previously calculated gradients
        model.zero_grad()

        preds = model(sent_id.long(), mask.long())
        loss = cross_entropy(preds, labels.long())
        total_loss = total_loss + loss.item()

        loss.backward()

         # Clip the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(batch),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break

    avg_loss = total_loss / len(train_loader)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


def evaluate(
    model: nn.Module,
    device: torch.device,
    cross_entropy: nn.NLLLoss,
    val_loader: DataLoader
):
    model.eval()

    # Total Loss and predictions
    total_loss = 0.0
    total_preds = []

    # Iterate over batches
    for batch_idx, batch in enumerate(val_loader):

        if batch_idx % 50 == 0 and not batch_idx == 0:

            # elapsed = format_time(time.time() - t0)
            print(' Batch {:>5,} of {:>5,}.'.format(batch_idx, len(val_loader)))
        
        # Push the batch to GPU
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():

            # Model Predictions
            preds = model(sent_id.long(), mask.long())

            # Compute validation loss between actual and predicted value
            loss = cross_entropy(preds, labels.long())

            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
        
    # Compute the validation loss of the epoch
    avg_loss = total_loss / len(val_loader)
    total_preds = np.concatenate(total_preds, axis=0)
    
    return avg_loss, total_preds


def _confusion_matrix(targets: List[int], preds: List[int]):

    matrix = confusion_matrix(y_true=targets, y_pred=preds)
    data = Heatmap(
        z=matrix,
        text=matrix,
        texttemplate="%{text}",
        x=list(range(10)),
        y=list(range(10)),
    )
    layout = {
        "title": "Confusion Matrix",
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
    }
    return Figure(data=data, layout=layout)


def test(model: nn.Module, cross_entropy: nn.NLLLoss, device: torch.device, test_loader: DataLoader):

    model.eval()

    correct = 0
    probas = []
    preds = []
    targets = []

    # Total Loss and predictions
    total_loss = 0.0
    total_preds = []

    # Iterate over batches
    for batch_idx, batch in enumerate(test_loader):

        if batch_idx % 50 == 0 and not batch_idx == 0:

            # elapsed = format_time(time.time() - t0)
            print(' Batch {:>5,} of {:>5,}.'.format(batch_idx, len(test_loader)))
        
        # Push the batch to device
        batch = [t.to(device) for t in batch]
        
        sent_id, mask, labels = batch

        with torch.no_grad():

            # Model Predictions
            preds = model(sent_id.long(), mask.long())

            # Compute loss between actual and predicted value
            loss = cross_entropy(preds, labels.long())
            

            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu()
            
            probas.append(preds)
            targets.append(labels)

            pred = preds.argmax(dim=1)
            total_preds.append(pred)
            correct += pred.eq(labels).sum().item()
        
    # Compute the validation loss of the epoch
    avg_loss = total_loss / len(test_loader)
    # total_preds = np.concatenate(total_preds, axis=0)

    pr_curve = PrecisionRecallCurve(num_classes=2, task='multiclass')

    precision, recall, thresholds = pr_curve(torch.cat(probas, axis=0), torch.cat(targets))

    classes = []
    for i in range(2):
        classes += [i] * len(precision[i])
    
    df = pandas.DataFrame(
        {
            "precision": list(torch.cat(precision).cpu()),
            "recall": list(torch.cat(recall).cpu()),
            "class": classes,
        }
    )

    fig = px.scatter(
        df,
        x="recall",
        y="precision",
        color="class",
        labels={"x": "Recall", "y": "Precision"},
    )

    return dict(
        average_loss=avg_loss,
        accuracy=correct/len(test_loader),
        pr_curve=fig,
        confusion_matrix=_confusion_matrix(
            torch.cat(targets).cpu(), torch.cat(total_preds).cpu()
        )
    )