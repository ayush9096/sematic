# Standard Library
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

# Third-Party
import plotly
import plotly.express as px
from plotly.graph_objs import Figure
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset

# Sematic
import sematic
from sematic import KubernetesResourceRequirements, ResourceRequirements
from sematic.examples.bert_fine_tune_scam.train_eval import SpamClassifier, train, evaluate, test

@dataclass
class SpamDataset:
    data : pd.DataFrame

@sematic.func(inline=True)
def load_spam_dataset(data_path: str) -> SpamDataset:
    df = pd.read_csv(data_path)
    return SpamDataset(data=df)

@dataclass
class DataLoaderConfig:
    batch_size: Optional[int] = 32

@sematic.func(inline=True)
def get_dataloader(dataset: Dataset, config: DataLoaderConfig) -> DataLoader:
    return DataLoader(dataset, batch_size=config.batch_size)


@dataclass
class TrainConfig:
    learning_rate: float = 1.0
    epochs: int = 14
    max_seq_len: int = 25
    batch_size: int = 32
    dry_run: bool = False
    log_interval: int = 50
    cuda: bool = False
    


@dataclass
class PipelineConfig:
    dataloader_config: DataLoaderConfig
    train_config: TrainConfig
    use_cuda: bool = False
    data_path: str = './data/spam_data.csv'

@sematic.func(inline=True)
def train_model(
    config: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_labels: List[object]
) -> nn.Module:
    """ Train the model """

    bert, _ = load_bert_model_tokenizer()

    # Freeze all the parameters of bert
    for param in bert.parameters():
        param.requires_grad = False

    # Spam Classifier Model
    model = SpamClassifier(bert=bert).to(device)

    class_wts = compute_class_weight(class_weight = "balanced", classes = np.unique(train_labels), y = train_labels)
    weights= torch.tensor(class_wts,dtype=torch.float)
    weights = weights.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr = 1e-3)
    cross_entropy = nn.NLLLoss(weight=weights)
    
    # Set initial loss to inf
    best_valid_loss = float('inf')

    # Store training and validation loss for each epochs
    train_losses = []
    valid_losses = []

    for epoch in range(1, config.epochs + 1):
        
        # Train Model
        train_loss, _ = train(
            model,
            device,
            train_loader,
            optimizer,
            cross_entropy,
            epoch,
            config.log_interval,
            config.dry_run
        )

        # Evaluate Model
        valid_loss, _ = evaluate(
            model,
            device,
            cross_entropy,
            val_loader
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')
        

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Training Loss : {train_loss:.2f}')
        print(f'Validation Loss : {valid_loss:.2f}')

    return model


@dataclass
class EvaluationResults:
    test_set_size: int
    average_loss: float
    accuracy: float
    pr_curve: plotly.graph_objs.Figure
    confusion_matrix: plotly.graph_objs.Figure


@sematic.func(inline=True)
def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device, train_labels: List[object]
) -> EvaluationResults:
    """
    Evaluate the model.
    """
    model = model.to(device)

    class_wts = compute_class_weight(class_weight = "balanced", classes = np.unique(train_labels), y = train_labels)
    weights= torch.tensor(class_wts,dtype=torch.float)
    weights = weights.to(device)
   
    cross_entropy = nn.NLLLoss(weight=weights)
    
    results = test(model, cross_entropy, device, test_loader)
    return EvaluationResults(
        test_set_size=len(test_loader.dataset),  # type: ignore
        average_loss=results["average_loss"],
        accuracy=results["accuracy"],
        pr_curve=results["pr_curve"],
        confusion_matrix=results["confusion_matrix"],
    )

@sematic.func(inline=True)
def train_eval(
    train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader, train_config: TrainConfig, train_labels: List[object]
) -> EvaluationResults:
    """
    The train/eval sub-pipeline.
    """

    device = torch.device("cuda" if train_config.cuda else "cpu")

    model = train_model(
        config=train_config, train_loader=train_dataloader, val_loader=val_dataloader, device=device, train_labels=train_labels
    )

    print(" Model Trained Successfully ")

    evaluation_results = evaluate_model(
        model=model, test_loader=test_dataloader, device=device, train_labels=train_labels
    )

    return evaluation_results

def get_tokenized_text(
    dataset: pd.DataFrame, max_seq_len: int, tokenizer: BertTokenizerFast
):

    tokens_set = tokenizer.batch_encode_plus(
        dataset.tolist(),
        max_length = max_seq_len,
        pad_to_max_length = True,
        truncation = True,
        return_token_type_ids=False
    )

    return tokens_set

@sematic.func(inline=True)
def get_dataloaders(
    spam_dataset: SpamDataset, tokenizer: BertTokenizerFast, max_seq_len: int, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader, List[object], Figure]:

    data_df = spam_dataset.data
    train_text, temp_text, train_labels, temp_labels = train_test_split( 
        data_df['text'], data_df['label'], random_state=218, test_size=0.3, stratify=data_df['label'])
    
    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels, random_state=218, test_size=0.5, stratify=temp_labels
    )

    # get length of all the messages in the train set
    seq_len = [len(i.split()) for i in train_text]
    seq_df = pd.DataFrame(seq_len, columns=['length'])
    fig = px.histogram(seq_df, x='length')

    tokens_train = get_tokenized_text(train_text, max_seq_len, tokenizer)
    tokens_val = get_tokenized_text(val_text, max_seq_len, tokenizer)
    tokens_test = get_tokenized_text(test_text, max_seq_len, tokenizer)

    train_data = TensorDataset(
        torch.Tensor(tokens_train['input_ids']), torch.Tensor(tokens_train['attention_mask']), torch.Tensor(train_labels.tolist()))
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(
        torch.LongTensor(tokens_val['input_ids']), torch.LongTensor(tokens_val['attention_mask']), torch.Tensor(val_labels.tolist()))
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


    test_data = TensorDataset(
        torch.LongTensor(tokens_test['input_ids']), torch.LongTensor(tokens_val['attention_mask']), torch.Tensor(test_labels.tolist()))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_loader, train_labels, fig


def load_bert_model_tokenizer():
    
    # Load BERT based pre-trained model
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    return bert, tokenizer

@sematic.func(inline=True)
def pipeline(config: PipelineConfig) -> EvaluationResults:
    """
    The root function of the pipeline.
    """
    spam_dataset = load_spam_dataset(config.data_path)

    _, tokenizer = load_bert_model_tokenizer()

    # Data Loaders
    train_dataloader, val_dataloader, test_dataloader, train_labels, fig = get_dataloaders(
        spam_dataset, tokenizer, config.train_config.max_seq_len, config.train_config.batch_size
    )

    evaluation_results = train_eval(
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        test_dataloader = test_dataloader,
        train_config= config.train_config,
        train_labels=train_labels
    )

    return evaluation_results
