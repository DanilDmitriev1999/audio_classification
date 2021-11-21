import torch
import pytorch_lightning as pl

from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AdamW



class ModelTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 criterion,
                 lr=1e-4):
        super().__init__()

        self.model = model

        # other
        self.lr = lr
        self.criterion = criterion
        
        self.test_res = {'prob': [], 'pred': [], 'label': []}
        self.flatten = lambda t: [item for sublist in t for item in sublist]

    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        labels = batch['label']

        predictions = self(inputs)
        loss = self.criterion(predictions, labels)

        predict = torch.round(torch.sigmoid(predictions))
        predict = predict.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        f1 = f1_score(labels, predict)

        values = {'train_loss': loss,
                  'train_f1': f1}

        self.log_dict(values)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']
        labels = batch['label']

        predictions = self(inputs)
        loss = self.criterion(predictions.float(), labels.float())

        predict = torch.round(torch.sigmoid(predictions))
        predict = predict.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        
        self.test_res['pred'].append(predict)
        self.test_res['label'].append(labels)
        self.test_res['prob'].append(predictions)

        values = {'val_loss': loss}

        self.log_dict(values)

        return loss
    
    def validation_epoch_end(self, outputs):
        epoch_pred = [i for i in self.flatten(self.test_res['pred'])]
        epoch_label = [i for i in self.flatten(self.test_res['label'])]
        
        precision = precision_score(epoch_label, epoch_pred)
        recall = recall_score(epoch_label, epoch_pred)
        f1 = f1_score(epoch_label, epoch_pred)
        
        values = {'val_f1': f1, 'val_precision': precision, 'val_recall': recall}
        self.log_dict(values)
        
        self.test_res = {'prob': [], 'pred': [], 'label': []}

    def test_step(self, batch, batch_idx):
        inputs = batch['inputs']
        labels = batch['label']

        predictions = self(inputs)
        loss = self.criterion(predictions.float(), labels.float())

        predict = torch.round(torch.sigmoid(predictions))
        predict = predict.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        self.test_res['pred'].append(predict)
        self.test_res['label'].append(labels)
        self.test_res['prob'].append(predictions)

        values = {'test_loss': loss}

        self.log_dict(values)

        return loss
    
    def test_epoch_end(self, outputs):
        epoch_pred = [i for i in self.flatten(self.test_res['pred'])]
        epoch_label = [i for i in self.flatten(self.test_res['label'])]
        
        precision = precision_score(epoch_label, epoch_pred)
        recall = recall_score(epoch_label, epoch_pred)
        f1 = f1_score(epoch_label, epoch_pred)
        
        values = {'test_f1': f1, 'test_precision': precision, 'test_recall': recall}
        self.log_dict(values)