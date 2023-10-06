import torch
import torchmetrics
from utils import latency_metrics,str_to_num
from model import NeuralNetwork
import torch.utils.data as Data
from tqdm import tqdm

features = {
    'conv-bn-relu': 7,
    'dwconv-bn-relu': 7,
    'add': 3,
    'add-relu': 3,
    'avgpool': 5,
    'fc': 5,
    'global-avgpool': 5,
    'maxpool': 5,
    'se': 5,
    'swish': 5,
    'hswish': 5,
}

class Trainer:
    def __init__(
       self,
       train_dataset=None,
       eval_dataset=None,
       kernel_type=None,
       batch_size=32,
       learning_rate=0.001,
       epochs=350,
       save_dir = None,
       weights = None
   ):   
        self.train_dataset = train_dataset 
        self.eval_dataset = eval_dataset
        self.kernel_type = kernel_type if kernel_type else 'kernel'
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_fn = None
        self.save_dir = save_dir if save_dir else '../predictors'
        
        self.model = NeuralNetwork(input_features=features[kernel_type])
        if weights is not None:
            self.model.load_state_dict(weights)
            print('successfully load similar device weights!')
        
    
    def create_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
    def create_loss_fn(self,loss_fn=None):
        if loss_fn == None:
            self.loss_fn = torchmetrics.MeanAbsolutePercentageError()
        else:
            self.loss_fn = loss_fn
       
    def train_one_epoch(self,dataloader,loss_fn):
        
        model = self.model
        model.train()
        size = len(dataloader.dataset)
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X, y
            pred = model(X)
            loss = loss_fn(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        self.scheduler.step()
        
        
    def train(self):
        epochs = self.epochs
        train_dataloader = self.get_dataloader(self.train_dataset)

        if self.loss_fn == None:
            self.create_loss_fn()
        loss_fn = self.loss_fn
        self.create_optimizer_and_scheduler()

        for t in tqdm(range(epochs)):
            self.train_one_epoch(train_dataloader,loss_fn)
            
        if self.eval_dataset:
            self.evaluate()

    def save(self):
        import os
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        savename = os.path.join(self.save_dir,self.kernel_type+'.pth')
        torch.save(self.model.state_dict(), savename)
        print(f'save model in {savename}')


    def evaluate(self):
        dataloader = self.get_dataloader(self.eval_dataset)
        
        if self.loss_fn == None:
            self.create_loss_fn()
        loss_fn = self.loss_fn
        num_batches = len(dataloader)
        model = self.model
        model.eval()
        test_loss = 0
        rmse_total, rmspe_total, error_total, acc5_total, acc10_total, acc15_total = 0,0,0,0,0,0
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                y1= y.cpu().numpy()
                pred1 = pred.cpu().numpy()
                rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(pred1, y1)
                rmse_total +=rmse
                rmspe_total += rmspe
                error_total += error
                acc5_total += acc5
                acc10_total += acc10
                acc15_total += acc15
        test_loss /= num_batches
        rmse_f = rmse_total / num_batches
        rmspe_f = rmspe_total / num_batches
        error_f = error_total / num_batches
        acc5_f = acc5_total / num_batches
        acc10_f = acc10_total / num_batches
        acc15_f = acc15_total / num_batches
        print(f"mlp: rmse: {rmse_f:.4f}; rmspe: {rmspe_f:.4f}; error: {error_f:.4f}; 5% accuracy: {acc5_f:.4f}; 10% accuracy: {acc10_f:.4f}; 15% accuracy: {acc15_f:.4f}.")
        print(f"Test Error: \n 10% Accuracy: {acc10_f:.4f}, Avg loss: {test_loss:>8f} \n")
        
        
    def get_dataloader(self,dataset):
        dataLoader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return dataLoader
         



