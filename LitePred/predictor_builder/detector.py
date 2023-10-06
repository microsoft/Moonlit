import os
import torch
from model import NeuralNetwork
import scipy.stats
class Detector:
    def __init__(self,pool_path,
                 dataset,
                 kernel_type):
        
        self.pool_path = pool_path
        self.dataset = dataset
        self.kernel_type = kernel_type
        
    def get_predictors_from_pool(self):
        self.predictors = {}
        predictors = os.listdir(self.pool_path)
        for predictor in predictors:
            weight_path = os.path.join(self.pool_path,predictor,f'{self.kernel_type}.pth')
            if os.path.exists(weight_path):
                self.predictors[predictor] = weight_path
    
    def get_pred_list(self):
        self.predlists = {}
        model = NeuralNetwork(input_features=7)  # for conv and dwconv 
        for device,predictor in self.predictors.items():
            model.load_state_dict(torch.load(predictor))
            self.predlists[device] = []
            for X,Y in self.dataset:
                self.predlists[device].append(model(X).item()) 
        return self.predlists
    
    def get_similar_score(self):
        self.scores = {}
        rep_list = [data[1].item() for data in self.dataset]
        for device,pred_list in self.predlists.items():
            self.scores[device] = round(scipy.stats.entropy(pred_list,rep_list),4) 
    
    def get_similar_device(self):
        self.get_predictors_from_pool()
        self.get_pred_list()
        self.get_similar_score()
        
        sorted_score = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_score[:3]
        top_3_device = [item[0] for item in top_3]
        
        return top_3_device
