import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from utils import dataset_split, dataset_split_with_test


class RegressionExperiment:
    def __init__(self, tree_model, simple_model, config, model_leaves, tree_root, leaves, num_classes=1, use_test=True,
                 baselines=False):
        self.model_leaves = model_leaves
        self.tree_root = tree_root
        self.data_leaves = leaves
        self.tree_model = tree_model
        self.simple_model = simple_model
        self.config = config
        self.use_test = use_test
        self.num_classes = num_classes
        self.baselines = baselines

        if self.use_test:
            self.all_x, self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y, \
                self.train_idx, self.valid_idx, self.test_idx = dataset_split_with_test(self.data_leaves,
                                                                                        self.config['seed'],
                                                                                        torch_format=True)
        else:
            self.all_x, self.train_x, self.valid_x, self.train_y, self.valid_y, self.train_idx, self.valid_idx = \
                dataset_split(self.data_leaves, self.config['seed'], torch_format=True)
        self.loss_function = nn.MSELoss()#.cuda()
        self.calc_auc = False
        self.dendro_auc = None
        self.simple_auc = None
        if self.num_classes == 1:
            self.train_y = self.train_y.view(len(self.train_y), 1)
            self.valid_y = self.valid_y.view(len(self.valid_y), 1)
            if self.use_test:
                self.test_y = self.test_y.view(len(self.test_y), 1)

    def dendronet_train_epoch(self, optimizer, validation=False):
        optimizer.zero_grad()
        if self.baselines:
            preds = torch.cat([self.tree_model(self.model_leaves[i], self.data_leaves[i].bias_feature).reshape(1, self.num_classes) for i in self.train_idx])
        else:
            preds = torch.cat([self.tree_model.forward(self.model_leaves[i], self.all_x[i]).reshape(1, self.num_classes) for i in self.train_idx])#.cuda()
        delta_loss = self.tree_model.calc_delta_loss() * self.config['delta_penalty_factor']
        # delta_loss = torch.tensor([0.0]) #.cuda()
        # prediction_loss = self.loss_function(preds, self.train_y)#.cuda()
        prediction_loss = self.loss_function(preds, self.train_y)
        # loss = (prediction_loss + delta_loss)#.cuda()
        loss = prediction_loss + delta_loss
        loss.backward()
        optimizer.step()
        valid_auc = None
        if validation:
            with torch.no_grad():
                if self.baselines:
                    torch.cat([self.tree_model(self.model_leaves[i], self.data_leaves[i].bias_feature).reshape(1, self.num_classes) for i in self.valid_idx])
                else:
                    valid_preds = torch.cat(
                    [self.tree_model(self.model_leaves[i], self.all_x[i]).reshape(1, self.num_classes) for i in self.valid_idx])
                valid_loss = self.loss_function(valid_preds, self.valid_y)
                if self.calc_auc:  # only relevant in classification experiments
                    valid_auc = roc_auc_score(self.valid_y, torch.argmax(valid_preds.detach(), dim=1))
            return prediction_loss, delta_loss, valid_loss, valid_auc

        return prediction_loss, delta_loss

    # todo: add conditional auc calculation for classification experiments
    def simple_train_epoch(self, optimizer, validation=False):
        optimizer.zero_grad()
        if self.baselines:
            preds = torch.cat([self.simple_model(self.data_leaves[x_i].extended_features).reshape(1, self.num_classes) for x_i in self.train_idx])
        else:
            preds = torch.cat([self.simple_model(x_i).reshape(1, self.num_classes) for x_i in self.train_x])
        # loss = self.loss_function(preds, self.train_y)
        loss = self.loss_function(preds, self.train_y)
        loss.backward()
        optimizer.step()
        valid_auc = None
        if validation:
            with torch.no_grad():
                if self.baselines:
                    torch.cat(
                        [self.simple_model(self.data_leaves[x_i].extended_features).reshape(1, self.num_classes) for x_i
                         in self.valid_idx])
                else:
                    valid_preds = torch.cat([self.simple_model(x_i).reshape(1, self.num_classes) for x_i in self.valid_x])
                valid_loss = self.loss_function(valid_preds, self.valid_y)
                if self.calc_auc:  # only relevant in classification experiments
                    valid_auc = roc_auc_score(self.valid_y, torch.argmax(valid_preds.detach(), dim=1))
            return loss, valid_loss, valid_auc
        return loss

    def train_dendronet(self):
        delta_losses = list()
        prediction_losses = list()
        validation_losses = list()
        validation_aucs = list()
        # optimizer = torch.optim.Adam(self.tree_model.parameters(), lr=self.config['lr'])
        optimizer = torch.optim.SGD(self.tree_model.parameters(), lr=self.config['lr'])
        for epoch in range(self.config['epochs']):
            if epoch % self.config['validation_interval'] == 0:
                prediction_loss, delta_loss, validation_loss, validation_auc = self.dendronet_train_epoch(optimizer, validation=True)
                validation_losses.append(validation_loss)
                prediction_losses.append(prediction_loss)
                delta_losses.append(delta_loss)
                validation_aucs.append(validation_auc)
                print('validation loss:' + str(validation_loss))
            else:
                prediction_loss, delta_loss = self.dendronet_train_epoch(optimizer)

            if epoch + 1 == self.config['epochs']:
                print('Done training dendronet')
        return delta_losses, prediction_losses, validation_losses, validation_aucs

    def train_simple_model(self):
        prediction_losses = list()
        validation_losses = list()
        validation_aucs = list()
        # optimizer = torch.optim.Adam(self.simple_model.parameters(), lr=self.config['lr'])
        optimizer = torch.optim.SGD(self.simple_model.parameters(), lr=self.config['lr'])
        for epoch in range(self.config['epochs']):
            if epoch % self.config['validation_interval'] == 0:
                prediction_loss, validation_loss, validation_auc = self.simple_train_epoch(optimizer, validation=True)
                prediction_losses.append(prediction_loss)
                validation_losses.append(validation_loss)
                validation_aucs.append(validation_auc)
                print('validation loss:' + str(validation_loss))
            else:
                prediction_loss = self.simple_train_epoch(optimizer)
            if epoch + 1 == self.config['epochs']:
                print('Done training baseline model')
        return prediction_losses, validation_losses, validation_aucs


class ClassificationExperiment(RegressionExperiment):
    def __init__(self, tree_model, simple_model, config, model_leaves, tree_root, leaves, num_classes=2, use_test=False,
                 baselines=False):
        super(ClassificationExperiment, self).__init__(tree_model, simple_model, config, model_leaves, tree_root,
                                                       leaves, num_classes, use_test, baselines)
        self.loss_function = nn.CrossEntropyLoss()
        # cast y's to long format for loss function
        self.train_y = torch.tensor(self.train_y, dtype=torch.long)
        self.valid_y = torch.tensor(self.valid_y, dtype=torch.long)
        self.test_y = torch.tensor(self.test_y, dtype=torch.long)
        self.calc_auc = True
        self.dendro_auc = None
        self.simple_auc = None
