import os
import time
import copy
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim

from utils import utils
from utils.EarlyStopping import EarlyStopping
from utils.GaussianSmoothing import GaussianSmoothing

from networks.UNet_PriCheXyNet import UNet
from networks.UNet_PrivacyNet import Unet2D_encoder
from networks.SiameseNetwork import SiameseNetwork


class AgentSiameseNetwork:
    def __init__(self, config):
        """This is the agent that provides code for re-training, validating and testing the patient verification model.

        :param config: dict
            A dictionary that stores the hyper-parameter configuration and some other important variables.
        """

        self.config = config

        # Set path used to save experiment-related files and results
        self.SAVINGS_PATH = './archive/' + self.config['experiment_description'] + '/'
        self.IMAGE_PATH = self.config['image_path']

        # Reproducibility
        utils.seed_all(42)

        # Set all the important variables
        self.perturbation_type = self.config['perturbation_type']
        self.blur_sigma = self.config.get('blur_sigma', None)
        self.elastic_mu = self.config.get('elastic_mu', None)
        self.perturbation_model_file = self.config['perturbation_model_file']
        self.mu = self.config['mu']

        # P1 2x2 attacker-pairing audit (added; both default to the released protocol so every
        # existing config is unaffected unless it opts in explicitly).
        #
        # snn_init: which checkpoint the attacker starts from.
        #   'imagenet'      (default) -- SiameseNetwork()'s own ImageNet-pretrained ResNet-50.
        #   'pretrained_ver' -- overwritten below with networks/pretrained_verification_model.pth.
        # train_pairing: whether the SECOND training/validation endpoint is sanitized.
        #   'sanitized_sanitized' (default) -- reproduces the released protocol exactly.
        #   'sanitized_raw'       -- only the query side is sanitized in training/validation too,
        #                            matching the fixed test-time asymmetry (test_snn is always
        #                            query-sanitized / reference-raw and is never controlled by
        #                            this flag).
        # Explicit string enums, not booleans: 'sanitized=true' would not say relative to what.
        self.snn_init = self.config.get('snn_init', 'imagenet')
        if self.snn_init not in ('imagenet', 'pretrained_ver', 'vstar'):
            raise ValueError("snn_init must be 'imagenet', 'pretrained_ver' or 'vstar', got "
                             + repr(self.snn_init))
        self.train_pairing = self.config.get('train_pairing', 'sanitized_sanitized')
        if self.train_pairing not in ('sanitized_sanitized', 'sanitized_raw'):
            raise ValueError("train_pairing must be 'sanitized_sanitized' or 'sanitized_raw', got "
                             + repr(self.train_pairing))
        # val_pairing: pairing of the checkpoint-selection (validation) loader and validation
        # loss. Defaults to train_pairing, so every pre-existing run is unchanged. Setting it
        # differently from train_pairing separates the training effect from the checkpoint-
        # selection effect in the SS/SR comparison.
        self.val_pairing = self.config.get('val_pairing', self.train_pairing)
        if self.val_pairing not in ('sanitized_sanitized', 'sanitized_raw'):
            raise ValueError("val_pairing must be 'sanitized_sanitized' or 'sanitized_raw', got "
                             + repr(self.val_pairing))

        self.b = self.config['b']
        self.m = self.config['m']
        self.eps = self.config['eps']

        self.image_size = self.config['image_size']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.max_epochs = self.config['max_epochs']
        self.early_stopping = self.config['early_stopping']

        self.num_workers = 16
        self.pin_memory = True

        # Define the identity grid and the 
        if self.perturbation_type == 'flow_field':
            d = torch.linspace(-1, 1, 256)
            mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
            grid_identity = torch.stack((mesh_y, mesh_x), 2)
            self.grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            self.gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
        elif self.perturbation_type in ['privacy_net', 'dp_pix', 'none', 'blur', 'elastic']:
            self.grid_identity = None
            self.gauss_filter = None
        else:
            raise Exception('Invalid argument: ' + self.perturbation_type)

        self.start_epoch = 0
        self.es = EarlyStopping(patience=self.early_stopping)
        self.best_loss = 100000
        self.loss_dict = {'training': [],
                          'validation': []}

        # Initialize the perturbation network
        if self.perturbation_type == 'none':
            self.n_channels = 3
            self.perturbation_net = None
        else:
            self.n_channels = 1
            if self.perturbation_type == 'flow_field':
                self.perturbation_net = UNet(1, 2, 32).cuda()
                self.perturbation_net.load_state_dict(torch.load(self.perturbation_model_file))
            elif self.perturbation_type == 'privacy_net':
                self.perturbation_net = Unet2D_encoder(self.n_channels, self.n_channels, 16).cuda()
                self.perturbation_net.load_state_dict(torch.load(self.perturbation_model_file))
            elif self.perturbation_type in ['dp_pix', 'blur', 'elastic']:
                # Both are applied inside the dataset, so there is no perturbation network.
                self.perturbation_net = None
            else:
                raise Exception('Invalid argument: ' + self.perturbation_type)
            

        # Define the siamese neural network architecture
        self.net = SiameseNetwork().cuda()
        if self.snn_init == 'pretrained_ver':
            # networks/pretrained_verification_model.pth is the SAME checkpoint file used by the
            # frozen/verifier-init arms elsewhere in this audit (Table 1's 'verifier' column);
            # loaded here strictly, so any architecture drift raises rather than silently
            # partial-loading.
            state = torch.load('./networks/pretrained_verification_model.pth')
            self.net.load_state_dict(state, strict=True)
        elif self.snn_init == 'vstar':
            state = torch.load('./networks/vstar_verifier.pth')
            self.net.load_state_dict(state, strict=True)
        self.best_net = copy.deepcopy(self.net)

        # Choose loss function
        self.loss = nn.BCEWithLogitsLoss().cuda()

        # Set the optimizer function
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Initialize data loaders
        # train_pairing only ever governs the training/validation loaders -- the test loader is
        # ALWAYS the released, fixed query-sanitized/reference-raw threat model, never a variable
        # under this study, so it does not take train_pairing at all (SiameseDataset defaults it
        # to 'sanitized_sanitized' for the testing phase, where that default is inert: the dataset
        # only reads self.train_pairing inside `self.phase in ('training', 'validation')` guards).
        self.training_loader = utils.get_data_loader(phase='training', experimental_step='retrainSNN', 
                                                     image_size=self.image_size, n_channels=self.n_channels, 
                                                     batch_size=self.batch_size, shuffle=True, 
                                                     num_workers=self.num_workers, pin_memory=self.pin_memory, 
                                                     b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH,
                                                     blur_sigma=self.blur_sigma,
                                                     elastic_mu=self.elastic_mu,
                                                     train_pairing=self.train_pairing)
        self.validation_loader = utils.get_data_loader(phase='validation', experimental_step='retrainSNN',
                                                       image_size=self.image_size, n_channels=self.n_channels, 
                                                       batch_size=self.batch_size, shuffle=False, 
                                                       num_workers=self.num_workers, pin_memory=self.pin_memory, 
                                                       b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH,
                                                     blur_sigma=self.blur_sigma,
                                                     elastic_mu=self.elastic_mu,
                                                     train_pairing=self.val_pairing)
        self.test_loader = utils.get_data_loader(phase='testing', experimental_step='retrainSNN', 
                                                 image_size=self.image_size, n_channels=self.n_channels, 
                                                 batch_size=self.batch_size, shuffle=False, 
                                                 num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                 b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH,
                                                     blur_sigma=self.blur_sigma,
                                                     elastic_mu=self.elastic_mu)

    def training_validation(self):
        os.makedirs(self.SAVINGS_PATH, exist_ok=True)
        # Training and validation loop
        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            training_loss = utils.train_snn(self.perturbation_type, self.net, self.perturbation_net, self.grid_identity, 
                                            self.gauss_filter, self.mu, self.training_loader, self.loss, self.optimizer, 
                                            epoch, self.max_epochs, train_pairing=self.train_pairing)
            validation_loss = utils.validate_snn(self.perturbation_type, self.net, self.perturbation_net, 
                                                 self.grid_identity, self.gauss_filter, self.mu, self.validation_loader, 
                                                 self.loss, epoch, self.max_epochs, train_pairing=self.val_pairing)

            self.loss_dict['training'].append(training_loss)
            self.loss_dict['validation'].append(validation_loss)

            end_time = time.time()
            print('Time elapsed for epoch ' + str(epoch + 1) + ': ' + str(
                round((end_time - start_time) / 60, 2)) + ' minutes')

            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                self.best_net = copy.deepcopy(self.net)

            torch.save(self.best_net.state_dict(), self.SAVINGS_PATH + self.config[
                'experiment_description'] + '_best_network.pth')

            utils.save_loss_curves_snn(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])
            utils.plot_loss_curves_snn(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])

            if self.es.step(validation_loss):
                break

        print('Finished Training!')
    
    def testing_evaluation(self):
        os.makedirs(self.SAVINGS_PATH, exist_ok=True)
        # Testing phase
        y_true, y_pred = utils.test_snn(self.perturbation_type, self.best_net, self.perturbation_net, 
                                        self.grid_identity, self.gauss_filter, self.mu, self.test_loader)

        y_true, y_pred = [y_true.numpy(), y_pred.numpy()]

        # Compute the evaluation metrics
        fp_rates, tp_rates, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        y_pred_thresh = utils.apply_threshold(y_pred, 0.5)
        accuracy, f1_score, precision, recall, report, confusion_matrix = utils.get_evaluation_metrics(y_true,
                                                                                                       y_pred_thresh)
        auc_mean, confidence_lower, confidence_upper = utils.bootstrap(1000,
                                                                       y_true,
                                                                       y_pred,
                                                                       self.SAVINGS_PATH,
                                                                       self.config['experiment_description'])

        # Plot ROC curve
        utils.plot_roc_curve(fp_rates, tp_rates, self.SAVINGS_PATH, self.config['experiment_description'])

        # Save all the results to files
        utils.save_labels_predictions(y_true, y_pred, y_pred_thresh, self.SAVINGS_PATH,
                                      self.config['experiment_description'])

        utils.save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix,
                                   self.SAVINGS_PATH, self.config['experiment_description'])

        utils.save_roc_metrics_to_file(fp_rates, tp_rates, thresholds, self.SAVINGS_PATH,
                                       self.config['experiment_description'])

        # Print the evaluation metrics
        print('EVALUATION METRICS:')
        print('AUC: ' + str(auc))
        print('Accuracy: ' + str(accuracy))
        print('F1-Score: ' + str(f1_score))
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('Report: ' + str(report))
        print('Confusion matrix: ' + str(confusion_matrix))

        print('BOOTSTRAPPING: ')
        print('AUC Mean: ' + str(auc_mean))
        print('Confidence interval for the AUC score: ' + str(confidence_lower) + ' - ' + str(confidence_upper))
    
    def run(self):
        # Call training/validation and testing loop successively
        self.training_validation()
        self.testing_evaluation()
