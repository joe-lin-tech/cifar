import torch
import random
import os
import numpy as np
from argparse import ArgumentParser
from InquirerPy import prompt
from cnn import train_cnn
from resnet import train_resnet
from vit import train_vit

def validate_params(params):
    if "epochs" in params and params['epochs'] < 1:
        print("Error: Number of epochs must be >= 1.")
        exit(1)
    if "batch_size" in params and params['batch_size'] < 1:
        print("Error: Batch size must be >= 1.")
        exit(1)
    if "learning_rate" in params and params['learning_rate'] <= 0:
        print("Error: Learning rate must be positive.")
        exit(1)
    if not os.path.exists(params['save_folder']) and params['save_folder'] != "":
        print("Error: Save folder path must be a valid directory.")
        exit(1)

SEED = 2023
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


parser = ArgumentParser(prog='train.py', description='Train a deep learning classifier for CIFAR-10 images.')
parser.add_argument('-m', '--model', choices=['cnn', 'resnet', 'previt', 'vit'], help='class of model to train')
parser.add_argument('-e', '--epochs', type=int, help='number of epochs')
parser.add_argument('-b', '--batch-size', type=int, help='batch size')
parser.add_argument('-l', '--learning-rate', type=float, help='learning rate')
parser.add_argument('-d', '--device', choices=['cpu', 'mps', 'cuda'], default='cpu', help='device')
parser.add_argument('-c', '--cross-validate', action='store_true', help='enable 5-fold cross validation')
parser.add_argument('-w', '--wandb', dest='logging', action='store_true', help='enable wandb logging')
parser.add_argument('-s', '--save-folder', type=str, default='', help='path to save folder')
parser.add_argument('-f', '--ckpt-frequency', type=int, default=0, help='model checkpoint save frequency')

args = parser.parse_args()
params = { arg: getattr(args, arg) for arg in vars(args) if getattr(args, arg) is not None }
model = args.model

if model is None:
    models = {
        'Basic Convolutional Neural Network': "cnn",
        'ResNet': "resnet",
        'Vision Transformer (pretrained on ImageNet)': "previt",
        'Vision Transformer': "vit"
    }

    model = models[prompt({
        'type': 'list',
        'name': 'model',
        'message': 'Select a model architecture. (Use arrow keys)',
        'choices': models.keys()
    })['model']]

    hyperparameters = [
        { 'type': 'input', 'name': 'epochs', 'message': 'Input number of epochs.', 'default': '20' },
        { 'type': 'input', 'name': 'batch_size', 'message': 'Input a batch size.', 'default': '128' },
        { 'type': 'input', 'name': 'learning_rate', 'message': 'Input a learning rate.', 'default': '0.0001' }
    ]
    
    if model == "resnet":
        hyperparameters[0]['default'] = '50'
        hyperparameters[2]['default'] = '0.1'
    elif model == "previt" or model == "vit":
        hyperparameters[0]['default'] = '10'
        hyperparameters[1]['default'] = '32'
    
    params = prompt(hyperparameters)
    params['epochs'] = int(params['epochs'])
    params['batch_size'] = int(params['batch_size'])
    params['learning_rate'] = float(params['learning_rate'])

    device = prompt({
        'type': 'list',
        'name': 'device',
        'message': 'Select a device. (Use arrow keys)',
        'choices': ['cpu', 'mps', 'cuda']
    })['device']

    options = prompt({
        'type': 'checkbox',
        'name': 'options',
        'message': 'Toggle training options. (Use space to select)',
        'choices': ['5-Fold Cross Validation', 'Wandb Logging']
    })['options']

    save_folder = prompt({
        'type': 'input',
        'name': 'save_folder',
        'message': 'Input path to save folder. (Leave blank for current directory)'
    })['save_folder']

    ckpt_frequency = prompt({
        'type': 'input',
        'name': 'ckpt_frequency',
        'message': 'Input model checkpoint save frequency in number of epochs. (Input 0 to only save final model)'
    })['ckpt_frequency']

    params = { **params, "device": device, "cross_validate": "5-Fold Cross Validation" in options,
              "logging": "Wandb Logging" in options, "save_folder": save_folder }
    
validate_params(params)
    
cross_validate = params['cross_validate']
params.pop('cross_validate', None)
params.pop('model', None)
params['fold'] = -1

if model == 'previt':
    params['pretrained'] = True

train_functions = dict(cnn=train_cnn, resnet=train_resnet, previt=train_vit, vit=train_vit)

if cross_validate:
    for i in range(5):
        params['fold'] = i
        train_functions[model](**params)
else:
    train_functions[model](**params)