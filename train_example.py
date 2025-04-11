# This is a sample Python script.
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef, confusion_matrix, classification_report, multilabel_confusion_matrix
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os
import sys
from datetime import datetime


'''
LAST UPDATED 11/10/2021, lsdr
'''

## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep


os.chdir(OR_PATH) # Come back to the directory where the code resides , all files will be left on this directory

n_epoch = 30
BATCH_SIZE = 120
LR = 0.001

## Image processing
CHANNELS = 3
IMAGE_SIZE = 100

NICKNAME = "Metis"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

# CUSTOMIZATION VARIABLES FROM EXPERIMENTATION THROUGHOUT THE PROJECT
SHUFFLE_SEED = 1998 # Select a set seed for repeatable training as needed
CUSTOM_LOSS_WEIGHT = 'log' # Custom loss weights could be 1/examples <- recip, 1/ln(examples) <- log, 1/log10(examples) <- log10, etc. or None
ARCHITECTURE = 'ResNet18-Trainable4'

# For logging to a file and the terminal in case I briefly lose terminal connection
class DualLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


#---- Define the model ---- #
class Transfer_CNN(nn.Module):
    def __init__(self):
        super(Transfer_CNN, self).__init__()
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        # Freeze all layers before replacing the fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final classification layer as the only trainable layer, based on the starting recommendation from
        # the course PyTorch labs.
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 17)

    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.linear_input = 256

        # I have decided to put the padding in before processing the first layer to maintain size during initial processing rather than add zeroes after.
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=2)
        self.convnorm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.33)

        self.conv3 = nn.Conv2d(128, self.linear_input, (3, 3))
        self.convnorm3 = nn.BatchNorm2d(self.linear_input)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(self.linear_input, OUTPUTS_a)
        self.act = torch.relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.convnorm1(x) # Based on Medium article, batch norm before ReLU may sligthly improve performance.
        x = self.act(x)

        x = self.conv2(x)
        x = self.convnorm2(x)
        x = self.act(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.convnorm3(x)
        x = self.act(x)

        x = self.global_avg_pool(x).view(-1, self.linear_input)
        return self.linear(x)

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data, target_type):
        #Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        #Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        if self.type_data == 'train':
            y = xdf_dset.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_test.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")


        if self.target_type == 2:
            labels_ohe = [ int(e) for e in y]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)

            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            file = DATA_DIR + xdf_dset.id.get(ID)
        else:
            file = DATA_DIR + xdf_dset_test.id.get(ID)

        img = cv2.imread(file)

        img= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))

        # Augmentation only for train
        if self.type_data == 'train':
            X = self.train_transform(img)
        else:
            X = self.test_transform(img)

        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        return X, y


def read_data(target_type):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file


    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])

    ds_targets = xdf_dset['target_class']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)


    # Datasets
    partition = {
        'train': list_of_ids,
        'test' : list_of_ids_test
    }

    # Data Loaders

    # Add a fixed seed to the random generator, this will allow training to be reproducible,
    # rather than the order of training images being completely uncontrollable.
    gen = torch.Generator()
    gen.manual_seed(SHUFFLE_SEED)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'generator': gen}

    training_set = Dataset(partition['train'], 'train', target_type)
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, test_generator

def save_model(model):
    # Open the file

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

def model_definition(pretrained=False, loss_weights = None):
    # Define a Keras sequential model
    # Compile the model

    if pretrained == True:
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    else:
        model = Transfer_CNN() #CNN() testing
        print(model)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # If we have defined a custom set of loss weights, use those, otherwise, the normal loss function should be used
    if loss_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_weights).float().to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    # The provided scheduler could get down to basically no LR because a minimum was not provided.
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.33, patience=1, min_lr=1e-6)

    save_model(model)

    return model, optimizer, criterion, scheduler

# I have added an optional variable for custom loss weights.
def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on, pretrained = False, loss_weights = None):
    # Use a breakpoint in the code line below to debug your script.

    model, optimizer, criterion, scheduler = model_definition(pretrained, loss_weights)

    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = 0
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])
        test_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata,xtarget in train_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()

                output = model(xdata)

                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(train_hist) == 0:
                    train_hist = xtarget.cpu().numpy()
                else:
                    train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        test_loss, steps_test = 0, 0
        met_test = 0

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata,xtarget in test_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    optimizer.zero_grad()

                    output = model(xdata)

                    loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    cont += 1

                    steps_test += 1

                    test_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(test_hist) == 0:
                        tast_hist = xtarget.cpu().numpy()
                    else:
                        test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        #acc_test = accuracy_score(real_labels[1:], pred_labels)
        #hml_test = hamming_loss(real_labels[1:], pred_labels)

        avg_test_loss = test_loss / steps_test

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)


        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        print(xstrres)

        if met_test > met_test_best and SAVE_MODEL:

            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
            xdf_dset_results = xdf_dset_test.copy()

            ## The following code creates a string to be saved as 1,2,3,3,
            ## This code will be used to validate the model
            xfinal_pred_labels = []
            for i in range(len(pred_labels)):
                joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_test['results'] = xfinal_pred_labels
            xdf_dset_results['results'] = xfinal_pred_labels

            xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index = False)
            print("The model has been saved!")
            met_test_best = met_test


def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict

def process_target(target_type):
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''

    dict_target = {}
    xerror = 0

    if target_type == 2:
        ## The target comes as a string  x1, x2, x3,x4
        ## the following code creates a list
        target = np.array(xdf_data['target'].apply( lambda x : x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 1:
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names=(xtarget)
        xdf_data['target_class'] = final_target

    ## We add the column to the main dataset


    return class_names


from collections import Counter

def print_class_counts(df, label_col='target', name=''):
    # Convert comma-separated strings into a flat list of all class names
    label_lists = df[label_col].apply(lambda x: x.split(','))
    all_labels = [label for sublist in label_lists for label in sublist]

    # Count occurrences of each label
    counts = Counter(all_labels)
    count_array = []

    #print(f"\n############ PER-CLASS COUNT ({name}) ############")
    for label, count in sorted(counts.items(), key=lambda x: int(x[0].replace("class", ""))):
        #print(f"{label}: {count}")
        count_array.append(count)
    #print("############ END ############")
    return np.array(count_array)

if __name__ == '__main__':
    # Make all logs during this script also print to an output file for future reference
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = open(f'{timestamp}_{ARCHITECTURE}_{CUSTOM_LOSS_WEIGHT}.log', 'a')
    sys.stdout = DualLogger(sys.stdout, log_file)
    sys.stderr = sys.stdout

    print(f'Using CUSTOM LOSS WEIGHT: {CUSTOM_LOSS_WEIGHT} and Architecture: {ARCHITECTURE}')

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    ## Process Classes
    ## Input and output


    ## Processing Train dataset
    ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    class_names = process_target(target_type = 2)

    ## Comment

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    pos_class_count = print_class_counts(xdf_dset, name="TRAIN+VAL")

    xdf_dset_test= xdf_data[xdf_data["split"] == 'test'].copy()

    #print_class_counts(xdf_dset_test, name="TEST")

    ## read_data creates the dataloaders, take target_type = 2

    train_ds,test_ds = read_data(target_type = 2)

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    if CUSTOM_LOSS_WEIGHT:
        # Extract ground truth from train set
        print(CUSTOM_LOSS_WEIGHT)
        print(pos_class_count)
        EPS = 1+1e-7

        if CUSTOM_LOSS_WEIGHT == 'log10':
            lw = 1.0 / np.log10(pos_class_count+EPS)
        elif CUSTOM_LOSS_WEIGHT == 'log':
            lw = 1.0 / np.log(pos_class_count+EPS)
        elif CUSTOM_LOSS_WEIGHT == 'recip':
            lw = 1.0 / (pos_class_count+EPS)
        else:
            lw = None
    else:
        lw = None

    train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on='f1_macro', pretrained=False, loss_weights=lw)

    # Extract ground truth and predictions from test set
    y_true_str = xdf_dset_test['target_class'].tolist()
    y_pred_str = xdf_dset_test['results'].tolist()

    # Convert comma-separated strings to NumPy arrays
    y_true = np.array([list(map(int, s.split(','))) for s in y_true_str])
    y_pred = np.array([list(map(int, s.split(','))) for s in y_pred_str])

    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    mcm = multilabel_confusion_matrix(y_true, y_pred)

    for i, label in enumerate(class_names):
        print(f"\nConfusion matrix for class: {label}")
        print(pd.DataFrame(
            mcm[i],
            index=[f"True ≠ {label}", f"True = {label}"],
            columns=[f"Pred ≠ {label}", f"Pred = {label}"]
        ))