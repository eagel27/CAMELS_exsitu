import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from constants import BATCHES
from nn_data import input_fn_split, get_num_examples, get_data
from nn_models import *


def load_ds(mode, dataset_str):
    """
    Load the respective dataset by taking into account which channels to ignore
    and outside which mask radius to mask
    :param mode: 'train' or 'test' or 'validation'
    :param dataset_str: The dataset string identifying which dataset to load
    :return:
    """
    return input_fn_split(mode, dataset_str)


def get_predictions(test_set, model):
    """
    Load predictions from the trained model.
    The model predicts a distribution
    The mean value can be the model predicted ex-situ
    The std of the distribution corresponds to the uncertainty
    of the model on the prediction (error bar)
    """
    y_pred_distr = model(test_set)
    Y_pred = y_pred_distr.mean().numpy().reshape(-1)
    Y_pred_samples = y_pred_distr.sample(1000)
    y_std = y_pred_distr.stddev().numpy().reshape(-1)
    return Y_pred_samples, Y_pred, y_std


def plot_predictions(test_set_y, y_pred, y_std, test_sim, train_sim,
                     model_id, results_path):
    """
    Plot the 1-1 relation between the true values of ex-situ and
    the predicted values from the model.
    For that we will use the unseen by the model test set and
    we will plot 128 random galaxies from there.
    """
    plt.figure(figsize=(5, 5))
    plt.title('MLP trained on {}'.format(train_sim))
    plt.errorbar(test_set_y[:128], y_pred[:128],
                 yerr=y_std[:128], linestyle="None", fmt='o',
                capsize=3, color='blue', capthick=0.5)
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k--')
    plt.xlim(0, 1)
    plt.ylim(-0.1, 1.1)
    plt.ylabel('Predictions', fontsize=14)
    plt.xlabel('True Values', fontsize=14)
    plt.savefig('{}/{}_on_{}_model_{}'.format(results_path, train_sim,
                                              test_sim, model_id))


# Load datasets
dataset_str = 'tng_dataset'
ds_train = load_ds('train', dataset_str=dataset_str)
ds_val = load_ds('validation', dataset_str=dataset_str)
ds_test = load_ds('test', dataset_str=dataset_str)

len_ds_train = get_num_examples('train', dataset_str=dataset_str)
len_ds_val = get_num_examples('validation', dataset_str=dataset_str)
len_ds_test = get_num_examples('test', dataset_str=dataset_str)
print(len_ds_train, len_ds_val, len_ds_test)

# Build the CNN model
model = build_cnn_model(ds_train.shape)
model_id = 0

# Create directories for saving the model weights and the results
model_path = 'Saved_models'
if not os.path.exists(model_path):
    os.makedirs(model_path)

results_path = 'Results'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Train model
model_file_path = os.path.join(model_path,
                               'model_cnn_{}.h5'.format(model_id))

es = EarlyStopping(monitor='val_loss', patience=50)
mc = ModelCheckpoint(filepath=model_file_path, monitor='val_mse',
                     save_best_only=True, save_weights_only=True)

history = model.fit(ds_train,
                    epochs=50,
                    steps_per_epoch=len_ds_train // BATCHES,
                    validation_steps=len_ds_val // BATCHES,
                    validation_data=ds_val,
                    callbacks=[es, mc],
                    use_multiprocessing=True, workers=4)

# Evaluate model
images, y_true = get_data(ds_test, batches=8)
y_samples_1, Y_pred1, y_std1 = get_predictions(images, model)

plot_predictions(y_true, Y_pred1, y_std1,
                 'TNG', 'TNG', model_id, results_path)

