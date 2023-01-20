import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    plt.title('Model trained on {}'.format(train_sim))
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


def format_output(data):
    """
    Format 1D data in a tuple of the form (inputs, output).
    Remove unnecessary columns from inputs array.
    Standardize every input column individually.
    """
    # Extract output from data table
    # (ExsituF column is the output we want to predict)
    y = np.asarray(data.pop('ExsituF')).astype('float32')

    # Clean up inputs by removing unneeded columns
    for remove_col in ['GalaxyID', 'Snapshot', 'TotalSpin',
                       'Aligns', 'Central']:
        data.pop(remove_col)

    # Standardize input columns to a mean value = 0 and std = 1
    # (Neural networks' performance is improved on standardized inputs)
    data = (data - data.mean()) / data.std()
    data = np.asarray(data).astype('float32')
    return data, y


def split_datasets(df):
    """
    Split a dataframe table to 3 datasets:
    Training set (80%): used during training.
    Test set (16%): used to evaluate model.
    Validation set (4%): used to avoid overfitting during training.
    """
    train, test = train_test_split(df, test_size=0.2)
    test, val = train_test_split(test, test_size=0.2)
    train_tuple = format_output(train)
    test_tuple = format_output(test)
    val_tuple = format_output(val)
    return train_tuple, test_tuple, val_tuple