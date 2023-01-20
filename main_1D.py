import os
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from nn_models import *
from utils import plot_predictions, get_predictions, split_datasets

# Load datasets
tng_data = pd.read_hdf('Data/properties_tng.h5py', '/data')
eagle_data = pd.read_hdf('Data/properties_eagle.h5py', '/data')

# Inspect the tables
print('Printing head of the TNG100 dataset...\n')
print(tng_data.head(3))

print('\nPrinting head of the EAGLE dataset...\n')
print(eagle_data.head(3))

# Split tng_data to train, test and validation test
(tng_train, tng_train_Y), (tng_test, tng_test_Y), \
        (tng_val, tng_val_Y) = split_datasets(tng_data)

train_set, train_set_Y, val_set, val_set_Y = tng_train, tng_train_Y, \
                                                        tng_val, tng_val_Y


# Build the NN model
model = build_model(train_set.shape[1])
model_id = 0

# Create directories for saving the model weights and the results
model_path = 'Saved_models/1D'
if not os.path.exists(model_path):
    os.makedirs(model_path)

results_path = 'Results/1D'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Train model
model_file_path = os.path.join(model_path,
                               'model_dense_{}.h5'.format(model_id))

es = EarlyStopping(monitor='val_loss', patience=50)
mc = ModelCheckpoint(filepath=model_file_path, monitor='val_mse',
                      save_best_only=True, save_weights_only=True)

# Train the model for 100 epochs
history = model.fit(train_set, train_set_Y,
                    epochs=10, batch_size=32,
                    validation_data=(val_set, val_set_Y), callbacks=[es])

# Test the model and print loss and mean-squared error
loss, mse = model.evaluate(x=tng_test, y=tng_test_Y)
print('loss: {}'.format(loss))
print('MSE: {}'.format(mse))

n_epochs = len(history.history['loss'])
with open(results_path +
          '/Results_{}.txt'.format(model_id), "w") as result_file:
    result_file.write("Trained for epochs: %s\n\n"
                      "Test loss, MSE: %s %s" % (n_epochs, loss, mse))

# Save the model weights to the specified path
# This way you can reload it later if you are satisfied with the training
model.save_weights(model_file_path)

# Evaluate model
y_samples_1, Y_pred1, y_std1 = get_predictions(tng_test, model)

plot_predictions(tng_test_Y, Y_pred1, y_std1,
                 'TNG', 'TNG', model_id, results_path)
