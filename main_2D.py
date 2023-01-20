import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from constants import BATCHES
from nn_data import input_fn_split, get_num_examples, get_data
from nn_models import *
from utils import plot_predictions, get_predictions


# Load datasets
dataset_str = 'tng_dataset'
ds_train = input_fn_split('train', dataset_str=dataset_str)
ds_val = input_fn_split('validation', dataset_str=dataset_str)
ds_test = input_fn_split('test', dataset_str=dataset_str)

len_ds_train = get_num_examples('train', dataset_str=dataset_str)
len_ds_val = get_num_examples('validation', dataset_str=dataset_str)
len_ds_test = get_num_examples('test', dataset_str=dataset_str)
print(len_ds_train, len_ds_val, len_ds_test)

# Build the CNN model
model = build_cnn_model((128, 128, 5))
model_id = 0

# Create directories for saving the model weights and the results
model_path = 'Saved_models/2D'
if not os.path.exists(model_path):
    os.makedirs(model_path)

results_path = 'Results/2D'
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

# Save the model weights to the specified path
# This way you can reload it later if you are satisfied with the training
model.save_weights(model_file_path)

# Evaluate model
images, y_true = get_data(ds_test, batches=8)
y_samples_1, Y_pred1, y_std1 = get_predictions(images, model)

plot_predictions(y_true, Y_pred1, y_std1,
                 'TNG', 'TNG', model_id, results_path)

