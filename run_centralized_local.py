#!/usr/bin/env python3
"""
Breast Cancer Classification: Centralized vs. Local Trainer.

This script trains and evaluates deep learning models on breast cancer ultrasound datasets.
Its behavior changes based on the number of datasets provided:

- Local Mode (one dataset): Trains a model on a single, specified dataset.
  Usage:
    python run_centralized_local.py --datasets BCMID

- Centralized Mode (multiple datasets): Pools data from all specified datasets
  and trains a single model on the combined data.
  Usage:
    python run_centralized_local.py --datasets BCMID BUSI BUS-UCLM
"""
import os
import random
import time
import csv
import argparse
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, DenseNet121, InceptionV3
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================================================================
# 1. CONFIGURATION SECTION
# ==============================================================================

# --- DEFINE AVAILABLE DATASETS ---
# NOTE: Using relative paths is recommended for portability.
# Replace with your absolute paths if needed.
AVAILABLE_DATASETS = {
    "BCMID": "./data/BCMID/",
    "BUSI": "./data/BUSI/",
    "BUS-UCLM": "./data/BUS-UCLM/"
}

# --- DEFINE MODELS TO RUN ---
MODELS = {
    "MobileNetV2": MobileNetV2,
    "ResNet50V2": ResNet50V2,
    "DenseNet121": DenseNet121,
    "InceptionV3": InceptionV3
}

# --- HYPERPARAMETERS & SETTINGS ---
INPUT_SHAPE = (224, 224, 3)
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
GLOBAL_SEED = 42
LOG_FILE = "./centralized_local_results.csv"
MODELS_SAVE_DIR = "./saved_models/"


# ==============================================================================
# 2. HELPER FUNCTIONS (Largely unchanged from your original script)
# ==============================================================================

def set_seeds(seed):
    """Sets all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.config.experimental.enable_op_determinism()
    return tf.keras.initializers.GlorotUniform(seed=seed)

def read_data_paths(split_path):
    """Reads image paths and labels from a directory structure."""
    categories = ["Benign", "Malignant", "Normal"]
    data, labels = [], []
    for category in categories:
        folder_path = os.path.join(split_path, category)
        if os.path.exists(folder_path):
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                data.append(img_path)
                labels.append(category)
    return data, labels

def log_experiment(
    filename: str, dataset_name: str, model_name: str, batchsize: int,
    epochs: int, num_actual_epochs: int, accuracy: float, f1: float, precision:float, recall:float,
    duration: float, report: str, path: str):
    """Logs the results of a single experiment to a CSV file."""
    fieldnames = [
         "timestamp", "dataset", "model", "accuracy", "f1", "precision", "recall", "batchsize", "epochs",
        "num_actual_epochs",  "report", "duration", "path"]
    log_entry = {
        "dataset": dataset_name,
        "timestamp": f"NOHA_{datetime.now().isoformat()}",
        "model": model_name,
        "batchsize": batchsize,
        "epochs": epochs,
        "num_actual_epochs": num_actual_epochs,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "duration": f"{duration:.2f}s",
        "report": report,
        "path": path
    }

    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    print(f" Logged experiment for model: {model_name} on dataset: {dataset_name}")

# ==============================================================================
# 3. CORE TRAINING & EVALUATION FUNCTION
# ==============================================================================

def train_and_evaluate(
    experiment_name, model_name, model_builder_func,
    train_gen, val_gen, test_gen, class_weight_dict
):
    """
    Handles the core logic of building, training, evaluating, and logging a model.
    This function is now reusable for both centralized and local modes.
    """
    print("\n" + "="*80)
    print(f"STARTING EXPERIMENT: Model '{model_name}' on Dataset '{experiment_name}'")
    print("="*80 + "\n")

    # --- 1. Set Seeds and Build Model ---
    initializer = set_seeds(GLOBAL_SEED)
    base_model = model_builder_func(input_shape=INPUT_SHAPE, weights='imagenet', include_top=False)
    base_model.trainable = True # Fine-tuning the whole model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_initializer=initializer)(x)
    x = Dropout(0.3, seed=GLOBAL_SEED)(x)
    x = Dense(len(train_gen.class_indices), activation='softmax', kernel_initializer=initializer)(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Successfully built model: {base_model.name}")

    # --- 2. Train the Model ---
    start_time = time.time()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )
    duration = time.time() - start_time

    # --- 3. Evaluate and Log Results ---
    print("\nEvaluating model after training:")
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys())
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    print(f"Final Test Accuracy: {acc:.2%}")
    print("Classification Report:\n", report)

    # --- 4. Save Model and Log ---
    model_save_path = os.path.join(
        MODELS_SAVE_DIR, f"{base_model.name}_{experiment_name}_{datetime.now().strftime('%Y%m%d')}.keras"
    )
    # model.save(model_save_path) # Uncomment to save the trained model file

    log_experiment(
        filename=LOG_FILE, dataset_name=experiment_name, model_name=base_model.name,
        batchsize=BATCH_SIZE, epochs=EPOCHS, num_actual_epochs=len(history.history['loss']),
        accuracy=acc, precision=precision, recall=recall, f1=f1, duration=duration, report=report, path=model_save_path
    )
    
    # --- 5. Clean up ---
    tf.keras.backend.clear_session()
    print(f"FINISHED EXPERIMENT: {model_name} on {experiment_name}")

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run centralized or local classification experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--datasets", required=True, nargs='+', choices=AVAILABLE_DATASETS.keys(),
        help="One or more datasets to run on.\n"
             "- Providing 1 dataset runs in LOCAL mode.\n"
             "- Providing >1 dataset runs in CENTRALIZED mode.\n"
             "Available: " + ", ".join(AVAILABLE_DATASETS.keys())
    )
    args = parser.parse_args()
    selected_dataset_names = args.datasets

    os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
    
    # --- Decide Mode Based on Number of Datasets ---
    
    if len(selected_dataset_names) == 1:
        # --- LOCAL MODE ---
        dataset_name = selected_dataset_names[0]
        print(f"\n--- Running in LOCAL mode for dataset: {dataset_name} ---")
        dataset_path = AVAILABLE_DATASETS[dataset_name]

        # Load data
        train_data, train_labels = read_data_paths(os.path.join(dataset_path, "train"))
        val_data, val_labels = read_data_paths(os.path.join(dataset_path, "val"))
        test_data, test_labels = read_data_paths(os.path.join(dataset_path, "test"))
        
        train_df = pd.DataFrame({"filename": train_data, "label": train_labels})
        val_df = pd.DataFrame({"filename": val_data, "label": val_labels})
        test_df = pd.DataFrame({"filename": test_data, "label": test_labels})

        # Create generators
        data_gen = ImageDataGenerator(rescale=1./255)
        classes = ["Benign", "Malignant", "Normal"]
        train_gen = data_gen.flow_from_dataframe(train_df, x_col="filename", y_col="label", target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True, classes=classes)
        val_gen = data_gen.flow_from_dataframe(val_df, x_col="filename", y_col="label", target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False, classes=classes)
        test_gen = data_gen.flow_from_dataframe(test_df, x_col="filename", y_col="label", target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False, classes=classes)

        # Calculate class weights
        class_indices = train_gen.class_indices
        train_labels_int = [class_indices[label] for label in train_labels]
        class_weights = compute_class_weight("balanced", classes=np.unique(train_labels_int), y=train_labels_int)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        # Loop through models and run experiments
        for model_name, model_builder_func in MODELS.items():
            train_and_evaluate(dataset_name, model_name, model_builder_func, train_gen, val_gen, test_gen, class_weight_dict)

    elif len(selected_dataset_names) > 1:
        # --- CENTRALIZED MODE ---
        experiment_name = "Centralized_" + "_".join(selected_dataset_names)
        print(f"\n--- Running in CENTRALIZED mode for datasets: {', '.join(selected_dataset_names)} ---")

        # Pool data from all specified datasets
        all_train_data, all_train_labels = [], []
        all_val_data, all_val_labels = [], []
        all_test_data, all_test_labels = [], []

        for name in selected_dataset_names:
            path = AVAILABLE_DATASETS[name]
            print(f"Pooling data from: {name}")
            
            train_data, train_labels = read_data_paths(os.path.join(path, "train"))
            val_data, val_labels = read_data_paths(os.path.join(path, "val"))
            test_data, test_labels = read_data_paths(os.path.join(path, "test"))

            all_train_data.extend(train_data)
            all_train_labels.extend(train_labels)
            all_val_data.extend(val_data)
            all_val_labels.extend(val_labels)
            all_test_data.extend(test_data)
            all_test_labels.extend(test_labels)

        # Create combined dataframes
        train_df = pd.DataFrame({"filename": all_train_data, "label": all_train_labels})
        val_df = pd.DataFrame({"filename": all_val_data, "label": all_val_labels})
        test_df = pd.DataFrame({"filename": all_test_data, "label": all_test_labels})

        print(f"\nTotal Pooled Data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test images.")

        # Create generators from the pooled dataframes
        data_gen = ImageDataGenerator(rescale=1./255)
        classes = ["Benign", "Malignant", "Normal"]
        train_gen = data_gen.flow_from_dataframe(train_df, x_col="filename", y_col="label", target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True, classes=classes)
        val_gen = data_gen.flow_from_dataframe(val_df, x_col="filename", y_col="label", target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False, classes=classes)
        test_gen = data_gen.flow_from_dataframe(test_df, x_col="filename", y_col="label", target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False, classes=classes)
        
        # Calculate class weights on the combined dataset
        class_indices = train_gen.class_indices
        train_labels_int = [class_indices[label] for label in all_train_labels]
        class_weights = compute_class_weight("balanced", classes=np.unique(train_labels_int), y=train_labels_int)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        # Loop through models and run experiments
        for model_name, model_builder_func in MODELS.items():
            train_and_evaluate(experiment_name, model_name, model_builder_func, train_gen, val_gen, test_gen, class_weight_dict)
            
    print("\nAll experiments have been completed!")
    print(f"Results have been logged to: {LOG_FILE}")
