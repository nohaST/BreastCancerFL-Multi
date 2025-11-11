#!/usr/bin/env python3
"""
Flower Federated Learning Client.

This script launches a client that connects to a Flower server. The client's
dataset and model architecture are specified via command-line arguments.

Usage:
    python federated/client.py --dataset BUSI --model ResNet50V2
"""
import os
import random
import flwr as fl
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse  # Import for command-line parsing
import math

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, InceptionV3, DenseNet121

# ==============================================================================
# 1. CONFIGURATION AND DATA LOADING
# ==============================================================================

# --- DEFINE AVAILABLE DATASETS ---
# Maps dataset names to their respective paths.
AVAILABLE_DATASETS = {
    "BCMID": "./data/BCMID/",
    "BUSI": "./data/BUSI/",
    "BUS-UCLM": "./data/BUS-UCLM/"
}

def readData(categories, split_path):
    """Helper function to read image paths and labels."""
    data, labels = [], []
    for category in categories:
        folder_path = os.path.join(split_path, category)
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            data.append(img_path)
            labels.append(category)
    return data, labels

def load_data(dataset_name: str):
    """
    Loads and preprocesses the data for a specific client dataset.
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Available options: {list(AVAILABLE_DATASETS.keys())}")

    final_path = AVAILABLE_DATASETS[dataset_name]
    print(f"Loading data from path: {final_path}")
    
    # --- Client Name Extraction ---
    client_name = os.path.basename(os.path.normpath(final_path))
    print(f"Client Name Extracted: {client_name}")
    
    categories = ["Benign", "Malignant", "Normal"]
    train_data, train_labels = readData(categories, os.path.join(final_path, "train"))
    val_data, val_labels = readData(categories, os.path.join(final_path, "val"))
    test_data, test_labels = readData(categories, os.path.join(final_path, "test"))
    
    train_df = pd.DataFrame({"filename": train_data, "label": train_labels})
    val_df = pd.DataFrame({"filename": val_data, "label": val_labels})
    test_df = pd.DataFrame({"filename": test_data, "label": test_labels})
    
    data_gen = ImageDataGenerator(rescale=1./255)
    target_size = (224, 224)
    batch_size = 32

    train_gen = data_gen.flow_from_dataframe(
        train_df, x_col="filename", y_col="label", target_size=target_size,
        batch_size=batch_size, class_mode="categorical", shuffle=True, classes=categories
    )
    val_gen = data_gen.flow_from_dataframe(
        val_df, x_col="filename", y_col="label", target_size=target_size,
        batch_size=batch_size, class_mode="categorical", shuffle=False, classes=categories
    )
    test_gen = data_gen.flow_from_dataframe(
        test_df, x_col="filename", y_col="label", target_size=target_size,
        batch_size=batch_size, class_mode="categorical", shuffle=False, classes=categories
    )

    class_weights = compute_weights(train_gen)

    return train_gen, val_gen, test_gen, class_weights, client_name

def compute_weights(train_gen):
    """Computes class weights for handling imbalanced data."""
    class_indices = train_gen.class_indices
    train_labels_int = train_gen.labels
    class_weights = compute_class_weight("balanced", classes=np.unique(train_labels_int), y=train_labels_int)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    return class_weight_dict

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================

def create_model(model_name: str):
    """
    Creates and compiles a Keras model based on the specified name.
    """
    GLOBAL_SEED = 42
    tf.keras.utils.set_random_seed(GLOBAL_SEED)
    initializer = tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED)
    
    model_map = {
        "MobileNetV2": MobileNetV2,
        "ResNet50V2": ResNet50V2,
        "InceptionV3": InceptionV3,
        "DenseNet121": DenseNet121
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model '{model_name}' not supported. Choose from {list(model_map.keys())}")

    base_model_class = model_map[model_name]
    base_model = base_model_class(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    base_model.trainable = True # Set to True for fine-tuning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_initializer=initializer)(x)
    x = Dropout(0.3, seed=GLOBAL_SEED)(x)
    x = Dense(3, activation='softmax', kernel_initializer=initializer)(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
    
    print(f"Successfully built model: {model_name}")
    return model

# ==============================================================================
# 3. FLOWER CLIENT IMPLEMENTATION
# ==============================================================================

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_gen, val_gen, test_gen, class_weights, client_name):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.class_weights = class_weights
        self.class_labels = list(train_gen.class_indices.keys())
        self.client_name = client_name

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)

        history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=1,  # Train for one epoch per round
            verbose=1,
            class_weight=self.class_weights,
            callbacks=[early_stopping]
        )
        
        num_examples = self.train_gen.samples
        # Use the final epoch's metrics, as early stopping might not trigger in 1 epoch
        results = {
            "accuracy": history.history["accuracy"][-1],
            "loss": history.history["loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "client_name": self.client_name
        }
        return self.get_parameters(config), num_examples, results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_gen, verbose=0)

        self.test_gen.reset()
        y_true = self.test_gen.classes
        steps = math.ceil(self.test_gen.samples / self.test_gen.batch_size)
        y_pred = np.argmax(self.model.predict(self.test_gen, steps=steps), axis=1)

        cm = confusion_matrix(y_true, y_pred)
        cr_dict = classification_report(y_true, y_pred, target_names=self.class_labels, output_dict=True)

        print("\n--- Client-Side Evaluation ---")
        print(f"Dataset: {self.client_name}")
        print(classification_report(y_true, y_pred, target_names=self.class_labels))

        num_examples = self.test_gen.samples
        return loss, num_examples, {
            "accuracy": accuracy,
            "loss": loss,
            "confusion_matrix_json": json.dumps(cm.tolist()),
            "classification_report_json": json.dumps(cr_dict),
            "client_name": self.client_name
        }

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Federated Learning Client")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=AVAILABLE_DATASETS.keys(),
        help="The name of the dataset for this client."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["MobileNetV2", "ResNet50V2", "InceptionV3", "DenseNet121"],
        help="The model architecture to train."
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Address and port of the Flower server."
    )
    args = parser.parse_args()

    # 1. Load data for the specified dataset
    train_gen, val_gen, test_gen, class_weights, client_name = load_data(args.dataset)
    
    # 2. Create the specified model
    model = create_model(args.model)
    
    # 3. Instantiate the Flower client
    client = FederatedClient(model, train_gen, val_gen, test_gen, class_weights, client_name)
    
    # 4. Start the client
    print(f"\nStarting Flower client for dataset '{args.dataset}' with model '{args.model}'...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
        grpc_max_message_length=1024 * 1024 * 1024
    )
