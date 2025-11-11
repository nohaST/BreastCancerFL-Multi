#!/usr/bin/env python3
"""
Flower Federated Learning Server (FedProx Strategy).

This script starts a Flower server using the FedProx strategy. It waits for a
specified number of clients to connect and runs for a given number of rounds.

Usage:
    python federated/server_fedprox.py --num_clients 2 --rounds 50 --proximal_mu 0.1
"""
import flwr as fl
import json
import os
import csv
import argparse  # Import argparse for command-line arguments
from typing import Dict, Tuple, Optional, List
from flwr.common import (
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
)
from flwr.server.strategy import FedProx
from flwr.server.client_proxy import ClientProxy
import numpy as np

# Define the path for the log file
EVAL_LOG_FILE = "evaluation_log_fedprox.csv"

# The Custom Strategy class remains unchanged
class PrintAccuracyStrategy(FedProx):
    client_names: List[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_csv_log()

    def _initialize_csv_log(self, client_names: Optional[List[str]] = None):
        """Creates the CSV file and writes the header."""
        if client_names is None or not client_names:
            self.csv_headers = ['round number', 'global_accuracy', 'global_loss']
            mode = 'w'
        else:
            self.client_names = sorted(client_names)
            self.csv_headers = ['round number']
            for name in self.client_names:
                self.csv_headers.extend([f'{name} acc', f'{name} f1', f'{name} cm', f'{name} cr'])
            mode = 'w'

        with open(EVAL_LOG_FILE, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()
            print(f"CSV log headers {'set' if client_names else 'initialized'} with: {self.csv_headers}")

    def _write_metrics_to_csv(self, server_round: int, round_data: Dict[str, Dict], global_acc: float, global_loss: float):
        """Writes the collected per-client data for the current round to CSV."""
        if not self.client_names:
            client_names_from_data = sorted(round_data.keys())
            if client_names_from_data:
                self._initialize_csv_log(client_names_from_data)

        def format_metric(metric_value):
            try:
                if isinstance(metric_value, dict):
                    metric_value = metric_value.get('weighted avg', {}).get('f1-score', 'N/A')
                return f"{float(metric_value):.4f}"
            except (ValueError, TypeError):
                return 'N/A'

        row = {'round number': server_round}
        for name in self.client_names:
            metrics = round_data.get(name, {})
            row[f'{name} acc'] = format_metric(metrics.get('accuracy', 'N/A'))
            row[f'{name} f1'] = format_metric(metrics.get('weighted_f1', 'N/A'))
            row[f'{name} cm'] = str(metrics.get('cm_json', 'N/A'))
            row[f'{name} cr'] = str(metrics.get('cr_json', 'N/A'))

        try:
            with open(EVAL_LOG_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writerow(row)
            print(f"Metrics for Round {server_round} successfully logged to {EVAL_LOG_FILE}")
        except Exception as e:
            print(f"ERROR writing to CSV file: {e}")

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """Aggregates evaluation results, prints local metrics, and logs to CSV."""
        if not results:
            print(f"Round {server_round}: No evaluation results received.")
            return None, {}

        round_data = {}
        losses = []
        accuracies = []
        sorted_results = sorted(results, key=lambda x: x[0].cid)

        print("\n" + "="*80)
        print(f"ROUND {server_round} EVALUATION SUMMARY")
        print("Client Local Metrics:")

        for client, res in sorted_results:
            client_id = client.cid
            metrics = res.metrics
            client_name = metrics.get("client_name", f"CID_{client_id}")
            loss = metrics.get("loss", 0.0)
            acc = metrics.get("accuracy", 0.0)
            losses.append(loss)
            accuracies.append(acc)

            cr_json = metrics.get("classification_report_json")
            weighted_f1 = 'N/A'
            if cr_json and cr_json != 'N/A':
                try:
                    cr_data = json.loads(cr_json)
                    weighted_f1 = cr_data.get('weighted avg', {}).get('f1-score', 'N/A')
                except json.JSONDecodeError:
                    pass

            cm_json = metrics.get("confusion_matrix_json")
            
            round_data[client_name] = {
                'accuracy': acc, 'weighted_f1': weighted_f1,
                'cm_json': cm_json if cm_json and cm_json != 'N/A' else 'N/A',
                'cr_json': cr_json if cr_json and cr_json != 'N/A' else 'N/A',
            }

            print(f"--- Client {client_name} (CID {client_id}) --- | Loss: {loss:.4f}, Accuracy: {acc:.4f}, Weighted F1: {weighted_f1 if isinstance(weighted_f1, str) else f'{weighted_f1:.4f}'}")

        if server_round == 1:
            self._initialize_csv_log(list(round_data.keys()))

        avg_loss = sum(losses) / len(losses)
        avg_accuracy = sum(accuracies) / len(accuracies)
        self._write_metrics_to_csv(server_round, round_data, avg_accuracy, avg_loss)

        print("="*80)
        print(f"Global Aggregated - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
        print("="*80 + "\n")

        return avg_loss, {"accuracy": avg_accuracy}

# Main execution block is now dynamic
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Flower Federated Learning Server (FedProx)")
    parser.add_argument(
        "--rounds",
        type=int,
        required=True,
        help="Total number of federated learning rounds.",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        required=True,
        help="Minimum number of clients to connect before starting training.",
    )
    parser.add_argument(
        "--proximal_mu",
        type=float,
        default=0.1,
        help="The proximal term mu for the FedProx strategy (default: 0.1)."
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Address and port for the server (default: 0.0.0.0:8080)."
    )
    args = parser.parse_args()

    # --- Initialize Strategy using Parsed Arguments ---
    strategy = PrintAccuracyStrategy(
        fraction_fit=1.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        proximal_mu=args.proximal_mu  # Use the parsed mu value
    )

    # --- Start Flower Server using Parsed Arguments ---
    print("\n" + "="*80)
    print("STARTING FEDPROX SERVER")
    print(f"Server address: {args.server_address}")
    print(f"Number of rounds: {args.rounds}")
    print(f"Waiting for {args.num_clients} clients to connect...")
    print(f"FedProx proximal_mu: {args.proximal_mu}")
    print("="*80 + "\n")
    
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024,
    )
