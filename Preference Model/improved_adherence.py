import os
import math
import json
import copy
import time
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import Counter

import pyomo.environ as pyo
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import seaborn as sns
import geopandas as gpd
from scipy.stats import wilcoxon

import contextily as cx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch

from src.base_adherence import Adherence

class TransitionModelAll(nn.Module):
    def __init__(self, input_dim, num_zones, hidden_dim, dropout):
        super().__init__()

        self.num_zones = num_zones

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_zones * num_zones)
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, self.num_zones * self.num_zones)

        # softmax por fila
        out = F.softmax(out, dim=1)

        return out

class TransitionModelNeighbors(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__()

        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, self.output_dim)

        # softmax por fila
        out = F.softmax(out, dim=1)

        return out

class TransitionModelEdge(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

class PreferenceLearning:

    def __init__(self, n_envio, num_zones, create_dataset=False, problem='tsp'):

        # Adherence class
        adh_class = Adherence(n_envio, num_zones)
        self.adh_class = adh_class

        # Load Data
        self.data_instances = adh_class.data_instances
        self.data_gps = adh_class.data_gps
        self.data_wores = adh_class.data_wores
        self.data_woreq = adh_class.data_woreq
        self.comunas = adh_class.comunas

        self.instances_ids = adh_class.instances_ids
        self.plans_ids = adh_class.plans_ids

        # Clustering
        self.num_zones = adh_class.num_zones
        self.num_communes = 51
        self.problem_type = problem

        # Create dataset
        inst_train, inst_test = train_test_split(adh_class.instances_ids, test_size=0.15, random_state=19)
        if create_dataset:

            self.create_dataset_all_transitions(inst_train, 'train', problem=problem)
            self.create_dataset_all_transitions(inst_test, 'test', problem=problem)

            self.create_dataset_neighbors(inst_train, 'train', problem=problem)
            self.create_dataset_neighbors(inst_test, 'test', problem=problem)

            self.create_dataset_edges(inst_train, 'train', problem=problem)
            self.create_dataset_edges(inst_test, 'test', problem=problem)

        # Instancias per set
        self.inst_train = inst_train
        self.inst_test = inst_test

    def get_clients_per_zone(self, dict_clusters_gps, num_zones):

        number_of_zones = {}
        for route, list_zones in dict_clusters_gps.items():
            list_zones = np.array(list_zones)
            zone_count = [(list_zones == i).sum() for i in range(num_zones)]
            number_of_zones[route] = zone_count

        return number_of_zones

    def get_transition_matrix(self, dict_zones_gps, num_clusters):

        # Matrix with transitions
        matrix_route = {}
        for route, list_zones in dict_zones_gps.items():

            transition_matrix = np.zeros((num_clusters, num_clusters))

            # We complete transition matrix
            arcs = [(list_zones[i], list_zones[i + 1]) for i in range(len(list_zones) - 1)]
            for arc in arcs:
                if arc[0] == arc[1]:
                    continue
                transition_matrix[arc] += 1

            # We normalize matrix to obtain probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            transition_matrix = transition_matrix / row_sums
            matrix_route[route] = transition_matrix

        return matrix_route

    def compute_base_adh(self, ordered_simpli_routes, dict_zones_simpli, dict_zones_gps):

        # We compute adherence using both methods
        adherence_list = []
        for route in ordered_simpli_routes.keys():

            ## Adherence without duplicate clusters
            ids_simpli = dict_zones_simpli[route]
            ids_gps = dict_zones_gps[route]

            if len(ids_simpli) == 1:
                arcs_simpli = [ids_simpli[0], ids_simpli[0]]
                arcs_gps = [ids_gps[0], ids_gps[0]]
            else:
                arcs_simpli = [(ids_simpli[i], ids_simpli[i + 1]) for i in range(len(ids_simpli) - 1)]
                arcs_gps = [(ids_gps[i], ids_gps[i + 1]) for i in range(len(ids_gps) - 1)]

            # Calculate adherence [%] as commons arcs over planned arcs
            count_simpli = Counter(arcs_simpli)
            count_gps = Counter(arcs_gps)
            common = 0
            for arc in count_simpli:
                common += min(count_simpli[arc], count_gps.get(arc, 0))
            total = len(arcs_simpli)
            adherence = common / total
            adherence_list.append(adherence)

        route_list = list(ordered_simpli_routes.keys())

        return adherence_list, route_list

    def create_dataset_all_transitions(self, instance_list, set_name, problem='tsp'):

        if problem == 'tsp':
            dir_base = 'tsp'
        else:
            dir_base = 'vrp'

        dir_datasets_commune = f"dataset_outputs/BuscaLibre/{dir_base}/comunas/all_clusters"
        os.makedirs(dir_datasets_commune, exist_ok=True)
        dir_datasets_zones = f"dataset_outputs/BuscaLibre/{dir_base}/K={self.num_zones}/all_clusters"
        os.makedirs(dir_datasets_zones, exist_ok=True)

        rows_communes = []
        rows_zones = []
        total_instances = len(instance_list)
        for instance in tqdm(instance_list, total=total_instances, desc=f"Procesando instancias {set_name} set"):

            # Get stops according to Simpli and GPS
            _, simpli_coordinates, ordered_stops, _, depot_instance = self.adh_class.get_stops_vehicles(instance)

            # Get clusters ids desde las paradas
            dict_communes_gps, dict_zones_gps = self.adh_class.get_zones(ordered_stops)

            # Remove duplicate since we don't learn probabilities of staying in the same cluster

            # We get clientes per zone and transition matrix
            clients_per_zone = self.get_clients_per_zone(dict_zones_gps, self.num_zones)
            clients_per_commune = self.get_clients_per_zone(dict_communes_gps, self.num_communes)
            transition_matrix_zones = self.get_transition_matrix(dict_zones_gps, self.num_zones)
            transition_matrix_communes = self.get_transition_matrix(dict_communes_gps, self.num_communes)

            # We complete dataset using all transitions and clusters
            for route in ordered_stops.keys():

                row_commune = {
                    "instance": instance,
                    "route": route,
                }

                for i in range(self.num_communes):
                    row_commune[f"clients_{i}"] = clients_per_commune[route][i]
                for i in range(self.num_communes):
                    for j in range(self.num_communes):
                        row_commune[f"trans_{i}_{j}"] = (
                            transition_matrix_communes[route][i, j]
                        )
                rows_communes.append(row_commune)

                row_zone = {
                    "instance": instance,
                    "route": route,
                }
                for i in range(self.num_zones):
                    row_zone[f"clients_{i}"] = clients_per_zone[route][i]
                for i in range(self.num_zones):
                    for j in range(self.num_zones):
                        row_zone[f"trans_{i}_{j}"] = (
                            transition_matrix_zones[route][i, j]
                        )
                rows_zones.append(row_zone)

        df_communes = pd.DataFrame(rows_communes)
        df_zones = pd.DataFrame(rows_zones)

        df_communes.to_csv(
            f"{dir_datasets_commune}/dataset_all_clusters_{set_name}.csv",
            index=False
        )
        df_zones.to_csv(
            f"{dir_datasets_zones}/dataset_all_clusters_{set_name}.csv",
            index=False
        )

        print(f"Dataset comunas guardado con {len(df_communes)} rutas")
        print(f"Dataset zonas guardado con {len(df_zones)} rutas")

    def create_dataset_neighbors(self, instance_list, set_name, problem='tsp'):

        if problem == 'tsp':
            dir_base = 'tsp'
        else:
            dir_base = 'vrp'

        dir_datasets_commune = f"dataset_outputs/BuscaLibre/{dir_base}/comunas/neighbors"
        os.makedirs(dir_datasets_commune, exist_ok=True)
        dir_datasets_zones = f"dataset_outputs/BuscaLibre/{dir_base}/K={self.num_zones}/neighbors"
        os.makedirs(dir_datasets_zones, exist_ok=True)

        # use adjacency file to filter by neighbors
        comunas_rm = [
            "Alhué", "Buin", "Calera de Tango", "Cerrillos", "Cerro Navia",
            "Colina", "Conchalí", "Curacaví", "El Bosque", "El Monte",
            "Estación Central", "Huechuraba", "Independencia", "Isla de Maipo",
            "La Cisterna", "La Florida", "La Granja", "La Pintana", "La Reina",
            "Lampa", "Las Condes", "Lo Barnechea", "Lo Espejo", "Lo Prado",
            "Macul", "Maipú", "María Pinto", "Melipilla", "Ñuñoa",
            "Padre Hurtado", "Paine", "Pedro Aguirre Cerda", "Peñaflor",
            "Peñalolén", "Pirque", "Providencia", "Pudahuel", "Puente Alto",
            "Quilicura", "Quinta Normal", "Recoleta", "Renca",
            "San Bernardo", "San Joaquín", "San José de Maipo", "San Miguel",
            "San Pedro", "San Ramón", "Santiago", "Tiltil", "Vitacura"]
        map_comunas = {i: c for i, c in enumerate(comunas_rm)}
        map_comunas_2 = {c: i for i, c in enumerate(comunas_rm)}

        with open(f"clustering/BuscaLibre/comunas/adjacency.json", "r") as f:
            adjacency_communes = json.load(f)
        with open(f"clustering/BuscaLibre/K={self.num_zones}/adjacency.json", "r") as f:
            adjacency_zones = json.load(f)

        rows_communes = []
        rows_zones = []
        total_instances = len(instance_list)
        for instance in tqdm(instance_list, total=total_instances, desc=f"Procesando instancias {set_name} set"):

            # Get stops according to Simpli and GPS
            _, simpli_coordinates, ordered_stops, _, depot_instance = self.adh_class.get_stops_vehicles(instance)

            # Get clusters ids desde las paradas
            dict_communes_gps, dict_zones_gps = self.adh_class.get_zones(ordered_stops)

            # We get clientes per zone and transition matrix
            clients_per_zone = self.get_clients_per_zone(dict_zones_gps, self.num_zones)
            clients_per_commune = self.get_clients_per_zone(dict_communes_gps, self.num_communes)
            transition_matrix_zones = self.get_transition_matrix(dict_zones_gps, self.num_zones)
            transition_matrix_communes = self.get_transition_matrix(dict_communes_gps, self.num_communes)

            # We complete dataset using neighbor transitions and clusters
            for route in ordered_stops.keys():

                row_commune = {
                    "instance": instance,
                    "route": route,
                }

                for i in range(self.num_communes):
                    row_commune[f"clients_{i}"] = clients_per_commune[route][i]
                for i in range(self.num_communes):
                    commune_name = map_comunas[i]
                    for commune_j in adjacency_communes[commune_name]:
                        j = map_comunas_2[commune_j]
                        if i==j:
                            continue
                        row_commune[f"trans_{i}_{j}"] = (
                            transition_matrix_communes[route][i, j]
                        )
                rows_communes.append(row_commune)


                row_zone = {
                    "instance": instance,
                    "route": route,
                }

                for i in range(self.num_zones):
                    row_zone[f"clients_{i}"] = clients_per_zone[route][i]
                for i in range(self.num_zones):
                    for j in adjacency_zones[str(i)]:
                        if i==j:
                            continue

                        row_zone[f"trans_{i}_{j}"] = (
                            transition_matrix_zones[route][i, j]
                        )
                rows_zones.append(row_zone)

        df_communes = pd.DataFrame(rows_communes)
        df_zones = pd.DataFrame(rows_zones)

        df_communes.to_csv(
            f"{dir_datasets_commune}/dataset_neighbors_{set_name}.csv",
            index=False
        )
        df_zones.to_csv(
            f"{dir_datasets_zones}/dataset_neighbors_{set_name}.csv",
            index=False
        )

        print(f"Dataset comunas guardado con {len(df_communes)} rutas")
        print(f"Dataset zonas guardado con {len(df_zones)} rutas")

    def create_dataset_edges(self, instance_list, set_name, problem='tsp'):

        if problem == 'tsp': dir_base = 'tsp'
        else: dir_base = 'vrp'

        dir_datasets_commune = f"dataset_outputs/BuscaLibre/{dir_base}/comunas/edges"
        os.makedirs(dir_datasets_commune, exist_ok=True)
        dir_datasets_zones = f"dataset_outputs/BuscaLibre/{dir_base}/K={self.num_zones}/edges"
        os.makedirs(dir_datasets_zones, exist_ok=True)

        rows_communes = []
        rows_zones = []
        total_instances = len(instance_list)
        for instance in tqdm(instance_list, total=total_instances, desc=f"Procesando instancias {set_name} set"):

            # Get stops according to Simpli and GPS
            _, simpli_coordinates, ordered_stops, _, depot_instance = self.adh_class.get_stops_vehicles(instance)

            # Get clusters ids desde las paradas
            dict_communes_gps, dict_zones_gps = self.adh_class.get_zones(ordered_stops)

            # We get clientes per zone and transition matrix
            clients_per_zone = self.get_clients_per_zone(dict_zones_gps, self.num_zones)
            clients_per_commune = self.get_clients_per_zone(dict_communes_gps, self.num_communes)
            transition_matrix_zones = self.get_transition_matrix(dict_zones_gps, self.num_zones)
            transition_matrix_communes = self.get_transition_matrix(dict_communes_gps, self.num_communes)

            # We complete dataset using edges
            for route in ordered_stops.keys():

                edge_rows = []
                for i in range(self.num_communes):
                    for j in range(self.num_communes):

                        if i==j: continue
                        c_i = clients_per_commune[route][i]
                        c_j = clients_per_commune[route][j]
                        if c_i == 0 or c_j == 0: continue

                        if transition_matrix_communes[route][i, j] == 0: y_ij = 0
                        else: y_ij = 1

                        row = {
                            "instance": instance,
                            "route": route,
                            "origin": i,
                            "destination": j,
                            "c_i": c_i,
                            "c_j": c_j,
                            "y_ij": y_ij
                        }
                        edge_rows.append(row)

                rows_communes.extend(edge_rows)

                edge_rows = []
                for i in range(self.num_zones):
                    for j in range(self.num_zones):

                        if i==j: continue
                        c_i = clients_per_zone[route][i]
                        c_j = clients_per_zone[route][j]
                        if c_i == 0 or c_j == 0: continue

                        if transition_matrix_zones[route][i, j] == 0: y_ij = 0
                        else: y_ij = 1

                        row = {
                            "instance": instance,
                            "route": route,
                            "origin": i,
                            "destination": j,
                            "c_i": c_i,
                            "c_j": c_j,
                            "y_ij": y_ij
                        }
                        edge_rows.append(row)

                rows_zones.extend(edge_rows)

        df_communes = pd.DataFrame(rows_communes)
        df_zones = pd.DataFrame(rows_zones)

        df_communes.to_csv(
            f"{dir_datasets_commune}/dataset_edges_{set_name}.csv",
            index=False
        )
        df_zones.to_csv(
            f"{dir_datasets_zones}/dataset_edges_{set_name}.csv",
            index=False
        )

        print(f"Dataset comunas guardado con {len(df_communes)} rutas")
        print(f"Dataset zonas guardado con {len(df_zones)} rutas")

    def loss_function(self, strategy, pred, target):

        if strategy == "edges":
            criterion = nn.BCEWithLogitsLoss()
            return criterion(pred, target)

        else:
            log_probs = torch.log_softmax(pred, dim=1)
            return -(target * log_probs).sum(dim=1).mean()

    def load_dataset(self, strategy, cluster):

        scaler = StandardScaler()
        # output_scaler = StandardScaler()
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        input_one = None

        if cluster == 'zones':
            cluster = f'K={self.num_zones}'

        dataset_dict = {}
        for set_name in ["train", "test"]:

            if cluster == "comunas":
                path = f"dataset_outputs/BuscaLibre/tsp/comunas/{strategy}/dataset_{strategy}_{set_name}.csv"
                df = pd.read_csv(path)
                num = self.num_communes

            else:
                path = f"dataset_outputs/BuscaLibre/tsp/K=50/{strategy}/dataset_{strategy}_{set_name}.csv"
                df = pd.read_csv(path)
                num = self.num_zones

            # Get input and output
            if strategy == "all_clusters":

                encoder_input = None
                data_clients = df[[f"clients_{i}" for i in range(num)]].values

                output = df[[f"trans_{i}_{j}" for i in range(num) for j in range(num)]].values

            elif strategy == "neighbors":

                encoder_input = None
                data_clients = df[[f"clients_{i}" for i in range(num)]].values

                output_cols = sorted([c for c in df.columns if c.startswith("trans_")])
                output = df[output_cols].values

            elif strategy == "edges":

                origin = df[["origin"]]
                destination = df[["destination"]]
                encoder_input = np.vstack([origin.values, destination.values])
                data_clients = df[["c_i", "c_j"]].values

                output = df[["y_ij"]].to_numpy()

            else:
                raise NotImplementedError


            if set_name == "train":

                if strategy == "edges":

                    encoder.fit(encoder_input)

                    origin_ohe = encoder.transform(origin)
                    destination_ohe = encoder.transform(destination)
                    input_one = np.hstack([origin_ohe, destination_ohe])

                    with open(f"dataset_outputs/BuscaLibre/tsp/{cluster}/{strategy}/Transformations/one_hot_encoder.pkl", "wb") as f:
                        pickle.dump(encoder, f)

                # else:
                #
                #     output = output_scaler.fit_transform(output)
                #
                #     with open(f"dataset_outputs/BuscaLibre/tsp/{cluster}/{strategy}/Transformations/output_scaler.pkl",
                #               "wb") as f:
                #         pickle.dump(output_scaler, f)

                data_clients = scaler.fit_transform(data_clients)

                with open(f"dataset_outputs/BuscaLibre/tsp/{cluster}/{strategy}/Transformations/clients_scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)

            else:

                with open(f"dataset_outputs/BuscaLibre/tsp/{cluster}/{strategy}/Transformations/clients_scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)

                if strategy == "edges":

                    with open(f"dataset_outputs/BuscaLibre/tsp/{cluster}/{strategy}/Transformations/one_hot_encoder.pkl", "rb") as f:
                        encoder = pickle.load(f)

                    origin_ohe = encoder.transform(origin)
                    destination_ohe = encoder.transform(destination)
                    input_one = np.hstack([origin_ohe, destination_ohe])

                # else:
                #
                #     with open(
                #             f"dataset_outputs/BuscaLibre/tsp/{cluster}/{strategy}/Transformations/output_scaler.pkl",
                #             "rb"
                #     ) as f:
                #         output_scaler = pickle.load(f)
                #
                #     output = output_scaler.transform(output)

                data_clients = scaler.transform(data_clients)

            if strategy == "edges":
                X_final = np.hstack([input_one, data_clients])
            else:
                X_final = np.hstack([data_clients])
            dataset_dict[set_name] = {
                "X": X_final.astype(np.float32),
                "Y": output.astype(np.float32)
            }

        return dataset_dict

    def train_model(self, epochs=1500, patience=30, n_splits=5):

        torch.manual_seed(42)
        np.random.seed(42)

        # for strategy in ['all_clusters', 'neighbors', 'edges']:
        for strategy in ['all_clusters', 'neighbors']:

            print('Training NN with strategy {}'.format(strategy))

            # for cluster in ['comunas', 'zones']:
            for cluster in ['zones']:

                print('clustering using {}'.format(cluster))
                print()

                dataset_dict = self.load_dataset(strategy, cluster)

                X = dataset_dict["train"]["X"]
                Y = dataset_dict["train"]["Y"]
                input_dim = X.shape[1]
                output_dim = Y.shape[1]

                if strategy == "edges":
                    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                else:
                    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

                (
                    best_params,
                    best_grid_metric,
                    best_epochs,
                    best_curves_loss,
                    best_curves_metric
                ) = self.grid_search(
                    X,
                    Y,
                    input_dim,
                    output_dim,
                    kfold,
                    epochs,
                    patience,
                    strategy,
                    cluster
                )

                print("\nBEST PARAMS")
                print(best_params)
                if strategy == "edge":
                    print(f"Best CV AUC: {best_grid_metric:.4f}")
                print(f"Best Epochs: {best_epochs}")

                # save curves
                for fold in range(n_splits):
                    if strategy == "edges":
                        train_curve, valid_curve = best_curves_metric[fold]
                    else:
                        train_curve, valid_curve = best_curves_loss[fold]
                    self.plot_training_curve(strategy, cluster, train_curve, valid_curve, fold)

                model = self.train_final_model(
                    strategy,
                    cluster,
                    X,
                    Y,
                    input_dim,
                    output_dim,
                    best_params,
                    best_epochs
                )

                X_test = dataset_dict["test"]["X"]
                Y_test = dataset_dict["test"]["Y"]

                test_dataset = TensorDataset(
                    torch.from_numpy(X_test),
                    torch.from_numpy(Y_test)
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=32,
                    shuffle=False
                )
                test_loss, test_metric = self.evaluate(strategy, model, test_loader)

                print(f"\nTest Loss: {test_loss:.4f}")
                if strategy == "edges":
                    print(f"Test AUC: {test_metric:.4f}")

    def grid_search(self, X, Y, input_dim, output_dim, kfold, epochs, patience, strategy, cluster):

        grid = {
            "hidden_dim": [16, 32, 64],
            "lr": [1e-2, 1e-3, 1e-4],
            "dropout": [0.0, 0.2, 0.4]
        }

        best_grid_metric = -np.inf
        best_params = None
        best_epochs = None
        best_curves_loss = None
        best_curves_metric = None

        grid_results = []

        for hidden_dim in grid["hidden_dim"]:
            for lr in grid["lr"]:
                for dropout in grid["dropout"]:

                    print(
                        f"Grid Search | "
                        f"hidden={hidden_dim} "
                        f"lr={lr} "
                        f"dropout={dropout}"
                    )

                    fold_metrics = []
                    fold_epochs = []
                    fold_curves = []
                    fold_metric_curves = []

                    if strategy == "edges":
                        y_split = Y.ravel()
                    else:
                        y_split = None

                    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y_split)):

                        result = self.train_fold(
                            X[train_idx],
                            Y[train_idx],
                            X[val_idx],
                            Y[val_idx],
                            input_dim,
                            hidden_dim,
                            output_dim,
                            dropout,
                            lr,
                            epochs,
                            patience,
                            strategy,
                            cluster
                        )

                        fold_metrics.append(result["best_metric"])
                        fold_epochs.append(result["best_epoch"])
                        fold_curves.append((result["train_losses"], result["valid_losses"]))
                        fold_metric_curves.append((result["train_metric"], result["valid_metric"]))

                    mean_metric = np.mean(fold_metrics)
                    mean_epoch = int(np.round(np.mean(fold_epochs)))

                    if strategy == "edges":
                        print(f"Mean CV AUC: {mean_metric:.4f}")
                        print()

                    grid_results.append({
                        "hidden_dim": hidden_dim,
                        "lr": lr,
                        "dropout": dropout,
                        "mean_metric": mean_metric,
                        "mean_epoch": mean_epoch
                    })

                    if mean_metric > best_grid_metric:
                        best_grid_metric = mean_metric
                        best_params = {
                            "hidden_dim": hidden_dim,
                            "lr": lr,
                            "dropout": dropout
                        }
                        best_epochs = mean_epoch
                        best_curves_loss = copy.deepcopy(fold_curves)
                        best_curves_metric = copy.deepcopy(fold_metric_curves)

        df_grid = pd.DataFrame(grid_results)

        if cluster=='zones':
            cluster = 'K=50'
        df_grid.to_csv(f"dataset_outputs/BuscaLibre/tsp/{cluster}/{strategy}/grid_search_results_{strategy}.csv", index=False)

        return best_params, best_grid_metric, best_epochs, best_curves_loss, best_curves_metric

    def train_fold(
            self,
            X_train,
            Y_train,
            X_val,
            Y_val,
            input_dim,
            hidden_dim,
            output_dim,
            dropout,
            lr,
            epochs,
            patience,
            strategy,
            cluster
    ):

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        if cluster == "comunas":
            num = self.num_communes
        else:
            num = self.num_zones

        model = self.get_model(strategy, num, input_dim, hidden_dim, output_dim, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        best_metric = -np.inf
        best_epoch = 0
        epochs_without_improvement = 0

        train_losses = []
        valid_losses = []
        train_metric = []
        valid_metric = []

        for epoch in range(epochs):

            model.train()

            for batch_X, batch_Y in train_loader:

                optimizer.zero_grad()
                logits = model(batch_X.float())
                loss = self.loss_function(strategy, logits, batch_Y.float())
                loss.backward()
                optimizer.step()

            train_loss, train_metric_epoch = self.evaluate(strategy, model, train_loader)
            val_loss, val_metric = self.evaluate(strategy, model, valid_loader)

            train_metric.append(train_metric_epoch)
            valid_metric.append(val_metric)
            train_losses.append(train_loss)
            valid_losses.append(val_loss)

            if val_metric > best_metric:
                best_metric = val_metric
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        return {
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_metric": train_metric,
            "valid_metric": valid_metric
        }

    def get_model(self, strategy, num, input_dim, hidden_dim, output_dim, dropout):

        if strategy == "all_clusters":
            return TransitionModelAll(input_dim, num, hidden_dim, dropout)
        elif strategy == "neighbors":
            return TransitionModelNeighbors(input_dim, output_dim, hidden_dim, dropout)
        else:
            return TransitionModelEdge(input_dim, hidden_dim, dropout)

    def evaluate(self, strategy, model, loader):

        model.eval()
        total_loss = 0

        all_probs = []
        all_targets = []
        with torch.no_grad():

            for X, Y in loader:
                logits = model(X.float())
                loss = self.loss_function(strategy, logits, Y.float())
                total_loss += loss.item()

                if strategy == "edges":
                    probs = torch.sigmoid(logits)
                    all_probs.extend(probs.cpu().numpy().flatten())
                    all_targets.extend(Y.cpu().numpy().flatten())

        avg_loss = total_loss / len(loader)

        if strategy == "edges":
            metric = roc_auc_score(all_targets, all_probs)
        else:
            metric = -avg_loss

        return avg_loss, metric

    def plot_training_curve(self, strategy, cluster, train_losses, val_losses, fold):

        plt.figure(figsize=(8, 5))

        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")

        plt.xlabel("Epoch")
        if strategy == "edges":
            plt.ylabel("AUC")
        else:
            plt.ylabel("Loss")
        plt.title("Learning Curve")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if cluster == "zones":
            cluster = "K=50"
        plt.savefig(f"dataset_outputs/BuscaLibre/tsp/{cluster}/{strategy}/learning_curves/learning_curve_{fold}.png", dpi=300)
        plt.close()

    def train_final_model(self, strategy, cluster, X, Y, input_dim, output_dim, best_params, best_epochs):

        train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        if cluster == "comunas":
            num = self.num_communes
        else:
            num = self.num_zones
        model = self.get_model(strategy, num, input_dim, best_params["hidden_dim"], output_dim, best_params["dropout"])

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=1e-4
        )

        train_losses = []
        for epoch in range(best_epochs):

            model.train()
            total_loss = 0

            for batch_X, batch_Y in train_loader:

                optimizer.zero_grad()
                logits = model(batch_X.float())
                loss = self.loss_function(strategy, logits, batch_Y.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_loss = (
                    total_loss /
                    len(train_loader)
            )

            train_losses.append(
                train_loss
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{best_epochs} "
                f"Loss={train_loss:.4f}"
            )

        if cluster == "zones":
            cluster = "K=50"
        torch.save(
            model.state_dict(),
            f"dataset_outputs/BuscaLibre/tsp/{cluster}/{strategy}/best_model_{strategy}.pt"
        )

        return model

    # def test_feature_importance(self, X_valid, Y_valid):
    #
    #     # We test feature importance using valid set
    #     X_valid = torch.from_numpy(X_valid)
    #     Y_valid = torch.from_numpy(Y_valid)
    #     self.model.eval()
    #
    #     num_vehicles = self.input_dim - self.num_zones
    #     features_names = [f'veh_{i}' for i in range(num_vehicles)] + [f'n_cli_{j}' for j in range(self.num_zones)]
    #     results_df = pd.DataFrame(columns=features_names)
    #
    #     ## Gradient-based Importance
    #     X_valid_grad = X_valid.clone().detach().requires_grad_(True)
    #
    #     # Forward pass
    #     outputs = self.model(X_valid_grad.float())
    #     loss = self.transition_loss(outputs, Y_valid)
    #
    #     # Backward pass
    #     loss.backward()
    #
    #     # Calculate feature importance based on gradients
    #     gradient_importance = torch.mean(torch.abs(X_valid_grad.grad), dim=0)
    #     gradient_importance_array = gradient_importance.detach().numpy()
    #     results_df.loc[0] = gradient_importance_array
    #
    #     ## Permutation Importance
    #     X_valid_feat = X_valid
    #     with torch.no_grad():
    #         outputs = self.model(X_valid_feat.float())
    #         baseline_loss = self.transition_loss(outputs, Y_valid)
    #
    #     # Calculate permutation importance
    #     num_samples = len(X_valid)
    #     num_permutations = 10
    #     feature_importance = []
    #     for i in range(self.input_dim):
    #         importance_sum = 0
    #         for _ in range(num_permutations):
    #             X_permuted = X_valid_feat.clone()
    #             X_permuted[:, i] = X_permuted[torch.randperm(num_samples), i]
    #             with torch.no_grad():
    #                 outputs = self.model(X_permuted.float())
    #                 permuted_loss = self.transition_loss(outputs, Y_valid)
    #             importance = permuted_loss - baseline_loss
    #             importance = importance.detach().numpy()
    #             importance_sum += importance
    #         feature_importance.append(importance_sum / num_permutations)
    #     results_df.loc[1] = feature_importance
    #
    #     results_df.to_csv("dataset_outputs/comunas/feature_importance.csv")

def correct_list(list, mode='stops'):
    if mode == 'stops':
        list = [int(val) - 1 for val in list]
    else:
        list = [int(val) for val in list]
    final_list = [list[0]]
    for val in list[1:]:
        if val != final_list[-1]:
            final_list.append(int(val))
    final_list.append(int(final_list[0]))
    return final_list

def compute_adherence_lists(planned_list, real_list):

    arcs_simpli = [
        (planned_list[i], planned_list[i + 1])
        for i in range(len(planned_list) - 1)
    ]
    arcs_simpli.append(
        (planned_list[-1], planned_list[0])
    )

    arcs_gps = [
        (real_list[i], real_list[i + 1])
        for i in range(len(real_list) - 1)
    ]
    arcs_gps.append(
        (real_list[-1], real_list[0])
    )

    count_simpli = Counter(arcs_simpli)
    count_gps = Counter(arcs_gps)

    common = 0
    for arc in count_simpli:
        common += min(
            count_simpli[arc],
            count_gps.get(arc, 0)
        )

    return common / len(arcs_simpli)

class PreferenceImprovedAdherence:

    def __init__(self, n_envio, num_zones=50, problem='tsp'):

        adh_class = Adherence(n_envio, num_zones)
        pref_class = PreferenceLearning(n_envio=7, num_zones=50, problem=problem)

        self.adh_class = adh_class
        self.pref_class = pref_class

        self.num_zones = self.adh_class.num_zones
        self.problem_type = problem

    def compute_stop_adherence(
            self,
            ordered_planned,
            ordered_real
    ):

        adh = []
        routes = []

        for route in ordered_planned.keys():
            planned = ordered_planned[route]
            real = ordered_real[route]

            adh.append(
                compute_adherence_lists(
                    planned,
                    real
                )
            )

            routes.append(route)

        return adh, routes

    def compute_improved_adherence_all_instances(
            self,
            max_time_hours= 10
    ):

        start_time = time.time()
        max_time = max_time_hours * 3600

        rows = []

        for instance in self.adh_class.instances_ids:

            if instance != 2918:
                continue

            elapsed = time.time() - start_time

            if elapsed > max_time:
                print(f"\nTime limit reached ({elapsed / 3600:.2f} h). Stopping...")
                break

            # try:
            if True:

                print(f"Processing instance {instance} (elapsed {elapsed / 60:.1f} min)")

                # Improved (NN)
                # adherence_stops_all, adherence_zones_all, ordered_simpli_all, dict_zones_all = self.compute_improved_adherence_instance(
                #     instance, strategy='all_clusters')
                # adherence_stops_neighbors, adherence_zones_neighbors, ordered_simpli_neighbors, dict_zones_neighbors = self.compute_improved_adherence_instance(
                #     instance, strategy='neighbors')
                adherence_stops_edge, adherence_zones_edge, ordered_simpli_edge, dict_zones_edge = self.compute_improved_adherence_instance(
                    instance, strategy='edges')

                # Base TSP
                _, _, ordered_tsp = self.adh_class.get_routes_tsp(instance, lib="gurobi")
                _, dict_zones_tsp = self.adh_class.get_zones(ordered_tsp)

                # GPS route
                _, _, ordered_stops, _, _ = self.adh_class.get_stops_vehicles(instance)
                _, dict_zones_gps = self.adh_class.get_zones(ordered_stops)

                # Base adherence
                adherence_stops, adherence_zones = self.compute_adh(ordered_stops, ordered_tsp, dict_zones_gps, dict_zones_tsp)

                # Plot routes
                self.plot_improved_route(instance, ordered_stops, ordered_tsp, ordered_simpli_edge, dict_zones_gps, dict_zones_tsp, dict_zones_edge)

                routes = list(dict_zones_gps.keys())
                # Save results
                for (
                        r,
                        a_stop_tsp,
                        a_stop_all,
                        a_stop_neigh,
                        a_stop_edge,
                        a_zone_tsp,
                        a_zone_all,
                        a_zone_neigh,
                        a_zone_edge
                ) in zip(
                    routes,
                    adherence_stops,
                    adherence_stops_all,
                    adherence_stops_neighbors,
                    adherence_stops_edge,
                    adherence_zones,
                    adherence_zones_all,
                    adherence_zones_neighbors,
                    adherence_zones_edge
                ):
                    rows.append({
                        "instance": instance,
                        "route": r,

                        "adh_stops_tsp": a_stop_tsp,
                        "adh_stops_all": a_stop_all,
                        "adh_stops_neighbors": a_stop_neigh,
                        "adh_stops_edge": a_stop_edge,

                        "adh_zones_tsp": a_zone_tsp,
                        "adh_zones_all": a_zone_all,
                        "adh_zones_neighbors": a_zone_neigh,
                        "adh_zones_edge": a_zone_edge,
                    })

                pd.DataFrame(rows).to_csv(
                    "results_improved_adherence.csv",
                    index=False
                )

            # except Exception as e:
            #     print(f"Instance {instance}: {e}")

        pd.DataFrame(rows).to_csv(
            "results_improved_adherence.csv",
            index=False
        )

        print(
            f"\nFinished. "
            f"Processed {len(rows)} routes."
        )

    def _compute_adherence_lists(self, dict_gps, dict_tsp):

        adherence_list = []

        for route in dict_gps.keys():

            if route not in dict_tsp:
                continue

            ids_tsp = dict_tsp[route]
            ids_gps = dict_gps[route]

            if len(ids_tsp) == 1:
                arcs_tsp = [(ids_tsp[0], ids_tsp[0])]
                arcs_gps = [(ids_gps[0], ids_gps[0])]
            else:
                arcs_tsp = [
                    (ids_tsp[i], ids_tsp[i + 1])
                    for i in range(len(ids_tsp) - 1)
                ]
                arcs_gps = [
                    (ids_gps[i], ids_gps[i + 1])
                    for i in range(len(ids_gps) - 1)
                ]

            count_tsp = Counter(arcs_tsp)
            count_gps = Counter(arcs_gps)

            common = sum(
                min(count_tsp[arc], count_gps.get(arc, 0))
                for arc in count_tsp
            )

            adherence = common / len(count_tsp)
            adherence_list.append(adherence)

        return adherence_list

    def compute_adh(self, stops_gps, stops_tsp, dict_zones_gps, dict_zones_tsp):

        adherence_stops = self._compute_adherence_lists(stops_gps, stops_tsp)
        adherence_zones = self._compute_adherence_lists(dict_zones_gps, dict_zones_tsp)

        return adherence_stops, adherence_zones

    def compute_improved_adherence_instance(self, instance, strategy):

        # GPS
        _, _, ordered_stops, _, depot_instance = self.adh_class.get_stops_vehicles(instance)
        _, dict_zones_gps = self.adh_class.get_zones(ordered_stops)

        # TSP + NN
        ordered_simpli_nn = self.get_improved_routes_tsp(instance, ordered_stops, depot_instance, strategy)
        _, dict_zones_nn = self.adh_class.get_zones(ordered_simpli_nn)

        # Compute adherence
        adherence_stops, adherence_zones = self.compute_adh(ordered_stops, ordered_simpli_nn, dict_zones_gps, dict_zones_nn)

        return adherence_stops, adherence_zones, ordered_simpli_nn, dict_zones_nn

    def create_improved_vrp_pyomo(self, ordered_stops):

        ## Create a concrete model for the TSP
        model = pyo.ConcreteModel(name='VRP')

        ## Parameters of the model
        nodes_gps = []
        for route in ordered_stops:
            nodes_gps_route = ordered_stops[route]
            nodes_gps += nodes_gps_route

        # Luego habria que solucionar el problema de asignación entre las rutas VRP y los gps reales!!

    def create_improved_model_pyomo(self, ordered_stops, distance_dict, route):

        ## Create a concrete model for the TSP
        model = pyo.ConcreteModel(name='TSP')

        ## Parameters of the model
        nodes_gps = ordered_stops[route]
        distance_matrix = distance_dict[route]

        ## Define sets for the optimization model
        model.Nodes = range(1, len(nodes_gps))
        model.Arcs = [(i, j) for i in model.Nodes for j in model.Nodes if i != j]

        ## Define model variables
        model.x = pyo.Var(model.Arcs, within=pyo.Binary)
        model.u = pyo.Var(model.Nodes, within=pyo.NonNegativeReals, bounds=(1, len(model.Nodes)))

        ## Define the objective function
        def cost(model):
            return sum(distance_matrix[i][j] * model.x[i, j] for (i, j) in model.Arcs)
        model.Obj = pyo.Objective(rule=cost, sense=pyo.minimize)

        ## Define constraints
        def unica_entrada(model, j):
            return sum(model.x[i, j] for i in model.Nodes if i != j) == 1
        model.UniqueEntry = pyo.Constraint(model.Nodes, rule=unica_entrada)

        def unica_salida(model, i):
            return sum(model.x[i, j] for j in model.Nodes if i != j) == 1
        model.UniqueExit = pyo.Constraint(model.Nodes, rule=unica_salida)

        def subtours(model, i, j):
            if i != 1 and j != 1:
                return model.u[i] - model.u[j] + 1 <= len(model.Nodes) * (1 - model.x[i, j])
            else:
                return pyo.Constraint.Skip
        model.SubTours = pyo.Constraint(model.Arcs, rule=subtours)

        def depot_number(model, i):
            if i == 1:
                return model.u[i] == 1
            else:
                return pyo.Constraint.Skip
        model.Depot = pyo.Constraint(model.Nodes, rule=depot_number)

        return model

    def compute_cost_matrix(self, distance_dict, pred_matrix_route, dict_zones_gps, ordered_stops):

        distances_preference = {}
        valid_routes = []
        for route in ordered_stops:

            if route not in distance_dict:
                print(f"Route {route} skipped (no distance matrix)")
                continue

            if route not in pred_matrix_route:
                print(f"Route {route} skipped (no prediction matrix)")
                continue

            if route not in dict_zones_gps:
                print(f"Route {route} skipped (no zone assignment)")
                continue

            distance_matrix = distance_dict[route].to_numpy().copy()
            pred_matrix = pred_matrix_route[route].copy()

            updated_cost_matrix = distance_matrix.copy()

            # Zona de cada cliente (sin el depot)
            client_zones = np.array(dict_zones_gps[route][:-1])
            for i, zone_1 in enumerate(client_zones):
                for j, zone_2 in enumerate(client_zones):

                    # No modificar diagonal ni transiciones dentro del mismo cluster
                    if zone_1 == zone_2:
                        continue

                    prob_ij = max(pred_matrix[zone_1][zone_2], 1e-8)
                    cost_delta = math.log(prob_ij)

                    updated_cost_matrix[i,j] = distance_matrix[i,j] - cost_delta

            distances_preference[route] = updated_cost_matrix
            valid_routes.append(route)

        return distances_preference, valid_routes

    def get_clusters_gps(self, ordered_stops, depot_instance):

        stops_gps_df = self.adh_class.dict_to_df(ordered_stops, "route")
        df_communes, df_zones = self.adh_class.get_df_with_cluster(stops_gps_df)

        zone_depot = int(
            pd.read_csv(
                f"clustering/BuscaLibre/K={self.num_zones}/depot_clusters.csv"
            ).loc[0, "cluster"]
        )

        1/0

        dict_zones_gps = {}
        dict_zones_gps_no_dup = {}
        ordered_stops_with_depot = {}
        for route, list_nodes in ordered_stops.items():

            if depot_instance in list_nodes: list_nodes.remove(depot_instance)
            list_nodes.insert(0, depot_instance)
            ordered_stops_with_depot[route] = list_nodes

            df_route = stops_gps_df[stops_gps_df["route"] == route]
            list_zones = df_route['cluster'].values
            if self.mode == "cluster":
                df_depot_cluster = pd.read_csv(f'clustering/K={self.num_zones}/depot_clusters.csv')
                depot_cluster = int(df_depot_cluster['cluster'][0])
            else:
                depot_cluster = 11

            if len(list_zones) <= 1:
                del ordered_stops_with_depot[route]
                # print(f'Route {route} deleted: list of zones to short {list_zones}')
                continue
            else:
                if list_zones[0] != depot_cluster:
                    list_zones = np.concatenate([[depot_cluster], list_zones])

                dict_zones_gps[route] = list_zones
                list_zones_no_dup = [int(list_zones[0])]
                for zone in list_zones[1:]:
                    if zone != list_zones_no_dup[-1]:
                        list_zones_no_dup.append(int(zone))
                dict_zones_gps_no_dup[route] = list_zones_no_dup

        return dict_zones_gps, dict_zones_gps_no_dup, ordered_stops_with_depot

    def get_trained_model(self, strategy):

        # inputs to create model
        output_dim = 276
        if strategy == "edges":
            input_dim = 102
            hidden_dim = 64
            dropout = 0.2
        else:
            input_dim = 50
            hidden_dim = 64
            dropout = 0.3

        model_class_zones = self.pref_class.get_model(strategy, self.num_zones, input_dim, hidden_dim, output_dim, dropout)
        state_dict_zones = torch.load(f'dataset_outputs/BuscaLibre/tsp/K=50/{strategy}/best_model_{strategy}.pt')
        model_class_zones.load_state_dict(state_dict_zones)
        model_class_zones.eval()

        return model_class_zones

    def get_prediction_matrix(self, ordered_stops, strategy):

        dict_communes, dict_zones = self.adh_class.get_zones(ordered_stops)

        clients_per_zone = self.pref_class.get_clients_per_zone(dict_zones, self.num_zones)

        with open(f'dataset_outputs/BuscaLibre/tsp/K=50/{strategy}/Transformations/clients_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        if strategy == "edges":
            with open(f'dataset_outputs/BuscaLibre/tsp/K=50/{strategy}/Transformations/one_hot_encoder.pkl', 'rb') as f:
                encoder = pickle.load(f)

        trained_model = self.get_trained_model(strategy)

        with open(f"clustering/BuscaLibre/K={self.num_zones}/adjacency.json", "r") as f:
            adjacency_zones = json.load(f)

        # apply NN to predict transition probability between zones
        pred_matrix_route = {}
        for route in ordered_stops.keys():

            prediction_matrix = np.zeros((50, 50))

            if strategy == "all_clusters":

                # input: clients per zone
                clients = clients_per_zone[route]
                clients_scaled = scaler.transform([clients])

                X_tensor = torch.tensor(clients_scaled, dtype=torch.float32)

                with torch.no_grad():
                    pred_matrix = trained_model(X_tensor)
                prediction_matrix = pred_matrix.numpy().reshape(50, 50)

            elif strategy == "neighbors":

                # input: clients per zone
                clients = clients_per_zone[route]
                clients_scaled = scaler.transform([clients])

                X_tensor = torch.tensor(clients_scaled, dtype=torch.float32)

                with torch.no_grad():
                    prediction_vector = trained_model(X_tensor).squeeze(0)

                c = 0
                for i in adjacency_zones.keys():
                    origin = int(i)
                    for destination in adjacency_zones[i]:
                        if origin==destination:
                            continue
                        prediction_matrix[origin][destination] = prediction_vector[c].item()
                        c+=1

                        # TODO: comunas no vecinas quedan con probabilidad cero

            else:

                list_zones = sorted(set(dict_zones[route]))
                for i, zone_1 in enumerate(list_zones):
                    for j, zone_2 in enumerate(list_zones):
                        if zone_1 == zone_2:
                            continue

                        # aplicar el one-hot encoding a zone_1 y zone_2
                        origin = np.array([[zone_1]])
                        destination = np.array([[zone_2]])
                        origin_ohe = encoder.transform(origin)
                        destination_ohe = encoder.transform(destination)

                        # number of clientes
                        zone_count = clients_per_zone[route]
                        c_i = zone_count[zone_1]
                        c_j = zone_count[zone_2]

                        # aplicar el scaler al c_i y c_j
                        clients = np.array([[c_i, c_j]])
                        clients_scaled = scaler.transform(clients)

                        X_final = np.hstack([
                            origin_ohe,
                            destination_ohe,
                            clients_scaled
                        ])
                        X_tensor = torch.tensor(
                            X_final,
                            dtype=torch.float32
                        )

                        with torch.no_grad():
                            logits = trained_model(X_tensor)
                            prob_ij = torch.sigmoid(logits).item()

                        prediction_matrix[zone_1][zone_2] = prob_ij

            pred_matrix_route[route] = prediction_matrix

        return pred_matrix_route, dict_zones

    def get_improved_routes_tsp(self, instance, ordered_stops, depot_instance, strategy):

        # Get travel times, predicted matrix and updated cost matrix
        distance_dict = self.adh_class.compute_travel_times_api(instance, ordered_stops, depot_instance)

        pred_matrix_route, dict_zones = self.get_prediction_matrix(ordered_stops, strategy)

        cost_matrix, valid_routes = self.compute_cost_matrix(distance_dict, pred_matrix_route, dict_zones, ordered_stops)

        ordered_stops = {r: ordered_stops[r] for r in valid_routes}

        # We create one model for each route
        model_dict = {}
        ordered_simpli_routes = {}
        ordered_simpli_cords = {}

        for route, list_nodes in ordered_stops.items():

            if len(list_nodes) <= 1:
                # print(f'Route {route} deleted: list of nodes {list_nodes}')
                continue
            elif route not in cost_matrix:
                print(f'Route {route} deleted: not in cost matrix wtfff')
                continue
            route_ids = []

            if True:    # lib == 'gurobi'

                # Create model
                model = self.create_improved_model_pyomo(ordered_stops, cost_matrix, route)

                # Solve model
                solver = pyo.SolverFactory("gurobi", tee=True)
                solver.options['TimeLimit'] = 20
                logging.getLogger('pyomo').setLevel(logging.ERROR)
                solver.solve(model)
                order_dict = model.x.get_values()

                # Get routes from opt problem
                arcs = [arc for arc, val in order_dict.items() if val > 0.5]
                successor = {i: j for (i, j) in arcs}
                route_ids.append(1)
                current = 1
                while True:
                    next_node = successor[current]
                    route_ids.append(next_node)
                    if next_node == 1:
                        break
                    current = next_node

            elif False:     # TODO lib == 'pyvrp'

                # Create model
                model = self.create_model_pyvrp(ordered_stops, distance_dict, route, depot_instance)

                # Solve model
                res = model.solve(stop=MaxRuntime(30), display=False)
                best_solution = res.best
                best_route = list(best_solution.routes()[0])
                route_ids = [0] + best_route + [0]
                route_ids = [id+1 for id in route_ids]

            model_dict[route] = model
            ordered_simpli_routes[route] = route_ids
            ordered_simpli_cords[route] = [list_nodes[id-1] for id in route_ids]

            total_distance = 0
            for i, j in zip(route_ids[:-1], route_ids[1:]):
                total_distance += distance_dict[route].loc[i, j]
            # print(route)
            # print(total_distance)

            # Plot solution
            # self.plot_and_save(instance, route, ordered_stops, ordered_simpli_cords, depot_instance)

        return ordered_simpli_cords

    def plot_route_ax(self, ax, route_coords, title):

        # Depot = primer nodo
        depot = route_coords[0]

        rows = [
            {
                "lat": depot[0],
                "lon": depot[1],
                "type": "depot"
            }
        ]

        for coord in route_coords[1:]:
            rows.append({
                "lat": coord[0],
                "lon": coord[1],
                "type": "customer"
            })

        df = pd.DataFrame(rows)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df.lon,
                df.lat
            ),
            crs="EPSG:4326"
        )

        lon_min = df.lon.min() - 0.01
        lon_max = df.lon.max() + 0.01
        lat_min = df.lat.min() - 0.01
        lat_max = df.lat.max() + 0.01

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Flechas

        full_route = route_coords + [route_coords[0]]

        for i in range(len(full_route) - 1):
            start = full_route[i]
            end = full_route[i + 1]

            arrow = FancyArrowPatch(
                (start[1], start[0]),
                (end[1], end[0]),
                arrowstyle='->',
                mutation_scale=10,
                linewidth=2,
                alpha=0.8,
                color='C0',
                zorder=2
            )

            ax.add_patch(arrow)

        # Clientes

        gdf[gdf["type"] == "customer"].plot(
            ax=ax,
            color='#1f77b4',
            edgecolor='black',
            markersize=20,
            zorder=3
        )

        # Depot

        gdf[gdf["type"] == "depot"].plot(
            ax=ax,
            color='yellow',
            marker='*',
            edgecolor='black',
            markersize=120,
            zorder=4
        )

        cx.add_basemap(
            ax,
            crs=gdf.crs.to_string(),
            source=cx.providers.OpenStreetMap.Mapnik,
            alpha=0.8
        )

        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(0.05)
        )

        ax.yaxis.set_major_locator(
            ticker.MultipleLocator(0.05)
        )

        ax.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        ax.set_title(title)

    def plot_improved_route(
            self,
            instance,
            ordered_stops,
            ordered_tsp,
            ordered_simpli_edge,
            dict_zones_gps,
            dict_zones_tsp,
            dict_zones_edge
    ):

        for route in ordered_stops.keys():

            if route != '8fb34aca-262c-4228-87f8-1539634c8357':
                continue

            depot = ordered_tsp[route][0]

            ordered_stops[route].insert(0, depot)
            ordered_stops[route].append(depot)

            ordered_simpli_edge[route].insert(0, depot)
            ordered_simpli_edge[route].append(depot)

            print(ordered_stops[route])
            print(ordered_tsp[route])
            print(ordered_simpli_edge[route])
            print()
            print(dict_zones_gps[route])
            print(dict_zones_tsp[route])
            print(dict_zones_edge[route])
            print()
            print(len(ordered_stops[route]))
            print(len(ordered_tsp[route]))
            print(len(ordered_simpli_edge[route]))
            print()
            print(len(dict_zones_gps[route]))
            print(len(dict_zones_tsp[route]))
            print(len(dict_zones_edge[route]))

            gps_route = ordered_stops[route]
            tsp_route = ordered_tsp[route]
            nn_route = ordered_simpli_edge[route]

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 7)
            )

            self.plot_route_ax(
                axes[0],
                gps_route,
                "GPS"
            )

            self.plot_route_ax(
                axes[1],
                tsp_route,
                "PyVRP"
            )

            self.plot_route_ax(
                axes[2],
                nn_route,
                "NN + TSP"
            )

            plt.suptitle(
                f"Instance {instance} - Route {route}",
                fontsize=16
            )

            save_dir = (
                f"resultados_adh/improved_mapas/"
                f"instance_{instance}"
            )

            os.makedirs(
                save_dir,
                exist_ok=True
            )

            plt.savefig(
                f"{save_dir}/{route}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            # solo una ruta
            break

    # def plot_route_improvement(self, instance, route):
    #
    #     # Routes GPS
    #     _, _, ordered_stops, _, depot_instance = self.adh_class.get_stops_vehicles(instance)
    #     gps_df = self.adh_class.dict_to_df(ordered_stops, 'route_gps')
    #     gps_df = self.adh_class.prepare_route_df(gps_df, 'route_gps', route, depot_instance)
    #     gps_df["order_gps"] = gps_df.index
    #
    #     # Routes SimpliRoute
    #     _, ordered_simpli_routes, ordered_simpli_cords = self.adh_class.get_routes_tsp(instance, lib='gurobi')
    #     simpli_df = self.adh_class.dict_to_df(ordered_simpli_cords, 'route_simpli')
    #     simpli_df = self.adh_class.prepare_route_df(simpli_df, 'route_simpli', route, depot_instance)
    #     # Routes improved
    #     _, _, improved_ordered_simpli_cords = self.get_improved_routes_tsp(instance, ordered_stops, depot_instance)
    #     improved_simpli_df = self.adh_class.dict_to_df(improved_ordered_simpli_cords, 'route_simpli_improved')
    #     improved_simpli_df = self.adh_class.prepare_route_df(improved_simpli_df, 'route_simpli_improved', route, depot_instance)
    #
    #     # clusters
    #     unique_clusters = gps_df["cluster"].unique()
    #     cluster_map = {c: i for i, c in enumerate(unique_clusters)}
    #
    #     gps_df["cluster_small"] = gps_df["cluster"].map(cluster_map)
    #     simpli_df["cluster_small"] = simpli_df["cluster"].map(cluster_map)
    #     improved_simpli_df["cluster_small"] = improved_simpli_df["cluster"].map(cluster_map)
    #
    #     cmap = "tab10"
    #     colors = cm.get_cmap(cmap, len(unique_clusters))
    #     cluster_colors = {c: colors(i) for c, i in cluster_map.items()}
    #
    #     # figura
    #     fig, ax = plt.subplots(1, 3, figsize=(14, 7))
    #
    #     self.adh_class.plot_route(ax[0], gps_df, cluster_colors, gps_df["order_gps"], cmap)
    #     ax[0].set_title("GPS")
    #     self.adh_class.plot_route(ax[1], simpli_df, cluster_colors, gps_df["order_gps"], cmap)
    #     ax[1].set_title("SimpliRoute")
    #     self.adh_class.plot_route(ax[2], improved_simpli_df, cluster_colors, gps_df["order_gps"], cmap)
    #     ax[2].set_title("MLP + TSP")
    #
    #     fig.suptitle(f"Solution {instance} - {route}", fontsize=16)
    #
    #     # guardar
    #     save_path = f"resultados_adh/improved_mapas/instancia_{instance}/{route}"
    #     os.makedirs(save_path, exist_ok=True)
    #
    #     file_path = f"{save_path}/map_base_{instance}_{route}.png"
    #
    #     plt.tight_layout()
    #     plt.savefig(file_path, dpi=300)
    #     plt.close()




def create_boxplots_adherence():

    # =====================================================
    # LOAD RESULTS
    # =====================================================

    df = pd.read_csv(
        "results_improved_adherence_complete.csv"
    )

    methods_map = {
        "adh_stops_tsp": "PyVRP TSP",
        "adh_stops_all": "All Clusters",
        "adh_stops_neighbors": "Neighbors",
        "adh_stops_edge": "Edges"
    }

    # =====================================================
    # LOAD TRAIN / TEST SPLITS
    # =====================================================

    split_dict = {}

    for split in ["train", "test"]:

        path_split = (
            "dataset_outputs/BuscaLibre/"
            "tsp/K=50/edges/"
            f"dataset_edges_{split}.csv"
        )

        df_split = pd.read_csv(path_split)

        split_dict[split] = set(
            zip(
                df_split["instance"],
                df_split["route"]
            )
        )

    def get_split(row):

        key = (row["instance"], row["route"])

        if key in split_dict["test"]:
            return "Test"

        if key in split_dict["train"]:
            return "Train"

        return None

    df["split"] = df.apply(
        get_split,
        axis=1
    )

    df = df[df["split"].notna()]

    # =====================================================
    # BUILD LONG DATAFRAME
    # =====================================================

    df_plot = df.melt(
        id_vars=[
            "instance",
            "route",
            "split"
        ],
        value_vars=[
            "adh_stops_tsp",
            "adh_stops_all",
            "adh_stops_neighbors",
            "adh_stops_edge"
        ],
        var_name="method",
        value_name="value"
    )

    df_plot["method"] = (
        df_plot["method"]
        .map(methods_map)
    )

    # =====================================================
    # PLOT
    # =====================================================

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=True
    )

    methods_order = [
        "PyVRP TSP",
        "All Clusters",
        "Neighbors",
        "Edges"
    ]

    def format_pvalue(p):

        if p < 0.001:
            return "p < 0.001"

        return f"p = {p:.3f}"

    palette = {
        "PyVRP TSP": "#808080",
        "All Clusters": "#4C72B0",
        "Neighbors": "#55A868",
        "Edges": "#C44E52"
    }

    global_y_max = df_plot["value"].max()
    global_y_min = df_plot["value"].min()
    global_y_range = global_y_max - global_y_min

    for ax, split_name in zip(
        axes,
        ["Train", "Test"]
    ):

        df_split_plot = df_plot[
            df_plot["split"] == split_name
        ]

        sns.boxplot(
            data=df_split_plot,
            x="method",
            y="value",
            order=methods_order,
            palette=palette,
            showfliers=False,
            ax=ax
        )

        # Mean lines

        for method_idx, method in enumerate(methods_order):

            subset = df_split_plot[
                df_split_plot.method == method
            ]

            mean_val = subset.value.mean()

            ax.hlines(
                mean_val,
                method_idx - 0.3,
                method_idx + 0.3,
                colors="black",
                linestyles="dashed",
                linewidth=2
            )

        # Wilcoxon against Base

        df_pair = (
            df_split_plot
            .pivot_table(
                index=[
                    "instance",
                    "route"
                ],
                columns="method",
                values="value"
            )
            .dropna()
        )

        comparisons = [
            ("All Clusters", 1),
            ("Neighbors", 2),
            ("Edges", 3)
        ]

        for method, xpos in comparisons:

            stat, p_value = wilcoxon(
                df_pair["PyVRP TSP"],
                df_pair[method]
            )

            ax.text(
                xpos,
                global_y_max + 0.05 * global_y_range,
                format_pvalue(p_value),
                ha="center",
                fontsize=9,
                fontweight="bold"
            )

        ax.set_title(
            split_name,
            fontsize=14,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("Adherencia")
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.5
        )

        ax.set_ylim(
            global_y_min,
            global_y_max + 0.2 * global_y_range
        )

    fig.suptitle(
        "Comparación de Adherencia de Paradas",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.savefig(
        "results_improved_adherence_boxplot.png",
        dpi=300
    )

    # =====================================================
    # SUMMARY TABLE
    # =====================================================

    summary_rows = []

    for split in ["Train", "Test"]:

        df_split = df[df["split"] == split]

        summary_rows.append({
            "Split": split,
            "PyVRP TSP": df_split["adh_stops_tsp"].mean(),
            "All Clusters": df_split["adh_stops_all"].mean(),
            "Neighbors": df_split["adh_stops_neighbors"].mean(),
            "Edges": df_split["adh_stops_edge"].mean()
        })

    df_summary = pd.DataFrame(summary_rows)

    numeric_cols = [
        "PyVRP TSP",
        "All Clusters",
        "Neighbors",
        "Edges"
    ]

    df_summary[numeric_cols] = (
        df_summary[numeric_cols]
        .round(4)
    )

    print("\n")
    print("=" * 80)
    print("PROMEDIOS DE ADHERENCIA DE PARADAS")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    plt.show()