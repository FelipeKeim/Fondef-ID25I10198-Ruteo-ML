import os
import math
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter

from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

import json
import requests
from pathlib import Path

import pyomo.environ as pyo
from pyvrp import Model
from pyvrp.stop import MaxRuntime

import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.data_processing import DataLoad

class Adherence:
    def __init__(self, n_envio, num_zones=50):

        # Load Data
        data_loader = DataLoad(n_envio=n_envio)
        data_instances, data_gps, data_wores, data_woreq, instances_ids, map, plans_ids, comunas = data_loader.return_data()

        self.data_instances = data_instances
        self.data_gps = data_gps
        self.data_wores = data_wores
        self.data_woreq = data_woreq
        self.map = map
        self.comunas = comunas

        self.instances_ids = instances_ids
        self.plans_ids = plans_ids

        # Clustering
        self.num_zones = num_zones
        clustering_path = f'clustering/BuscaLibre/K={num_zones}'
        if not os.path.exists(clustering_path):
            os.makedirs(clustering_path)
        file_clustering = f'clustering/BuscaLibre/K={num_zones}/kmeans_model_{num_zones}.pkl'
        if not os.path.exists(file_clustering):
            self.make_clustering(num_zones)

    def make_clustering(self, n_clusters, elbow_method=False):

        list_all_instances = []
        depot_list = []
        for instance in self.instances_ids:
            _, simpli_coordinates, depot_instance = self.get_stops_simpli(instance)
            simpli_df = self.dict_to_df(simpli_coordinates, 'vehicle')
            list_all_instances.append(simpli_df)
            depot_list.append({
                "instance": instance,
                "latitude": depot_instance[0],
                "longitude": depot_instance[1]
            })
        df_final = pd.concat(list_all_instances)
        df_final.drop('vehicle', axis=1, inplace=True)
        df_depots = pd.DataFrame(depot_list)

        # Get cluster data
        cluster_data = df_final[['latitude', 'longitude']].values

        # Use the Elbow Method to determine the optimal K
        if elbow_method:
            wcss = []
            for k in range(1, 60):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(cluster_data)
                wcss.append(kmeans.inertia_)

            # Plot the Elbow Curve
            plt.plot(range(1, 15), wcss, marker='o')
            plt.title('Elbow Method')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('WCSS')
            plt.savefig('clustering/Elbow Method.png')
            plt.show()

        # Train the K-Means model with optimal K
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_final['Cluster'] = kmeans.fit_predict(cluster_data)
        centroids = kmeans.cluster_centers_

        # Get adjacent clusters
        vor = Voronoi(centroids)
        adjacency = {i: [i] for i in range(len(centroids))}

        for i, j in vor.ridge_points:
            adjacency[int(i)].append(int(j))
            adjacency[int(j)].append(int(i))

        for k in adjacency:
            adjacency[k] = sorted(list(set(adjacency[k])))

        depot_coords = df_depots[['latitude', 'longitude']].values
        df_depots["cluster"] = kmeans.predict(depot_coords)

        with open(f"clustering/BuscaLibre/K={n_clusters}/kmeans_model_{n_clusters}.pkl", "wb") as f:
            pickle.dump(kmeans, f)
        with open(f"clustering/BuscaLibre/K={n_clusters}/kmeans_centroids_{n_clusters}.pkl", "wb") as f:
            pickle.dump(centroids, f)
        df_depots.to_csv(f"clustering/BuscaLibre/K={n_clusters}/depot_clusters.csv", index=False)

        with open(f"clustering/BuscaLibre/K={n_clusters}/adjacency.json", "w") as f:
            json.dump(adjacency, f, indent=4)

        # Visualize the clusters
        plt.scatter(df_final['longitude'], df_final['latitude'], c=df_final['Cluster'], cmap='viridis', s=2)
        plt.title('Customer Segments')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f'clustering/BuscaLibre/K={n_clusters}/Clustering K={n_clusters}.png')
        plt.show()

    def compute_distance(self, lat1, lon1, lat2, lon2):

        radio_tierra = 6371.0

        # Convertir grados a radianes
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Diferencia de latitud y longitud
        d_lat = lat2_rad - lat1_rad
        d_lon = lon2_rad - lon1_rad

        # Calcular la distancia utilizando la fórmula haversine
        a = math.sin(d_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Distancia en kilómetros
        distancia = radio_tierra * c

        return distancia

    def get_gps_registers(self, plan_id):

        # Filtramos registros por plan_id (instancia)
        df_gps = self.data_gps.copy()
        df_gps = df_gps[df_gps['plan_id'] == plan_id]

        # Agrupamos por rutas
        routes = df_gps.route_id.unique()
        coordinates_by_routes = {}
        for route in routes:
            df_gps_route = df_gps[df_gps['route_id'] == route]

            latitudes = df_gps_route[df_gps_route['plan_id'] == plan_id].latitude.values
            longitudes = df_gps_route[df_gps_route['plan_id'] == plan_id].longitude.values
            time = df_gps_route[df_gps_route['plan_id'] == plan_id]['created_at'].values
            coordinates_by_routes[route] = (latitudes, longitudes, time)

        return coordinates_by_routes

    def get_stops_simpli(self, instance):

        wores_inst = self.data_wores[instance]

        simpli_routes = {}
        simpli_coordinates = {}
        stops_list = []
        depot_instance = None
        for route in wores_inst['vehicles']:

            vehicle_id = route['ident']
            simpli_routes[vehicle_id] = []
            simpli_coordinates[vehicle_id] = []

            stops = route['tours'][0]['nodes']
            depot = (stops[0]['lat'], stops[0]['lon'])
            for n_stop, stop in enumerate(stops):

                latitude = stop['lat']
                longitude = stop['lon']
                coord = (latitude, longitude)

                # Revisamos que depot sea el mismo
                if depot_instance is None:
                    depot_instance = depot
                else:
                    if depot != depot_instance:
                        print('Alerta. Depot distintos para distintas rutas.')

                # Revisamos que comience y termine en el depot
                if n_stop == len(stops)-1:
                    if coord != depot:
                        print(f'Alerta. Instancia {instance} no comienza y/o termina en depot.')
                        pass

                if coord != depot:
                    # Revisar si la parada está repetida en otra ruta. En tal caso, instancia mala
                    if coord in stops_list:
                        # print(f'Alerta. Instancia {instance} con parada {coord} repetida en dos rutas distintas.')
                        pass

                    # Revisar si la parada se repite en la misma ruta
                    if coord in simpli_coordinates[vehicle_id]:
                        if coord != simpli_coordinates[vehicle_id][-1]:
                            print(f'Alerta. Parada {coord} repetida.')
                        continue

                # Agregamos las paradas a las listas
                simpli_coordinates[vehicle_id].append(coord)
                stop_id = stop['ident']
                simpli_routes[vehicle_id].append(stop_id)
                stops_list.append(coord)

        return simpli_routes, simpli_coordinates, depot_instance

    def get_registros_base(self, coordinates_by_routes, simpli_coordinates):

        # Diccionario de rutas para cada parada y de registros cercanos a cada parada
        route_by_stop = {}
        registros_base = {}
        visited_nodes = []
        non_visited_nodes = []
        for vehicle, stops_vehicle in simpli_coordinates.items():
            for stop in stops_vehicle:
                route_by_stop[stop] = None
                registros_base[stop] = []

        # Asignamos cada parada a una ruta
        for stop in route_by_stop.keys():
            min_distance = 150
            for route in coordinates_by_routes.keys():
                lat_route, lon_route, _ = coordinates_by_routes[route]
                for lat_real, lon_real in zip(lat_route, lon_route):
                    distance = self.compute_distance(lat_real, lon_real, stop[0], stop[1]) * 1000
                    if distance < min_distance:
                        min_distance = min(min_distance, distance)
                        route_by_stop[stop] = route

            # Obtenemos lista con todas los registros gps cercanos a cada parada
            route = route_by_stop[stop]
            if route is None:
                non_visited_nodes.append(stop)
            else:
                visited_nodes.append(stop)
                lat_route, lon_route, time_route = coordinates_by_routes[route]
                for lat_real, lon_real, time_real in zip(lat_route, lon_route, time_route):
                    distance = self.compute_distance(lat_real, lon_real, stop[0], stop[1]) * 1000
                    if distance < 150:
                        registros_base[stop].append((lat_real, lon_real, time_real))

        return route_by_stop, registros_base, visited_nodes, non_visited_nodes

    def speed_criterion(self, stop, route, registros_base, coordinates_by_routes, speed_limit=10):

        # Real gps latitudes, longitudes and times
        lat_real, lon_real, time_real = coordinates_by_routes[route]

        # We compute the speed between the points in registros_base[stop], but adding next point
        for gps_stop in registros_base[stop]:
            # We search point in real gps latitudes and longitudes
            for n_register, (lat_gps, lon_gps, time_gps) in enumerate(zip(lat_real, lon_real, time_real)):
                if (lat_gps, lon_gps, time_gps) == gps_stop:
                    # Si es el primer o último registro, se asume que si es una parada
                    if n_register == 0 or n_register == len(lat_real)-1:
                        return True
                    else:
                        # Time and distance difference between with the previous and next register
                        delta_time_1 = (time_gps - time_real[n_register-1]) / np.timedelta64(1, 's')
                        delta_time_2 = (time_real[n_register + 1] - time_gps) / np.timedelta64(1, 's')
                        # delta_distance_1 = self.compute_distance(lat_gps, lon_gps, lat_real[n_register-1], lon_real[n_register-1])
                        # delta_distance_2 = self.compute_distance(lat_gps, lon_gps, lat_real[n_register+1], lon_real[n_register+1])
                        # # If speed is below 10 km/h, then it is a stop
                        # speed = min(delta_distance_1/delta_time_1, delta_distance_2/delta_time_2)
                        # if speed < speed_limit:
                        #     return True
                        if delta_time_1 > 45 or delta_time_2 > 45:
                            return True
        return False

    def evaluate_criteria(self, coordinates_by_routes, route_by_stop, registros_base, visited_nodes, non_visited_nodes):

        # Diccionario de paradas para cada ruta
        stops_by_route = {}
        ordered_stops = {}
        reason_stop = {}
        time_stop = {}
        routes_without_stops = []
        for route in coordinates_by_routes.keys():
            stops_by_route[route] = []
            ordered_stops[route] = []
            time_stop[route] = []

        # Utilizando los registros GPS, evaluamos los criterios para determinar si el vehículo pasó por la parada
        for stop, route in route_by_stop.items():

            if route is None:
                reason_stop[stop] = 'distance'
                continue

            # Velocidad menor a 10 km/h
            if not self.speed_criterion(stop, route, registros_base, coordinates_by_routes):
                reason_stop[stop] = 'speed'
                visited_nodes.remove(stop)
                non_visited_nodes.append(stop)
                continue

            # If the criteria are satisfied, then we get the time and save the stop
            times = np.array([time for _, _, time in registros_base[stop]])
            avg_time = times.view('int64').mean().astype('datetime64[ns]')
            time_stop[route].append(avg_time)
            stops_by_route[route].append(stop)

        # Ordenamos paradas de cada ruta
        for route in coordinates_by_routes.keys():
            if len(stops_by_route[route]) == 0:
                routes_without_stops.append(route)
                ordered_stops[route] = []
            else:
                list_stops = stops_by_route[route]
                list_times = time_stop[route]
                ordered_stops[route] = [
                    (lat, lon)
                    for _, (lat, lon) in sorted(zip(list_times, list_stops), key=lambda x: x[0])
                ]

        return ordered_stops, reason_stop, routes_without_stops

    def get_stops_vehicles(self, instance):

        # Obtenemos planificación de Simpli
        plan_id = self.plans_ids[instance]

        # Extraemos nodos (paradas) de Simpli
        simpli_routes, simpli_coordinates, depot_instance = self.get_stops_simpli(instance)

        # Obtenemos coordenadas GPS de las rutas
        coordinates_by_routes = self.get_gps_registers(plan_id)

        # Computamos el registro GPS, y la ruta más cercana a cada parada
        route_by_stop, registros_base, visited_nodes, non_visited_nodes = self.get_registros_base(coordinates_by_routes, simpli_coordinates)

        # Finalmente, aplicamos los criterios para determinar las paradas (con su tiempo) que efectivamente son recorridas
        ordered_stops, reason_stop, routes_without_stops = self.evaluate_criteria(coordinates_by_routes, route_by_stop, registros_base, visited_nodes, non_visited_nodes)

        return simpli_routes, simpli_coordinates, ordered_stops, reason_stop, depot_instance

    def get_routes_id(self, instance, simpli_routes, ordered_stops):

        # Obtenemos rutas según planificación de Simpli
        stops_id_simpli = {}
        map_stops_id_simpli = {}
        n_stop = 0
        for vehicle in simpli_routes.keys():
            stops_id_simpli[vehicle] = []
            stop_list_str = simpli_routes[vehicle]
            stops_id_simpli[vehicle].append('d')
            for stop_str in stop_list_str:
                map_stops_id_simpli[stop_str] = n_stop
                stops_id_simpli[vehicle].append(n_stop)
                n_stop+=1
            stops_id_simpli[vehicle].append('d')

        # Obtenemos rutas reales según GPS
        stops_id_gps = {}
        for route, stops_coords in ordered_stops.items():
            stops_id_gps[route] = []
            stops_id_gps[route].append('d')
            for stop in stops_coords:
                stop_lat, stop_lon = stop
                # Revisamos nodos para buscar el id de la parada
                for node in self.data_woreq[instance]['nodes']:
                    if stop_lat == node['lat'] and stop_lon == node['lon']:
                        if node['ident'] in map_stops_id_simpli:
                            node_id = map_stops_id_simpli[node['ident']]
                            stops_id_gps[route].append(node_id)
                        else:
                            node_name = node['ident']
                            # print(f'Nodo {node_name} no agregado. Revisar')
            stops_id_gps[route].append('d')

        return stops_id_simpli, stops_id_gps

    def dict_to_df(self, dic_cords, key_name):

        df_list = []

        # SimpliRoute or Stops dict
        for key, coord_list in dic_cords.items():

            rows = []
            for lat, lon in coord_list:
                rows.append({
                    f"{key_name}": key,
                    "latitude": lat,
                    "longitude": lon
                })
            df = pd.DataFrame(rows)
            df_list.append(df)

        # Concat dataframes
        df_final = pd.concat(df_list)
        return df_final

    def get_vehicle_id(self, instance, route_list):

        # Load df map vehicle-route
        df_map = self.map
        plan_id = self.plans_ids[instance]
        df_map = df_map[df_map['route_plan'] == plan_id]

        # Asociate each vehicle with a route
        vehicle_route = {}
        data_request = self.data_woreq[instance]['vehicles']
        for vehicle in data_request:
            id_vehicle = int(vehicle['ident'])
            df_vehicle = df_map[df_map['vehicle_id'] == id_vehicle]
            if len(df_vehicle) == 1:
                route = df_vehicle['route_id'].iloc[0]
                if route in route_list:
                    vehicle_route[id_vehicle] = route

        return vehicle_route

    def compute_travel_times_api(self, instance, ordered_stops, depot_instance):

        # factor tráfico
        fmv = self.data_woreq[instance]['fmv']

        url = "https://osrm-proxy-qa.simpliroute.com/table"
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'http://localhost:3000/',
            'Origin': 'http://localhost:3000',
            'User-Agent': 'Mozilla/5.0',
            'authorization': 'Token 729af6a4562a58036944c18abad921d7366c329d',
            'Content-Type': 'application/json'
        }

        travel_time_dict = {}

        for route, list_stops in ordered_stops.items():

            if len(list_stops) <= 1:
                continue

            list_stops = list(list_stops)

            # Agregar depot
            if depot_instance in list_stops:
                list_stops.remove(depot_instance)
            list_stops.insert(0, depot_instance)

            # Aplicamos API
            lista = [f"{lat},{lon}" for lat, lon in list_stops]
            payload = json.dumps({
                "loc": lista,
                "country": "CL",
                "speedRate": fmv
            })
            response = requests.post(url, headers=headers, data=payload)
            matrix_data = response.json()

            matrix = matrix_data['matrix']
            transformed_matrix = [
                [float(value) / 600 for value in row]
                for row in matrix
            ]

            n = len(list_stops)
            points = list(range(1, n + 1))

            df_matrix = pd.DataFrame(
                transformed_matrix,
                index=points,
                columns=points
            )

            travel_time_dict[route] = df_matrix

        return travel_time_dict

    def create_model_pyomo(self, ordered_stops, distance_dict, route):

        ## Create a concrete model for the TSP
        model = pyo.ConcreteModel(name='TSP')

        ## Parameters of the model
        nodes_gps = ordered_stops[route]
        distance_matrix = distance_dict[route]

        ## Define sets for the optimization model
        model.Nodes = range(1, len(nodes_gps) + 1)
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

    def create_model_pyvrp(self, ordered_stops, distance_dict, route, depot_instance):

        list_nodes = ordered_stops[route]

        m_pyvrp = Model()

        # Indicamos que es un único vehículo
        m_pyvrp.add_vehicle_type(
            1,
            capacity=100,
        )

        # Seteamos depot
        m_pyvrp.add_depot(
            x=depot_instance[1],
            y=depot_instance[0],
        )

        # Clients nodes
        clients = [
            m_pyvrp.add_client(
                x=list_nodes[idx][1],
                y=list_nodes[idx][0],

                # tw_early=TIME_WINDOWS[idx][0],
                # tw_late=TIME_WINDOWS[idx][1],
                delivery=1
            )
            for idx in range(1, len(list_nodes))
        ]

        # Arcos
        for frm_idx, frm in enumerate(m_pyvrp.locations):
            for to_idx, to in enumerate(m_pyvrp.locations):
                distance = distance_dict[route][frm_idx+1][to_idx+1]
                m_pyvrp.add_edge(frm, to, distance=distance)

        return m_pyvrp

    def get_routes_tsp(self, instance, lib='gurobi'):

        # We get the vehicle routes
        _, _, ordered_stops, _, depot_instance = self.get_stops_vehicles(instance)

        # Retrieve necessary datasets and parameters from input data
        distance_dict = self.compute_travel_times_api(instance, ordered_stops, depot_instance)

        # We create one model for each route
        model_dict = {}
        ordered_simpli_routes = {}
        ordered_simpli_cords = {}

        for route, list_nodes in ordered_stops.items():

            if len(list_nodes) <= 1: continue

            model = None
            route_ids = []

            if depot_instance in list_nodes: list_nodes.remove(depot_instance)
            list_nodes.insert(0, depot_instance)

            if lib == 'gurobi':

                # Create model
                model = self.create_model_pyomo(ordered_stops, distance_dict, route)

                # Solve model
                solver = pyo.SolverFactory("gurobi")
                solver.options['TimeLimit'] = 20   # 30
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

            elif lib == 'pyvrp':

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
            for i, j in zip(route_ids[:-2], route_ids[2:]):
                total_distance += distance_dict[route].loc[i, j]
            print(route)
            print(total_distance)

            # Plot solution
            # self.plot_and_save(instance, route, ordered_stops, ordered_simpli_cords, depot_instance)

        return model_dict, ordered_simpli_routes, ordered_simpli_cords

    def prepare_route_df(self, df, route_col, route, depot):
        df = df[df[route_col] == route]
        df = df[
            (df['latitude'] != depot[0]) &
            (df['longitude'] != depot[1])
            ].copy()
        df = self.get_df_with_cluster(df)
        return df.reset_index(drop=True)

    def map_clusters(self, gps_df, simpli_df):
        unique_clusters = gps_df["cluster"].unique()
        cluster_map = {c: i for i, c in enumerate(unique_clusters)}

        gps_df["cluster_small"] = gps_df["cluster"].map(cluster_map)
        simpli_df["cluster_small"] = simpli_df["cluster"].map(cluster_map)

        return cluster_map, unique_clusters

    def plot_route(self, ax, df, cluster_colors, order_labels, cmap):
        pts = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )

        ax.plot(df.longitude, df.latitude, color='black', linewidth=2, zorder=1)

        pts.plot(
            ax=ax,
            column="cluster_small",
            cmap=cmap,
            markersize=120,
            edgecolor='black',
            linewidth=0.8,
            zorder=3,
            legend=False
        )

        for x, y, c, label in zip(
                pts.geometry.x,
                pts.geometry.y,
                df["cluster"],
                order_labels
        ):
            ax.text(
                x, y, str(label),
                fontsize=8,
                ha='center',
                va='center',
                color='white',
                zorder=4,
                bbox=dict(
                    facecolor=cluster_colors[c],
                    edgecolor='black',
                    alpha=0.9,
                    boxstyle='round,pad=0.25'
                )
            )

        ctx.add_basemap(ax, crs="EPSG:4326",
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        attribution=False,
                        alpha=0.5)

        ax.set_axis_off()

    def get_df_with_comuna(self, df):

        gdf_puntos = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        )

        comunas_proj = self.comunas.to_crs("EPSG:4326")

        gdf_resultado = gpd.sjoin(
            gdf_puntos,
            comunas_proj[["name", "geometry"]],
            how="left",
            predicate="within"
        )

        # Eliminamos filas que se agregan con la provincia
        gdf_resultado = gdf_resultado[~gdf_resultado.name.str.contains("Provincia")]

        # Renombrar
        gdf_resultado = gdf_resultado.rename(columns={"name": "comuna"}).reset_index()

        # Agregar cluster number
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

        ids_comunas = []
        for i in range(len(gdf_resultado['comuna'])):
            for j in range(len(comunas_rm)):
                if gdf_resultado["comuna"][i] == comunas_rm[j]:
                    ids_comunas.append(j)
        gdf_resultado['cluster'] = ids_comunas

        return gdf_resultado

    def get_df_with_cluster(self, df):

        # df clustered by communes
        df_communes = self.get_df_with_comuna(df)

        # df clustered by zones
        df_zones = df.copy()
        with open(f"clustering/BuscaLibre/K={self.num_zones}/kmeans_model_{self.num_zones}.pkl", "rb") as f:
            kmeans = pickle.load(f)

            coords = df[["latitude", "longitude"]].copy()
            df_zones["cluster"] = kmeans.predict(coords)
            df_zones["zone"] = df_zones["cluster"]

        return df_communes, df_zones

    def plot_and_save(self, inst_num, route, ordered_stops, ordered_simpli_cords, depot_instance):

        # preparar data
        gps_df = self.dict_to_df(ordered_stops, 'route_gps')
        gps_df = self.prepare_route_df(gps_df, 'route_gps', route, depot_instance)
        gps_df["order_gps"] = gps_df.index

        simpli_df = self.dict_to_df(ordered_simpli_cords, 'route_simpli')
        simpli_df = self.prepare_route_df(simpli_df, 'route_simpli', route, depot_instance)

        # clusters
        cluster_map, unique_clusters = self.map_clusters(gps_df, simpli_df)

        cmap = "tab10"
        colors = cm.get_cmap(cmap, len(unique_clusters))
        cluster_colors = {c: colors(i) for c, i in cluster_map.items()}

        # figura
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        self.plot_route(ax[0], gps_df, cluster_colors, gps_df["order_gps"], cmap)
        ax[0].set_title("GPS")

        self.plot_route(ax[1], simpli_df, cluster_colors, gps_df["order_gps"], cmap)
        ax[1].set_title("Simpli")

        fig.suptitle(f"Solution {inst_num} - {route}", fontsize=16)

        # guardar
        save_path = f"resultados_adh/mapas/instancia_{inst_num}/{route}"
        os.makedirs(save_path, exist_ok=True)

        file_path = f"{save_path}/map_base_{inst_num}_{route}.png"

        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

    def get_zones(self, ordered_stops_gps):

        # We create dataframe to extract the zones
        gps_df = self.dict_to_df(ordered_stops_gps, "route")
        df_communes, df_zones = self.get_df_with_cluster(gps_df)

        zone_depot = int(
            pd.read_csv(
                f"clustering/BuscaLibre/K={self.num_zones}/depot_clusters.csv"
            ).loc[0, "cluster"]
        )
        
        # We extract dictionaries with the clusters id lists
        output = {}
        for method, df_clusters, depot_cluster in [
            ("communes", df_communes, 11),
            ("zones", df_zones, zone_depot),
        ]:

            dict_clusters = {}
            for route in ordered_stops_gps.keys():

                clusters = df_clusters[df_clusters["route"] == route]["cluster"].values

                if len(clusters) == 0:
                    dict_clusters[route] = [depot_cluster, depot_cluster]
                    continue

                if clusters[0] != depot_cluster:
                    clusters = np.concatenate([[depot_cluster], clusters])

                if clusters[-1] != depot_cluster:
                    clusters = np.concatenate([clusters, [depot_cluster]])

                dict_clusters[route] = clusters

            output[method] = dict_clusters

        return output["communes"], output["zones"]

    def compute_adherence_instance(self, instance, tsp=True):

        # Get stops according to Simpli and GPS
        simpli_routes, simpli_coordinates, ordered_stops, reason_stop, depot_instance = self.get_stops_vehicles(instance)

        # Get routes Simpli using TSP
        if tsp:
            model_dict, ordered_simpli_routes, ordered_simpli_cords = self.get_routes_tsp(instance, lib='gurobi')
        else:
            ordered_simpli_cords = simpli_coordinates

        # Get clusters ids desde las paradas
        dict_zones_simpli, dict_zones_gps = self.get_zones(ordered_simpli_cords, ordered_stops)

        # We compute adherence using both methods
        adherence_list = []
        for route in ordered_stops.keys():

            # We obtain vehicle id
            if not tsp:
                df_map = self.map
                plan_id = self.plans_ids[instance]
                df_map = df_map[df_map['route_plan'] == plan_id]
                df_map_route = df_map[df_map['route_id'] == route]
                if df_map_route.empty:
                    continue  # saltar este caso
                vehicle_id = str(df_map_route['vehicle_id'].iloc[0])
                if not vehicle_id in dict_zones_simpli:
                    continue
                ids_simpli = dict_zones_simpli[vehicle_id]
            else:
                ids_simpli = dict_zones_simpli[route]

            ## Adherence with clusters
            ids_gps = dict_zones_gps[route]

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

        route_list = list(ordered_stops.keys())

        return adherence_list, route_list

    def compute_adherence_all_instances(self, tsp=True):

        rows = []
        errors = []
        for instance in self.instances_ids:
            print(instance)

            try:
                adh_list_instance, adh_list_no_dup, route_list = self.compute_adherence_instance(instance, tsp=tsp)
                for route, adh, adh_no_dup in zip(route_list, adh_list_instance, adh_list_no_dup):
                    rows.append({
                        "instance": instance,
                        "route": route,
                        "adherence": adh,
                        "adherence_no_dup": adh_no_dup
                    })

            except Exception as e:
                print(f"Error en instance {instance}: {e}")
                errors.append({
                    "instance": instance,
                    "error": str(e)
                })
                continue

        adherencias_base = pd.DataFrame(rows)
        if tsp == True:
            adherencias_base.to_csv("resultados_adh/adherencias_base_tsp.csv", index=False)
            if errors:
                pd.DataFrame(errors).to_csv("resultados_adh/errors_tsp_improved.csv", index=False)
        else:
            adherencias_base.to_csv("resultados_adh/adherencias_base_vrp.csv", index=False)
            if errors:
                pd.DataFrame(errors).to_csv("resultados_adh/errors_vrp_improved.csv", index=False)

        return adherencias_base

    def create_tsp_request(self, instance):

        # Json VRP
        json_req = self.data_woreq[instance]

        # GPS stops
        simpli_routes, simpli_coordinates, ordered_stops, reason_stop, depot_instance = self.get_stops_vehicles(
            instance)
        route_list = list(ordered_stops.keys())

        # Routes to create TSP
        vehicle_route = self.get_vehicle_id(instance, route_list)

        json_per_route = {}
        for vehicle_id, route in vehicle_route.items():
            json_per_route[route] = {'nodes': []}

        for node in json_req['nodes']:
            lat_request = node['lat']
            lon_request = node['lon']
            for vehicle_id, route in vehicle_route.items():
                list_nodes = ordered_stops[route]
                if (lat_request, lon_request) in list_nodes:
                    json_per_route[route]['nodes'].append(node)

        for vehicle_id, route in vehicle_route.items():
            for vehicle in json_req['vehicles']:
                if int(vehicle['ident']) == vehicle_id:
                    json_per_route[route]['vehicles'] = [vehicle]

            for key in json_req.keys():
                if key in ['nodes', 'vehicles']:
                    continue
                else:
                    json_per_route[route][key] = json_req[key]

        # Save generated json TSP files
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        for vehicle_id, route in vehicle_route.items():

            if len(json_per_route[route]['nodes']) == 0:
                continue

            path_json_tsp = project_root / f'resultados_adh/json_tsp/Instance {instance}/Instancia-{instance}-{vehicle_id}.json'
            path_json_tsp.parent.mkdir(parents=True, exist_ok=True)
            with open(path_json_tsp, 'w') as fp:
                json.dump(json_per_route[route], fp)

            # Save in a common folder
            path_json_tsp_common = project_root / f'resultados_adh/json_tsp/all_request/Instancia-{instance}-{vehicle_id}.json'
            path_json_tsp_common.parent.mkdir(parents=True, exist_ok=True)
            with open(path_json_tsp_common, 'w') as fp:
                json.dump(json_per_route[route], fp)

    '''

    def plot_adherence_histograms(self, bins=20):

        # Adherence Felipe
        adh_felipe = self.compute_adherence_all_instances()
        # Adherence Tomas
        df_tomas = pd.read_excel('adherencias_df.xlsx')

        # Difference between adherence's
        dif_adherence = []
        max_difference = 0
        max_difference_instance = 0
        for instance in self.instances_ids:
            adh_felipe_inst = np.mean(adh_felipe[instance])
            adh_tomas_inst = df_tomas[df_tomas['num'] == instance]['Adh. Orden'].mean()
            dif_adherence.append(adh_tomas_inst - adh_felipe_inst)
            if np.abs(adh_tomas_inst - adh_felipe_inst) > max_difference:
                max_difference = np.abs(adh_tomas_inst - adh_felipe_inst)
                max_difference_instance = instance

        # Plot histogram Felipe
        adh_all = [a for v in adh_felipe.values() for a in v]
        plt.hist(adh_all, bins=bins, edgecolor="black")
        plt.xlabel("Adherencia")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de adherencia (sin red neuronal) Felipe")
        plt.savefig("adherencia normal Felipe.png")
        plt.show()

        # Plot histogram Tomas
        plt.hist(df_tomas['Adh. Orden'], bins=bins, edgecolor="black")
        plt.xlabel("Adherencia")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de adherencia (sin red neuronal) Tomás")
        plt.savefig("adherencia normal Tomás.png")
        plt.show()

        # Plot histogram difference
        plt.hist(dif_adherence, bins=bins, edgecolor="black")
        plt.xlabel("Adherencia")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de diferencia entre adherencias Felipe y Tomás")
        plt.savefig("adherencia diferencias.png")
        plt.show()

        return max_difference_instance

    def plot_routes_and_stops_interactive(self, instance):

        plan_id = self.plans_ids[instance]
        coordinates_by_routes = self.get_gps_registers(plan_id)
        simpli_routes, simpli_coordinates, ordered_stops, reason_stop, depot_instance = self.get_stops_vehicles(instance)

        # centro aproximado del mapa
        nodes_df = self.dict_to_df(simpli_coordinates, 'vehicle')
        center_lat = nodes_df.latitude.mean()
        center_lon = nodes_df.longitude.mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # ---- Stops ----
        visited_stops_group = FeatureGroup(name="Visited stops")
        distance_stops_group = FeatureGroup(name="Far away stops")
        speed_stops_group = FeatureGroup(name="Speed stops")
        for idx, row in nodes_df.iterrows():

            stop_coord = (row.latitude, row.longitude)

            popup_text = f"""
                <b>Cliente:</b> {idx}<br>
                <b>Lat:</b> {row.latitude:.5f}<br>
                <b>Lon:</b> {row.longitude:.5f}<br>
            """

            # We check if we visited the stop
            visited_point = False
            for stop_list in ordered_stops.values():
                if stop_coord in stop_list:
                    visited_point = True

            if visited_point:
                folium.CircleMarker(
                    location=[row.latitude, row.longitude],
                    radius=3,
                    color="black",
                    fill=True,
                    fill_color="black",
                    fill_opacity=0.9,
                    popup=f"{popup_text}"
                ).add_to(visited_stops_group)

            else:

                if reason_stop[stop_coord] == 'distance':

                    folium.Marker(
                        location=[row.latitude, row.longitude],
                        popup=popup_text,
                        icon=folium.DivIcon(
                            html='<div style="font-size:10px; color:black;">✖</div>'
                        )
                    ).add_to(distance_stops_group)

                elif reason_stop[stop_coord] == 'speed':

                    folium.CircleMarker(
                        location=[row.latitude, row.longitude],
                        radius=4,
                        color="black",  # color del borde
                        weight=1.5,  # grosor del borde
                        fill=False,  # ← sin relleno
                        popup=popup_text
                    ).add_to(speed_stops_group)

        visited_stops_group.add_to(m)
        distance_stops_group.add_to(m)
        speed_stops_group.add_to(m)

        # ---- Routes ----
        n_routes = len(coordinates_by_routes)
        cmap = plt.cm.tab20
        colors = [mcolors.to_hex(cmap(i % 20)) for i in range(n_routes)]

        for i, route in enumerate(coordinates_by_routes.keys()):

            group = FeatureGroup(name=f"Ruta {i + 1}")
            lats, lons, time = coordinates_by_routes[route]
            coords = list(zip(lats, lons))

            folium.PolyLine(
                coords,
                color=colors[i],
                weight=4,
                opacity=0.9,
                tooltip=f"Ruta {i + 1}"
            ).add_to(group)
            group.add_to(m)

            # ---- Time between GPS registers ----
            for j in range(len(coords) - 1):
                lat1, lon1 = coords[j]
                lat2, lon2 = coords[j + 1]

                # midpoint of the segment
                mid_lat = (lat1 + lat2) / 2
                mid_lon = (lon1 + lon2) / 2

                # time difference in seconds
                dt = (time[j + 1] - time[j]) / np.timedelta64(1, 's')
                if dt >= 45:
                    folium.Marker(
                        location=[mid_lat, mid_lon],
                        icon=folium.DivIcon(
                            html=f"""
                            <div style="
                                font-size:9px;
                                color:black;
                                background:white;
                                border-radius:3px;
                                padding:1px 3px;
                                border:1px solid gray;
                            ">
                            {int(dt)}
                            </div>
                            """
                        )
                    ).add_to(group)

            idx = 0
            for lat, lon, t in zip(lats, lons, time):
                t_formatted = pd.to_datetime(t).strftime("%H:%M:%S")
                popup_text = f"""
                    <b>Idx:</b> {idx}<br>
                    <b>Time:</b> {t_formatted}<br>
                    <b>Lat:</b> {lat}<br>
                    <b>Lon:</b> {lon}<br>
                """

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color=colors[i],
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"{popup_text}"
                ).add_to(group)

                idx += 1

        LayerControl().add_to(m)
        m.save(f"map instances/mapa_rutas_{instance}.html")
        return m



    def check_instances():
        n_routes = 0
        n_inst = 0
        for instance in adh_class.instances_ids:
            simpli_routes, simpli_coordinates, ordered_stops, reason_stop, depot_instance = adh_class.get_stops_vehicles(instance)
            instancia_mala = True
            for route_list, list_nodes in ordered_stops.items():
                if len(list_nodes) == 0:
                    continue
                elif len(list_nodes) == 1:
                    if list_nodes[0] == depot_instance[0] and list_nodes[1] == depot_instance[1]:
                        continue
                else:
                    vehicle_route = get_vehicle_id(adh_class, instance, route_list)
                    if len(vehicle_route) > 0:
                        instancia_mala = False
                    n_routes += len(vehicle_route)
            if not instancia_mala:
                n_inst += 1
        print(n_routes, n_inst)

    '''