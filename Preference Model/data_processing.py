import json
import pandas as pd
import geopandas as gpd
from pathlib import Path

class DataLoad:
    def __init__(self, n_envio):

        current_file = Path(__file__)
        project_root = current_file.parent.parent
        self.project_root = project_root

        self.n_envio = n_envio

        # Path data sources
        path_instances = None
        path_gps_data = None
        path_vehicle_route_map = None
        if n_envio == 2:
            path_instances = "data/Envio 2 (tesis Tomás)/Archivos generados por TR/1+ veh/envio2_TESIS1vehV3.xlsx"
            path_gps_data = "data/Envio 2 (tesis Tomás)/gps"
        elif n_envio == 6:
            path_instances = project_root / "data/Falabella/df_instancias.xlsx"
        elif n_envio == 7:
            path_instances = project_root / "data/Envío 7 (Buscalibre)/Características 410 instancias.xlsx"
            path_gps_data = project_root / "data/Envío 7 (Buscalibre)/gps.csv"
            path_vehicle_route_map = project_root / "data/Envío 7 (Buscalibre)/route_vehicle_map.csv"

        # Load data
        data_instances, instances_ids, plans_ids = self.read_data_instances(path_instances)
        data_gps = self.load_gps_data(path_gps_data)
        data_wores, data_woreq = self.load_wores_and_woreq(instances_ids)
        map_vehicle_route = pd.read_csv(path_vehicle_route_map)
        map_vehicle_route = map_vehicle_route.drop_duplicates()

        self.data_instances = data_instances
        self.data_gps = data_gps
        self.data_wores = data_wores
        self.data_woreq = data_woreq
        self.map = map_vehicle_route

        self.instances_ids = instances_ids
        self.plans_ids = plans_ids

        path_comunas = project_root / "clustering/BuscaLibre/comunas/comunas_rm.geojson"
        comunas = gpd.read_file(path_comunas)
        self.comunas = comunas

    def read_data_instances(self, path_data):
        df_instances = pd.read_excel(path_data)

        # # We filter instance
        # df_instances = df_instances[df_instances['n_cli'] >= 100].reset_index(drop=True)

        # Instances
        instances_ids = df_instances.num.values

        # Plan ids
        plans_id = {}
        for inst in instances_ids:
            plan_id = df_instances[df_instances['num'] == inst]['plan_id'].iloc[0]
            plans_id[inst] = plan_id

        return df_instances, instances_ids, plans_id

    def load_gps_data(self, path_gps_data):

        df_gps = None
        if self.n_envio == 2:
            list_df_gps = []
            for i in range(32):
                path_gps_data_i = f'{path_gps_data}/gps-0000000000{str(i).zfill(2)}.csv'
                df_gps_i = pd.read_csv(path_gps_data_i)
                list_df_gps.append(df_gps_i)
            df_gps = pd.concat(list_df_gps, axis=0)
        elif self.n_envio == 7:
            df_gps = pd.read_csv(path_gps_data)

        # Ordenar df temporalmente usando created_at
        df_gps["created_at"] = pd.to_datetime(df_gps["created_at"], utc=True)
        df_gps = df_gps.sort_values("created_at")

        return df_gps

    def load_wores_and_woreq(self, instances_ids):

        inst_path = None
        if self.n_envio == 2:
            inst_path = 'data/Envio 2 (tesis Tomás)/Masked_Instances/'
        elif self.n_envio == 7:
            inst_path = self.project_root / 'data/Envío 7 (Buscalibre)/instancias/'

        wores = {}
        woreq = {}
        for instance in instances_ids:
            path_data = f'{inst_path}/wores/Instancia-{instance}.json'
            with open(path_data, 'r', encoding='utf-8') as file:
                wores_inst = json.load(file)
                wores[instance] = wores_inst

            path_data = f'{inst_path}/woreq/Instancia-{instance}.json'
            with open(path_data, 'r', encoding='utf-8') as file:
                woreq_inst = json.load(file)
                woreq[instance] = woreq_inst

        return wores, woreq

    def filter_instances(self, instances_ids):

        # Preguntar si todos deben empezar y terminar en el mismo depot

        # se optó por considerar exclusivamente los casos donde la planificación no ha sido modificada manualmente
        # por el conductor y reporta idénticas coordenadas de origen-destino para la ruta de todos los vehículos.
        instances_filtered = []
        for instance in instances_ids:

            # We check whether the vehicles start and end at the same location
            woreq_inst = self.data_woreq[instance]
            vehicles_list = woreq_inst['vehicles']
            for vehicle in vehicles_list:

                start_lat = vehicle['location_start']['lat']
                start_lon = vehicle['location_start']['lon']
                end_lat = vehicle['location_end']['lat']
                end_lon = vehicle['location_end']['lon']

                if start_lat == end_lat and start_lon == end_lon:
                    instances_filtered.append(instance)

        return instances_filtered

    def return_data(self):
        return self.data_instances, self.data_gps, self.data_wores, self.data_woreq, self.instances_ids, self.map, self.plans_ids, self.comunas
