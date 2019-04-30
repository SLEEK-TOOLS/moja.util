import io
import os
import simplejson as json
import argparse
import logging
import shutil
import sys
import sqlite3
from itertools import chain
from argparse import ArgumentParser
from glob import iglob
from contextlib import contextmanager

class GCBMConfigurer:

    def __init__(self, layer_paths, template_path, input_db_path, output_path=".",
                 start_year=None, end_year=None, disturbance_order=None):
        self._layer_paths = layer_paths
        self._template_path = template_path
        self._input_db_path = input_db_path
        self._output_path = output_path
        self._start_year = start_year
        self._end_year = end_year
        self._user_disturbance_order = disturbance_order or []

    def configure(self):
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        
        for template in chain.from_iterable(
            iglob(os.path.join(args.template_path, ext))
            for ext in ["*.cfg", "*.json"]
        ):
            shutil.copy(template, self._output_path)
        
        combined_study_area = None
        for layer_path in self._layer_paths:
            study_area = self.get_study_area(layer_path)
            if not combined_study_area:
                combined_study_area = study_area

            current_layer_names = (layer["name"] for layer in combined_study_area["layers"])
            combined_study_area["layers"].extend(filter(
                lambda layer: layer["name"] not in current_layer_names,
                study_area["layers"]))

        self.update_simulation_study_area(combined_study_area)
        self.update_simulation_disturbances(combined_study_area)
        self.add_spinup_data_variables(combined_study_area)
        self.add_simulation_data_variables(combined_study_area)
        self.update_provider_config(combined_study_area)
        if self._start_year and self._end_year:
            self.update_simulation_years(self._start_year, self._end_year)
    
    @staticmethod
    @contextmanager
    def update_json_file(path):
        with open(path, "rb") as json_file:
            contents = json.load(json_file)
            
        yield contents
        
        with io.open(path, "w", encoding="utf8") as json_file:
            json_file.write(json.dumps(contents, indent=4, ensure_ascii=False))
    
    @staticmethod
    def find_config_file(config_path, *search_path):
        for config_file in (fn for fn in iglob(os.path.join(config_path, "*.json"))
                            if "internal" not in fn.lower()):
            # Drill down through the config file contents to see if the whole search path
            # is present; if it is, then this is the right file to modify.
            config = json.load(open(config_file, "r"))
            for entry in search_path:
                config = config.get(entry)
                if config is None:
                    break
            
            if config is not None:
                return config_file
        
        return None

    def update_provider_config(self, study_area):
        provider_config_path = self.find_config_file(self._output_path, "Providers")
        if not provider_config_path:
            logging.fatal("No provider configuration file found in {}".format(self._output_path))
            return
        
        with self.update_json_file(provider_config_path) as provider_config:
            provider_section = provider_config["Providers"]
            layer_config = None
            for provider, config in provider_section.items():
                if "layers" in config:
                    spatial_provider_config = config
                    break

            spatial_provider_config["tileLatSize"]  = study_area["tile_size"]
            spatial_provider_config["tileLonSize"]  = study_area["tile_size"]
            spatial_provider_config["blockLatSize"] = study_area["block_size"]
            spatial_provider_config["blockLonSize"] = study_area["block_size"]
            spatial_provider_config["cellLatSize"]  = study_area["pixel_size"]
            spatial_provider_config["cellLonSize"]  = study_area["pixel_size"]
                    
            provider_layers = []
            for layer in study_area["layers"]:
                logging.debug("Added {} to provider configuration".format(layer))
                provider_layers.append({
                    "name"        : layer["name"],
                    "layer_path"  : os.path.join(os.path.relpath(layer["path"], self._output_path)),
                    "layer_prefix": layer["prefix"]
                })
                
            layer_config = spatial_provider_config["layers"] = provider_layers
            logging.info("Updated provider configuration: {}".format(provider_config_path))

    def update_simulation_study_area(self, study_area):
        config_file_path = self.find_config_file(self._output_path, "LocalDomain", "landscape")
        with self.update_json_file(config_file_path) as study_area_config:
            tile_size    = study_area["tile_size"]
            pixel_size   = study_area["pixel_size"]
            tile_size_px = int(tile_size / pixel_size)
            
            landscape_config = study_area_config["LocalDomain"]["landscape"]
            landscape_config["tile_size_x"] = tile_size
            landscape_config["tile_size_y"] = tile_size
            landscape_config["x_pixels"]    = tile_size_px
            landscape_config["y_pixels"]    = tile_size_px
            landscape_config["tiles"]       = study_area["tiles"]
            logging.info("Study area configuration updated: {}".format(config_file_path))

    def update_simulation_years(self, start_year, end_year):
        config_file_path = self.find_config_file(self._output_path, "LocalDomain", "start_date")
        with self.update_json_file(config_file_path) as study_area_config:
            simulation_config = study_area_config["LocalDomain"]
            simulation_config["start_date"] = "{}/01/01".format(start_year)
            simulation_config["end_date"] = "{}/01/01".format(end_year)
            logging.info("Simulation time period updated: {} to {}".format(start_year, end_year))
            
    def update_simulation_disturbances(self, study_area):
        config_file_path = self.find_config_file(self._output_path, "Modules", "CBMDisturbanceListener")
        with self.update_json_file(config_file_path) as module_config:
            disturbance_listener_config = module_config["Modules"]["CBMDisturbanceListener"]
            if "settings" not in disturbance_listener_config:
                disturbance_listener_config["settings"] = {}

            disturbance_listener_config["settings"]["vars"] = [
                layer["name"] for layer in sorted(
                    filter(self.is_disturbance_layer, study_area["layers"]),
                    key=lambda layer: self.get_disturbance_order(layer))]
            
            if not disturbance_listener_config["settings"]["vars"]:
                disturbance_listener_config["enabled"] = False
                
            logging.info("Disturbance configuration updated: {}".format(config_file_path))

    def add_spinup_data_variables(self, study_area):
        config_file_path = self.find_config_file(self._output_path, "SpinupVariables")
        with self.update_json_file(config_file_path) as spinup_config:
            spinup_variables = spinup_config["SpinupVariables"]
            last_pass_disturbances = spinup_variables.get("last_pass_disturbance_timeseries")
            if not last_pass_disturbances:
                last_pass_disturbances = []
            
            for layer in study_area["layers"]:
                layer_tags = layer.get("tags") or []
                if "last_pass_disturbance" in layer_tags:
                    last_pass_disturbances.append(layer["name"])
            
            if last_pass_disturbances:
                spinup_variables["last_pass_disturbance_timeseries"] = {
                    "transform": {
                        "allow_nulls": "true",
                        "type": "CompositeTransform",
                        "library": "internal.flint",
                        "vars": last_pass_disturbances,
                        "format": "long"
                    }
                }

    def add_simulation_data_variables(self, study_area):
        config_file_path = self.find_config_file(self._output_path, "Variables")
        with self.update_json_file(config_file_path) as variable_config:
            variables = variable_config["Variables"]
            classifier_layers = variables["initial_classifier_set"]["transform"]["vars"]
            reporting_classifier_layers = variables["reporting_classifiers"]["transform"]["vars"]
            
            for layer in study_area["layers"]:
                layer_name = layer["name"]
                layer_type = layer["type"]

                layer_tags = layer.get("tags") or []
                if "classifier" in layer_tags:
                    classifier_layers.append(layer_name)
                elif "reporting_classifier" in layer_tags:
                    reporting_classifier_layers.append(layer_name)
                    
                variables[layer_name] = {
                    "transform": {
                        "library"      : "internal.flint",
                        "type"         : "TimeSeriesIdxFromFlintDataTransform",
                        "provider"     : "RasterTiled",
                        "data_id"      : layer_name,
                        "sub_same"     : "true",
                        "start_year"   : 0,
                        "data_per_year": layer["nStepsPerYear"],
                        "n_years"      : layer["nLayers"]
                    }
                } if layer_type == "RegularStackLayer" else {
                    "transform": {
                        "library" : "internal.flint",
                        "type"    : "LocationIdxFromFlintDataTransform",
                        "provider": "RasterTiled",
                        "data_id" : layer_name
                    }
                }
                
            logging.info("Variable configuration updated: {}".format(config_file_path))
    
    def is_disturbance_layer(self, layer):
        layer_tags = layer.get("tags") or []
        return "disturbance" in layer_tags and "last_pass_disturbance" not in layer_tags
        
    def get_default_disturbance_order(self):
        conn = sqlite3.connect(self._input_db_path)
        return [row[0] for row in conn.execute("SELECT name FROM disturbance_type ORDER BY code")]

    def get_disturbance_order(self, layer):
        disturbance_type = self.get_disturbance_type(layer)
        default_disturbance_order = self.get_default_disturbance_order()
        return -len(self._user_disturbance_order) + self._user_disturbance_order.index(disturbance_type) \
            if disturbance_type in self._user_disturbance_order \
            else default_disturbance_order.index(disturbance_type)
    
    def get_disturbance_type(self, layer):
        metadata_file = os.path.join(layer["path"], "{}_moja.json".format(layer["name"]))
        metadata = json.load(open(metadata_file, "rb"))
        dist_type = next((attr for attr in metadata["attributes"].values())).get("disturbance_type")
        
        return dist_type
    
    def scan_for_layers(self, layer_root):
        provider_layers = []
        layers = {fn for fn in os.listdir(layer_root)
                  if os.path.isdir(os.path.join(layer_root, fn))
                  or fn.endswith(".zip")}
        
        for layer in layers:
            logging.info("Found layer: {}".format(layer))
            layer_prefix, _ = os.path.splitext(os.path.basename(layer))
            layer_path = os.path.join(layer_root, layer_prefix)
            layer_name, _ = layer_prefix.split("_moja")
            provider_layers.append({
                "name"  : layer_name,
                "type"  : None,
                "path"  : layer_path,
                "prefix": layer_prefix
            })
            
        return provider_layers
        
    def get_study_area(self, layer_root):
        study_area = {
            "tile_size" : 1.0,
            "block_size": 0.1,
            "pixel_size": 0.00025,
            "tiles"     : [],
            "layers"    : []
        }
        
        study_area_path = os.path.join(layer_root, "study_area.json")
        if os.path.exists(study_area_path):
            with open(study_area_path, "rb") as study_area_file:
                study_area.update(json.load(study_area_file))

        # Find all of the layers for the simulation physically present on disk, then
        # add any extra metadata available from the tiler's study area output.
        layers = self.scan_for_layers(layer_root)
        study_area_layers = study_area.get("layers")
        if study_area_layers:
            for layer in layers:
                for layer_metadata \
                in filter(lambda l: l.get("name") == layer.get("name"), study_area_layers):
                    layer.update(layer_metadata)
        
        study_area["layers"] = layers
       
        return study_area

if __name__ == "__main__":
    logging.basicConfig(filename=r"logs\update_gcbm_config.log", filemode="w",
						level=logging.INFO, format="%(message)s")

    parser = ArgumentParser(description="Update GCBM spatial provider configuration.")
    parser.add_argument("--layer_root", help="one or more directories containing tiled layers and study area metadata", nargs="+", type=os.path.abspath)
    parser.add_argument("--template_path", help="GCBM config file template path", required=True, type=os.path.abspath)
    parser.add_argument("--input_db_path", help="GCBM input database path", required=True, type=os.path.abspath)
    parser.add_argument("--output_path", help="GCBM config file output path", default=".", type=os.path.abspath)
    parser.add_argument("--start_year", help="simulation start year")
    parser.add_argument("--end_year", help="simulation end year")
    args = parser.parse_args()
    
    configurer = GCBMConfigurer(
        args.layer_root, args.template_path, args.input_db_path,
        args.output_path, args.start_year, args.end_year)
    
    configurer.configure()