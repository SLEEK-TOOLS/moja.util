import os
import logging
import argparse
import sys
import csv
import subprocess
import rasterio
import imageio
import psutil
from numpy.ma import masked
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from moviepy.editor import VideoClip
from glob import glob
from future.utils import viewitems
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing import cpu_count

class FrameGenerator(object):
    
    def __init__(self, name, files, output_path, gdal_mem=64):
        self.name = name
        self.files = files
        self.output_path = output_path
        self.color_table = None
        self.color_table_file = None
        self.gdal_mem = gdal_mem
        
    def scale(self, in_size, out_size):
        in_width, in_height = in_size
        out_width, out_height = out_size
        scale_factor = float(out_width) / in_width if in_width > in_height else float(out_height) / in_height
        
        return (int(in_width * scale_factor), int(in_height * scale_factor))
        
    def make_frame(self, t):
        if not self.color_table:
            self.create_color_table()
    
        in_file = self.files[int(t)]
        frame_file = os.path.join(self.output_path, "{}_{}.png".format(self.name, int(t)))
        self.colorize(in_file, frame_file)

        # Add extra information to the colorized image: year, legend, etc.
        img = Image.open(frame_file)
        out_size = (1280, 720)
        img = img.resize(self.scale(img.size, out_size))
        canvas_size = img.size
        canvas = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 20)
        year = os.path.splitext(in_file)[0].rsplit("_", 1)[1]
        canvas.text((5, canvas_size[1] - 30), year, font=font, fill=(0, 0, 0, 255))
        canvas.text((5, 5), self.name, font=font, fill=(0, 0, 0, 255))

        legend_line_width = 2
        legend_top_label = str(round(max(self.color_table.keys()), 4))
        legend_bottom_label = str(round(min(self.color_table.keys()), 4))
        legend_label_width = max((len(legend_top_label), len(legend_bottom_label))) * 15
        legend_start = (canvas_size[0] - legend_label_width, 30)
        legend_end = (legend_start[0], len(self.color_table) * legend_line_width + legend_start[1])
        canvas.text((legend_start[0], 0), legend_top_label, font=font, fill=(0, 0, 0, 255))
        canvas.text(legend_end, legend_bottom_label, font=font, fill=(0, 0, 0, 255))
                    
        for i, value in enumerate(sorted(self.color_table, reverse=True)):
            color = self.color_table[value]
            line_start = (legend_start[0] + legend_label_width / 2 - 10, legend_start[1] + i * legend_line_width)
            line_end = (legend_start[0] + legend_label_width / 2 + 10, legend_start[1] + i * legend_line_width + legend_line_width)
            canvas.rectangle((line_start, line_end), fill=color, outline=color)

        img.save(frame_file)
        data = imageio.imread(frame_file)
        os.remove(frame_file)
        
        return data
        
    def create_color_table(self):
        min_max_by_file = [self.get_min_max(file) for file in self.files]
        min_value = min((stats[0] for stats in min_max_by_file))
        max_value = max((stats[1] for stats in min_max_by_file))
        step = (max_value - min_value) / 256
        self.color_table = {min_value + i * step: (125, i, 0, 255) for i in range(255)}
       
        self.color_table_file = os.path.join(self.output_path, "{}_color.txt".format(self.name))
        with open(self.color_table_file, "w") as color_file:
            writer = csv.writer(color_file)
            color_map = [["nv", 255, 255, 255, 255]] + [[val] + list(colors) for val, colors
                                                        in viewitems(self.color_table)]
            writer.writerows(color_map)
        
    def colorize(self, in_file, out_file):
        subprocess.check_call([
            "gdaldem",
            "color-relief",
            os.path.abspath(in_file),
            os.path.abspath(self.color_table_file),
            os.path.abspath(out_file),
            "-of", "png",
            "-q",
            "--config", "GDAL_CACHEMAX", str(self.gdal_mem)])
        
    def get_min_max(self, file):
        try:
            with rasterio.open(file, "r") as src:
                raster_data = src.read(masked=True)
                first_band = raster_data[:1]
                min = first_band.min()
                max = first_band.max()
                return (min if min is not masked else 0,
                        max if max is not masked else 0)
        except:
            return (0, 0)

def create_animation(name, files, output_path, worker_mem):
    '''
    Creates an animation single tiff file from a list of tiff files.
    '''
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    
    frame_generator = FrameGenerator(name, files, output_path, worker_mem)
    animation = VideoClip(frame_generator.make_frame, duration=len(files))
    animation.write_videofile(os.path.join(output_path, "{}.mp4".format(name)), fps=1)
    
    return name
    
def find_spatial_output(root_path):
    '''
    Gathers up all spatial output rooted in root_path by indicator and year.
    '''
    spatial_output = defaultdict(list)
    for dir, subdirs, files in os.walk(root_path):
        for file in sorted(filter(lambda f: f.endswith(".tiff"), files)):
            file_path = os.path.abspath(os.path.join(dir, file))
            indicator, year = os.path.splitext(file)[0].rsplit("_", 1)
            spatial_output[indicator].append(file_path)
    
    return spatial_output
    
def process_spatial_output(spatial_output, output_path=".", pool_size=4):
    '''
    Generates a merged tiff file for each indicator at each year from a dictionary
    of {indicator: {year: [.tiff files]}}
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    available_mem = psutil.virtual_memory().available
    worker_mem = int(available_mem * 0.8 / pool_size)
    pool = Pool(pool_size)
    for indicator, files in viewitems(spatial_output):
        pool.apply_async(create_animation, (indicator, files, output_path, worker_mem),
                         callback=logging.info)
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = ArgumentParser(description="Creates an animation from a timeseries of geotiffs.")
        
    parser.add_argument("--indicator_root", required=False, default=".",
                        help="path to the spatial output root directory")

    parser.add_argument("--output_path", required=False, default=".",
                        help="path to store generated animation in - will be created if it doesn't exist")

    parser.add_argument("--pool_size", help="Process pool size", required=False, default=4, type=int)

    args = parser.parse_args()
    spatial_output = find_spatial_output(args.indicator_root)
    if spatial_output:
        logging.info("Found {} total spatial indicators.".format(len(spatial_output)))
        process_spatial_output(spatial_output, args.output_path, args.pool_size)
        logging.info("Done")
    else:
        logging.info("No spatial output found.")
