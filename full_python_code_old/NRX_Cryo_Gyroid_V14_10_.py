#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pyvista as pv
from numba import njit, prange
import time
import psutil
import json
import os
import logging
import sys
import platform
import importlib.metadata
from datetime import datetime
from math import sin, cos

NRX_VERSION = "3.0.0"
DEFAULT_CONFIG = {
    'resolution': 100,
    'unit_size': 5.0,
    'buffer': 2.0,
    'wall_thickness': 0.5
}

class ExecutionProfiler:
    @classmethod
    def collect_metadata(cls):
        return {
            'system': cls._get_system_info(),
            'software': cls._get_software_stack(),
            'execution': cls._get_execution_context()
        }

    @staticmethod
    def _get_system_info():
        return {
            'os': f"{platform.system()} {platform.release()}",
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'cpu_cores': os.cpu_count()
        }

    @staticmethod
    def _get_software_stack():
        return {
            'nrx_version': NRX_VERSION,
            'python': sys.version,
            'libraries': {
                lib: importlib.metadata.version(lib)
                for lib in ['numpy', 'pyvista', 'numba', 'psutil']
            }
        }

    @staticmethod
    def _get_execution_context():
        return {
            'command': ' '.join(sys.argv),
            'timestamp': datetime.now().isoformat(),
            'working_dir': os.getcwd()
        }

class IndustrialLogger:
    def __init__(self, log_file=None):
        self.logger = logging.getLogger('NRXIndustrial')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter(
                    '[%(asctime)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
                file_handler.setLevel(logging.DEBUG)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"File logging error: {str(e)}")
        
        self._log_system_profile()

    def _log_system_profile(self):
        try:
            metadata = ExecutionProfiler.collect_metadata()
            self.logger.info(f"=== NRX Generator {NRX_VERSION} ===")
            self.logger.info(f"System RAM: {metadata['system']['ram_gb']} GB")
            self.logger.info("Dependencies:")
            for lib, ver in metadata['software']['libraries'].items():
                self.logger.info(f"{lib:10}: {ver}")
            self.logger.info("="*60)
        except Exception as e:
            self.logger.error(f"System profiling failed: {str(e)}")

    def get_logger(self):
        return self.logger

class PerformanceMonitor:
    def __init__(self, logger):
        self.logger = logger
        self.phases = {}
        self.current_phase = None

    def start_phase(self, name):
        self.current_phase = name
        self.phases[name] = {
            'start': time.time(),
            'memory': psutil.Process().memory_info().rss,
            'duration': 0.0,
            'memory_delta': 0.0
        }
        self.logger.info(f"Starting phase: {name}")

    def end_phase(self):
        if self.current_phase and self.current_phase in self.phases:
            phase = self.phases[self.current_phase]
            try:
                phase['duration'] = time.time() - phase['start']
                current_mem = psutil.Process().memory_info().rss
                phase['memory_delta'] = (current_mem - phase['memory']) / (1024**2)
                self.logger.info(
                    f"Completed {self.current_phase} in {phase['duration']:.2f}s | "
                    f"Memory Δ: {phase['memory_delta']:.1f} MB"
                )
            except KeyError:
                self.logger.warning("Invalid phase metrics")
            finally:
                self.current_phase = None

    def generate_report(self):
        report = ["=== Performance Metrics ==="]
        total = sum(p.get('duration', 0.0) for p in self.phases.values())
        report.append(f"Total Time: {total:.2f} seconds")
        for name, data in self.phases.items():
            report.append(
                f"{name:20}: {data.get('duration', 0.0):.2f}s | "
                f"Memory: {data.get('memory_delta', 0.0):.1f} MB"
            )
        self.logger.info('\n'.join(report))

class IndustrialValidator:
    def __init__(self, mesh):
        self.mesh = mesh
        self.validation_report = []
        self.valid = True

    def validate_all(self):
        self._check_wall_thickness()
        self._check_bounding_box()
        self._check_mesh_integrity()
        return self.valid

    def _check_wall_thickness(self, min_thickness=0.3):
        try:
            edges = self.mesh.extract_feature_edges(feature_angle=45)
            if edges.n_cells == 0:
                self._add_result("WallThickness", True)
                return

            voxels = self.mesh.voxelize(tolerance=0.1)
            thickness = voxels.compute_implicit_distance() * 2
            min_value = np.min(thickness['implicit_distance'])
            self._add_result("WallThickness", min_value >= min_thickness, 
                           f"Min: {min_value:.2f}mm")

        except Exception as e:
            self._add_result("WallThickness", False, "Check failed")

    def _check_bounding_box(self, max_dims=(300, 300, 400)):
        bounds = self.mesh.bounds
        dims = (bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        self._add_result("BoundingBox", all(d <= m for d, m in zip(dims, max_dims)),
                       f"Dims: {dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f}mm")

    def _check_mesh_integrity(self):
        connected = self.mesh.connectivity()
        self._add_result("MeshIntegrity", connected.n_blocks == 1,
                       f"Components: {connected.n_blocks}")

    def _add_result(self, check, passed, details=None):
        status = "PASS" if passed else "FAIL"
        self.validation_report.append({
            'check': check,
            'status': status,
            'details': details
        })
        if not passed: self.valid = False

    def generate_report(self):
        report = ["=== Validation Report ==="]
        for item in self.validation_report:
            line = f"[{item['status']}] {item['check']}"
            if item['details']: line += f" ({item['details']})"
            report.append(line)
        report.append("="*25)
        report.append("Validation Passed" if self.valid else "Validation Failed")
        return '\n'.join(report)

class GyroidGenerator:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('NRXGenerator')
        self.performance = PerformanceMonitor(self.logger)
        self.mesh = None
        self.unit_size = None

    def generate(self, config, preview=False):
        try:
            self._validate_config(config)
            self.unit_size = config['unit_size']
            
            self.performance.start_phase("LatticeConstruction")
            grid = self._create_lattice(
                config['resolution'],
                config['unit_size'],
                config['buffer']
            )
            self.performance.end_phase()

            self.performance.start_phase("FieldComputation")
            field = self._compute_field(grid)
            self.performance.end_phase()

            self.performance.start_phase("SurfaceExtraction")
            self.mesh = self._extract_surface(field, config['wall_thickness'])
            self.performance.end_phase()

            if preview:
                self._show_preview()

            return self.mesh

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise

    def _validate_config(self, config):
        if config['resolution'] < 50 or config['resolution'] > 500:
            raise ValueError("Invalid resolution (50-500)")
        if config['wall_thickness'] < 0.1:
            raise ValueError("Minimum wall thickness 0.1mm")
        if config['unit_size'] <= 0:
            raise ValueError("Unit size must be positive")

    def _create_lattice(self, resolution, unit_size, buffer):
        try:
            axis = np.linspace(
                -unit_size - buffer,
                unit_size + buffer,
                resolution
            )
            return np.meshgrid(axis, axis, axis, indexing='ij')
        except MemoryError:
            self.logger.error("Memory allocation failed")
            raise

    @staticmethod
    @njit(parallel=True)
    def _compute_field(grid):
        X, Y, Z = grid
        field = np.zeros_like(X)
        for i in prange(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                    field[i,j,k] = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
        return field

    def _extract_surface(self, field, thickness):
        try:
            max_range = self.unit_size + DEFAULT_CONFIG['buffer']
            x = np.linspace(-max_range, max_range, field.shape[0])
            y = np.linspace(-max_range, max_range, field.shape[1])
            z = np.linspace(-max_range, max_range, field.shape[2])
            
            grid = pv.RectilinearGrid(x, y, z)
            grid.point_data['values'] = field.flatten(order='F')
            mesh = grid.contour([thickness]).clean()
            
            if mesh.n_cells == 0:
                raise ValueError("No surface detected")
                
            return mesh
        except Exception as e:
            self.logger.error(f"Surface error: {str(e)}")
            raise

    def export(self, filename):
        self.performance.start_phase("Export")
        try:
            if not filename.endswith('.stl'):
                filename += '.stl'
            if self.mesh.n_cells == 0:
                raise ValueError("Empty mesh")
            self.mesh.save(filename)
            self.logger.info(f"Exported: {filename}")
        except Exception as e:
            self.logger.error(f"Export error: {str(e)}")
            raise
        finally:
            self.performance.end_phase()

    def _show_preview(self):
        if self.mesh is None or self.mesh.n_cells == 0:
            self.logger.warning("Nothing to preview")
            return

        try:
            plotter = pv.Plotter(title=f"NRX Preview {NRX_VERSION}")
            plotter.add_mesh(self.mesh, 
                           color='#1F77B4',
                           show_edges=True,
                           opacity=0.9)
            plotter.add_axes()
            plotter.show()
        except Exception as e:
            self.logger.error(f"Preview failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description=f"NRX Industrial Gyroid Generator {NRX_VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-o", "--output", required=True,
                      help="Output STL filename")
    parser.add_argument("-r", "--resolution", type=int, 
                      default=DEFAULT_CONFIG['resolution'],
                      help="Grid resolution (50-500)")
    parser.add_argument("-u", "--unit-size", type=float,
                      default=DEFAULT_CONFIG['unit_size'],
                      help="Unit cell size (mm)")
    parser.add_argument("-b", "--buffer", type=float,
                      default=DEFAULT_CONFIG['buffer'],
                      help="Buffer zone (mm)")
    parser.add_argument("-t", "--wall-thickness", type=float,
                      default=DEFAULT_CONFIG['wall_thickness'],
                      help="Wall thickness (mm)")
    parser.add_argument("--preview", action='store_true',
                      help="Show 3D preview window")
    parser.add_argument("--log-file", help="Log file path")
    
    args = parser.parse_args()
    
    try:
        logger = IndustrialLogger(args.log_file).get_logger()
        generator = GyroidGenerator(logger)
        
        generator.generate({
            'resolution': args.resolution,
            'unit_size': args.unit_size,
            'buffer': args.buffer,
            'wall_thickness': args.wall_thickness
        }, args.preview)
        
        generator.export(args.output)
        generator.performance.generate_report()
        
        validator = IndustrialValidator(generator.mesh)
        if validator.validate_all():
            logger.info(validator.generate_report())
        else:
            logger.error(validator.generate_report())
            raise ValueError("Validation failed")
            
        logger.info("=== Process Completed ===")

    except Exception as e:
        logging.getLogger('NRXBasic').critical(f"Critical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()