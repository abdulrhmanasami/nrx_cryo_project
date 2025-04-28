#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pyvista as pv
from numba import njit, prange
import warnings
import tracemalloc
from math import sqrt

# ================================================================
# Core Gyroid Generation
# ================================================================

@njit(fastmath=True, cache=True, inline='always')
def gyroid(x, y, z, freq):
    return (
        np.sin(freq*x)*np.cos(freq*y) + 
        np.sin(freq*y)*np.cos(freq*z) + 
        np.sin(freq*z)*np.cos(freq*x)
    )

@njit(fastmath=True, cache=True)
def compute_frequency(r, r_inner, r_outer, f0, f_min, alpha):
    r_clamped = min(max(r, r_inner), r_outer)
    ratio = (r_outer - r_clamped)/(r_outer - r_inner)
    return f_min + (f0 - f_min)*(ratio**alpha)

@njit(parallel=True, fastmath=True, nogil=True)
def parallel_compute(X, Y, Z, params):
    G = np.empty_like(X, dtype=np.float32)
    r_inner, r_outer, f0, f_min, alpha, buffer = params
    
    for i in prange(X.shape[0]):
        for j in prange(X.shape[1]):
            for k in prange(X.shape[2]):
                x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                r = sqrt(x**2 + y**2)
                if r > r_outer * (1 + buffer/100):
                    G[i,j,k] = -1.0
                else:
                    freq = compute_frequency(r, r_inner, r_outer, f0, f_min, alpha)
                    G[i,j,k] = gyroid(x, y, z, freq)
    return G

# ================================================================
# Thickness Generation System
# ================================================================

def apply_wall_thickness(mesh, thickness):
    """Apply bidirectional offset to create solid walls"""
    try:
        outer = mesh.offset(thickness/2, inplace=False)
        inner = mesh.offset(-thickness/2, inplace=False)
        return outer.merge(inner).extract_surface()
    except Exception as e:
        warnings.warn(f"Offset failed: {str(e)}")
        return mesh

# ================================================================
# Main Generator Class
# ================================================================

class GyroidGenerator:
    def __init__(self):
        self.params = {
            'r_inner': 77.03,
            'r_outer': 84.14,
            'height': 300.0,
            'f0': 0.7019,
            'f_min': 0.0702,
            'alpha': 0.3
        }
    
    def create_grid(self, resolution, buffer_percent):
        buffer_factor = 1 + (buffer_percent/100)
        x = np.linspace(-self.params['r_outer']*buffer_factor,
                        self.params['r_outer']*buffer_factor,
                        resolution, dtype=np.float32)
        y = x.copy()
        z = np.linspace(0, self.params['height'], resolution, dtype=np.float32)
        return np.meshgrid(x, y, z, indexing='ij')
    
    def generate(self, args):
        tracemalloc.start()
        try:
            X, Y, Z = self.create_grid(args.resolution, args.buffer)
            params = (self.params['r_inner'], self.params['r_outer'],
                     self.params['f0'], self.params['f_min'],
                     self.params['alpha'], args.buffer)
            
            G = parallel_compute(X, Y, Z, params)
            grid = pv.StructuredGrid(X, Y, Z)
            grid["F"] = G.ravel(order='F')
            mesh = grid.contour([0.0])
            
            if args.apply_thickness:
                mesh = apply_wall_thickness(mesh, args.thickness_value)
            
            if mesh.n_points == 0:
                raise RuntimeError("Surface generation failed")
            
            return mesh
        except MemoryError:
            peak = tracemalloc.get_traced_memory()[1]
            print(f"Memory overflow ({peak/1e9:.2f} GB)")
            print("Solutions: Reduce resolution/buffer or use --safe")
            return None
        finally:
            tracemalloc.stop()

# ================================================================
# CLI Interface
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Industrial Gyroid Generator")
    parser.add_argument("-r", "--resolution", type=int, default=400)
    parser.add_argument("-b", "--buffer", type=float, default=5.0)
    parser.add_argument("--thickness-value", type=float, default=0.3,
                      help="Wall thickness in mm")
    parser.add_argument("--apply-thickness", action='store_true',
                      help="Enable industrial wall thickness")
    parser.add_argument("--safe", action='store_true')
    parser.add_argument("--export", type=str, help="Export filename (STL/VTK)")
    
    args = parser.parse_args()
    
    if args.safe:
        args.resolution = min(args.resolution, 350)
        args.buffer = min(args.buffer, 10.0)
        warnings.warn(f"Safe mode: res={args.resolution}, buffer={args.buffer}%")
    
    generator = GyroidGenerator()
    
    try:
        mesh = generator.generate(args)
        if mesh is not None:
            if args.export:
                mesh.save(args.export)
                print(f"Exported: {args.export}")
            else:
                plotter = pv.Plotter()
                plotter.add_mesh(mesh, color='white', show_edges=True)
                plotter.show()
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("Operation completed")

if __name__ == "__main__":
    main()