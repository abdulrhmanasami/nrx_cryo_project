#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pyvista as pv
from numba import njit, prange
import tracemalloc
import time
import psutil
import warnings
from math import ceil, sqrt

# ================================================================
# Optimized Core Functions
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
def compute_chunk(X_chunk, Y_chunk, Z_chunk, params):
    G_chunk = np.empty_like(X_chunk, dtype=np.float32)
    r_inner, r_outer, f0, f_min, alpha, buffer = params
    
    for i in prange(X_chunk.shape[0]):
        for j in prange(X_chunk.shape[1]):
            for k in prange(X_chunk.shape[2]):
                x, y, z = X_chunk[i,j,k], Y_chunk[i,j,k], Z_chunk[i,j,k]
                r = sqrt(x**2 + y**2)
                
                if r > r_outer * (1 + buffer/100):
                    G_chunk[i,j,k] = -1.0
                else:
                    freq = compute_frequency(r, r_inner, r_outer, f0, f_min, alpha)
                    G_chunk[i,j,k] = gyroid(x, y, z, freq)
    return G_chunk

# ================================================================
# Memory-Optimized Generator
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
        self.total_mesh = None
    
    def create_chunked_grid(self, resolution, buffer_percent, chunk_size):
        buffer_factor = 1 + (buffer_percent/100)
        x = np.linspace(-self.params['r_outer']*buffer_factor,
                        self.params['r_outer']*buffer_factor,
                        resolution, dtype=np.float32)
        y = x.copy()
        
        z_full = np.linspace(0, self.params['height'], resolution, dtype=np.float32)
        n_chunks = ceil(resolution / chunk_size)
        return x, y, z_full, n_chunks
    
    def generate_chunk(self, args, x, y, z_slice):
        X, Y, Z = np.meshgrid(x, y, z_slice, indexing='ij')
        params = (self.params['r_inner'], self.params['r_outer'],
                 self.params['f0'], self.params['f_min'],
                 self.params['alpha'], args.buffer)
        
        G = compute_chunk(X, Y, Z, params)
        grid = pv.StructuredGrid(X, Y, Z)
        grid["F"] = G.ravel(order='F')
        return grid.contour([0.0])
    
    def safe_mode_adjustment(self, args):
        mem = psutil.virtual_memory()
        expected_mem = (args.resolution**3 * 4 * 3) / 1e9
        
        if expected_mem > mem.available * 0.7:
            args.resolution = min(args.resolution, 350)
            args.buffer = min(args.buffer, 10.0)
            args.chunk_size = min(args.chunk_size, 32)
            warnings.warn(f"Auto-safe: res={args.resolution}, chunk={args.chunk_size}")
    
    def generate(self, args):
        start_time = time.perf_counter()
        tracemalloc.start()
        
        try:
            if args.auto_safe:
                self.safe_mode_adjustment(args)
            
            x, y, z_full, n_chunks = self.create_chunked_grid(
                args.resolution, args.buffer, args.chunk_size)
            
            for chunk_idx in range(n_chunks):
                if psutil.virtual_memory().percent > 90:
                    raise MemoryError("Memory usage exceeds 90%")
                
                start = chunk_idx * args.chunk_size
                end = min((chunk_idx+1)*args.chunk_size, args.resolution)
                z_chunk = z_full[start:end]
                
                chunk_mesh = self.generate_chunk(args, x, y, z_chunk)
                
                if self.total_mesh is None:
                    self.total_mesh = chunk_mesh
                else:
                    self.total_mesh += chunk_mesh
                
                print(f"Processed chunk {chunk_idx+1}/{n_chunks}")
            
            return self.total_mesh
        
        except MemoryError as e:
            print(f"[ERROR] {str(e)}")
            return None
        finally:
            peak_mem = tracemalloc.get_traced_memory()[1]/1e9
            total_time = time.perf_counter() - start_time
            print(f"[INFO] Peak Memory: {peak_mem:.1f} GB")
            print(f"[INFO] Total Time: {total_time:.1f} seconds")
            tracemalloc.stop()

# ================================================================
# Enhanced CLI Interface
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized Gyroid Generator")
    parser.add_argument("-r", "--resolution", type=int, default=400)
    parser.add_argument("-b", "--buffer", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=64,
                      help="Z-axis layers per chunk")
    parser.add_argument("--auto-safe", action='store_true',
                      help="Enable automatic safety adjustments")
    parser.add_argument("--export", type=str)
    
    args = parser.parse_args()
    
    generator = GyroidGenerator()
    mesh = generator.generate(args)
    
    if mesh and args.export:
        mesh.save(args.export)
        print(f"Exported to {args.export}")
    elif mesh:
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True)
        plotter.show()

if __name__ == "__main__":
    main()