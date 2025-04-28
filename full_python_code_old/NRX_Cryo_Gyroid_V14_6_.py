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
# Core Mathematical Functions
# ================================================================

@njit(fastmath=True, cache=True, inline='always')
def gyroid(x, y, z, freq):
    """Calculate pure gyroid surface value"""
    return (
        np.sin(freq*x)*np.cos(freq*y) + 
        np.sin(freq*y)*np.cos(freq*z) + 
        np.sin(freq*z)*np.cos(freq*x)
    )

@njit(fastmath=True, cache=True)
def compute_frequency(r, r_inner, r_outer, f0, f_min, alpha):
    """Radial frequency gradient calculation"""
    r_clamped = min(max(r, r_inner), r_outer)
    ratio = (r_outer - r_clamped)/(r_outer - r_inner)
    return f_min + (f0 - f_min)*(ratio**alpha)

@njit(parallel=True, fastmath=True, nogil=True)
def compute_boundary_transition(X, Y, transition_params):
    """Calculate boundary transition field B(r)"""
    B = np.ones_like(X, dtype=np.float32)
    r_inner, r_outer, transition_width = transition_params
    
    if transition_width == 0:
        return B
    
    for i in prange(X.shape[0]):
        for j in prange(X.shape[1]):
            for k in prange(X.shape[2]):
                x, y = X[i,j,k], Y[i,j,k]
                r = sqrt(x**2 + y**2)
                
                # Inner boundary transition
                inner_term = (r_inner - r)/transition_width
                inner_smooth = 0.5 * (1 + np.tanh(inner_term))
                
                # Outer boundary transition
                outer_term = (r_outer - r)/transition_width
                outer_smooth = 0.5 * (1 + np.tanh(outer_term))
                
                B[i,j,k] = inner_smooth * outer_smooth
    
    return B

# ================================================================
# Enhanced Computation Engine
# ================================================================

@njit(parallel=True, fastmath=True, nogil=True)
def parallel_compute(X, Y, Z, params, transition_params):
    """Generate gyroid field with boundary transition"""
    G = np.empty_like(X, dtype=np.float32)
    r_inner, r_outer, f0, f_min, alpha, buffer = params
    
    # Compute gyroid field
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
    
    # Apply boundary transition
    B = compute_boundary_transition(X, Y, transition_params)
    return G * B

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
        """Create cylindrical-aligned grid"""
        buffer_factor = 1 + (buffer_percent/100)
        x = np.linspace(
            -self.params['r_outer']*buffer_factor,
            self.params['r_outer']*buffer_factor,
            resolution,
            dtype=np.float32
        )
        y = x.copy()
        z = np.linspace(0, self.params['height'], resolution, dtype=np.float32)
        return np.meshgrid(x, y, z, indexing='ij')
    
    def generate(self, args):
        """Enhanced generation workflow with boundary transition"""
        tracemalloc.start()
        
        try:
            X, Y, Z = self.create_grid(args.resolution, args.buffer)
            
            # Main parameters
            main_params = (
                self.params['r_inner'],
                self.params['r_outer'],
                self.params['f0'],
                self.params['f_min'],
                self.params['alpha'],
                args.buffer
            )
            
            # Transition parameters
            transition_params = (
                self.params['r_inner'],
                self.params['r_outer'],
                args.transition_width
            )
            
            # Compute final field
            F = parallel_compute(X, Y, Z, main_params, transition_params)
            
            grid = pv.StructuredGrid(X, Y, Z)
            grid["F"] = F.ravel(order='F')
            mesh = grid.contour([0.0])
            
            if mesh.n_points == 0:
                raise RuntimeError("Surface extraction failed - Check parameters")
            
            return mesh
        
        except MemoryError:
            peak = tracemalloc.get_traced_memory()[1]
            print(f"Memory overflow ({peak/1e9:.2f} GB)")
            print("Solutions: 1. Lower resolution 2. Reduce buffer 3. Use --safe")
            return None
        
        finally:
            tracemalloc.stop()

# ================================================================
# Visualization System
# ================================================================

class VisualizationEngine:
    def __init__(self):
        self.plotter = pv.Plotter()
        self.plotter.set_background("#1f1f1f")
        self.plotter.enable_anti_aliasing('ssaa')
    
    def show(self, mesh, clip_half=False):
        """Visualize the enhanced structure"""
        if clip_half:
            mesh = mesh.clip(normal='x', invert=False)
        
        self.plotter.add_mesh(mesh, scalars="F", cmap="coolwarm",
                            smooth_shading=True, show_edges=False,
                            scalar_bar_args={'title': "Field Value"})
        
        self.plotter.add_text(
            f"Nodes: {mesh.n_points:,}\nCells: {mesh.n_cells:,}",
            position="lower_right", font_size=12, color="white")
        
        self.plotter.show()

# ================================================================
# Enhanced CLI Interface
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced Gyroid Generator")
    parser.add_argument("-r", "--resolution", type=int, default=400,
                      help="3D grid resolution")
    parser.add_argument("-b", "--buffer", type=float, default=5.0,
                      help="Buffer zone percentage")
    parser.add_argument("-tw", "--transition-width", type=float, default=0.5,
                      help="Boundary transition width (mm)")
    parser.add_argument("--safe", action='store_true',
                      help="Enable memory-safe mode")
    parser.add_argument("--clip", action='store_true',
                      help="Display front half only")
    
    args = parser.parse_args()
    
    if args.safe:
        args.resolution = min(args.resolution, 350)
        args.buffer = min(args.buffer, 10.0)
        warnings.warn(f"Safe mode: Resolution={args.resolution}, Buffer={args.buffer}%")
    
    generator = GyroidGenerator()
    engine = VisualizationEngine()
    
    try:
        mesh = generator.generate(args)
        if mesh is not None:
            engine.show(mesh, args.clip)
    except Exception as e:
        print(f"Generation error: {str(e)}")
    finally:
        print("Operation completed")

if __name__ == "__main__":
    main()