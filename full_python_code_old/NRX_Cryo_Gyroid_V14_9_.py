#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pyvista as pv
from numba import njit, prange
import warnings
from math import sqrt

# ================================================================
# Core Gyroid Generation (Optimized)
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
# Gyroid Generator Class
# ================================================================

class GyroidGenerator:
    def __init__(self):
        # Default parameters
        self.r_inner = 0.3
        self.r_outer = 0.8
        self.f0 = 8.0
        self.f_min = 2.0
        self.alpha = 1.5
        self.threshold = 0.0
    
    def generate(self, args):
        """Generate a gyroid mesh based on input parameters"""
        # Create a grid for computation
        res = args.resolution
        grid = pv.UniformGrid(
            dimensions=(res, res, res),
            spacing=(2.0/res, 2.0/res, 2.0/res),
            origin=(-1, -1, -1)
        )
        
        # Extract grid coordinates
        x, y, z = grid.points.T
        X, Y, Z = np.reshape(x, (res, res, res)), np.reshape(y, (res, res, res)), np.reshape(z, (res, res, res))
        
        # Compute gyroid field
        params = (self.r_inner, self.r_outer, self.f0, self.f_min, self.alpha, args.buffer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G = parallel_compute(X, Y, Z, params)
        
        # Add scalar field to grid
        grid.point_data["F"] = G.flatten()
        
        # Extract isosurface
        try:
            mesh = grid.contour([self.threshold])
            # Add scalar field for visualization
            mesh.point_data["F"] = np.ones(mesh.n_points)
            return mesh
        except Exception as e:
            print(f"Error generating mesh: {e}")
            return None

# ================================================================
# Advanced Visualization System
# ================================================================

class AdvancedVisualizer:
    def __init__(self, mesh, args):
        self.plotter = pv.Plotter()
        self.mesh = mesh
        self.args = args
        self.setup_visuals()

    def setup_visuals(self):
        """Configure all visualization parameters"""
        # Base mesh properties
        self.plotter.add_mesh(
            self.mesh,
            scalars="F" if not self.args.scalar_field else self.args.scalar_field,
            cmap=self.args.colormap,
            show_edges=self.args.show_edges,
            edge_color=self.args.edge_color,
            line_width=self.args.edge_width,
            smooth_shading=True,
            specular=self.args.specular,
            diffuse=self.args.diffuse,
            ambient=self.args.ambient,
            opacity=self.args.opacity,
            name="gyroid"
        )

        # Interactive controls
        if self.args.clip_dynamic:
            self.add_clipping_planes()
        
        self.add_lighting()
        self.add_sliders()

    def add_clipping_planes(self):
        """Add dynamic clipping plane widgets"""
        def clip_x(normal, origin):
            self.mesh.clip(normal=normal, origin=origin, inplace=True)
        
        for axis in ['x', 'y', 'z']:
            self.plotter.add_plane_widget(
                clip_x,
                normal=axis,
                color='white',
                implicit=False,
                pass_widget=True
            )

    def add_sliders(self):
        """Add interactive control sliders"""
        # Opacity control
        self.plotter.add_slider_widget(
            lambda value: self.plotter.actors["gyroid"].prop.opacity.SetValue(value),
            [0.1, 1.0],
            title='Opacity',
            color='white'
        )

        # Specular power control
        self.plotter.add_slider_widget(
            lambda value: self.plotter.actors["gyroid"].prop.SetSpecularPower(value),
            [0.1, 10.0],
            title='Specular Power',
            color='white'
        )

        # Ambient light control
        self.plotter.add_slider_widget(
            lambda value: self.plotter.actors["gyroid"].prop.SetAmbient(value),
            [0.0, 1.0],
            title='Ambient Light',
            color='white'
        )

    def add_lighting(self):
        """Enhance scene lighting"""
        self.plotter.set_environment_texture()
        self.plotter.add_light(
            position=(10, 10, 10),
            light_type='camera light',
            intensity=0.8
        )

    def show(self):
        """Final render"""
        self.plotter.show()

# ================================================================
# CLI Interface
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced Gyroid Visualizer")
    
    # Core parameters
    parser.add_argument("-r", "--resolution", type=int, default=400)
    parser.add_argument("-b", "--buffer", type=float, default=5.0)
    
    # Visualization parameters
    parser.add_argument("--clip-dynamic", action='store_true',
                      help="Enable dynamic clipping planes")
    parser.add_argument("--colormap", type=str, default="coolwarm",
                      choices=['coolwarm', 'viridis', 'plasma', 'inferno'])
    parser.add_argument("--show-edges", action='store_true',
                      help="Display mesh edges")
    parser.add_argument("--edge-color", type=str, default="black",
                      help="Edge color (name or hex)")
    parser.add_argument("--edge-width", type=float, default=1.0,
                      help="Edge line width")
    parser.add_argument("--specular", type=float, default=0.5,
                      help="Specular lighting coefficient")
    parser.add_argument("--diffuse", type=float, default=0.6,
                      help="Diffuse lighting coefficient")
    parser.add_argument("--ambient", type=float, default=0.2,
                      help="Ambient lighting coefficient")
    parser.add_argument("--opacity", type=float, default=1.0,
                      help="Initial mesh opacity")
    parser.add_argument("--scalar-field", type=str, default=None,
                      help="Scalar field to use for coloring")
    
    args = parser.parse_args()

    # Generate gyroid mesh (existing code)
    generator = GyroidGenerator()
    mesh = generator.generate(args)
    
    if mesh:
        visualizer = AdvancedVisualizer(mesh, args)
        visualizer.show()

if __name__ == "__main__":
    main()