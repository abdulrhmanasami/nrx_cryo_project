#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gc
import numpy as np
import pyvista as pv
from numba import njit, prange

@njit(fastmath=True)
def distance_to_helix(x, y, z, R0, phi0, k_helix, tube_radius):
    theta = (k_helix * z) + phi0
    dx = x - R0 * np.cos(theta)
    dy = y - R0 * np.sin(theta)
    return np.sqrt(dx**2 + dy**2) - tube_radius

@njit(fastmath=True)
def gyroid_value(x, y, z, freq):
    return (np.sin(freq*x)*np.cos(freq*y) 
            + np.sin(freq*y)*np.cos(freq*z) 
            + np.sin(freq*z)*np.cos(freq*x))

@njit(fastmath=True)
def calculate_frequency(r, r_inner, r_outer, f0, f_min, alpha):
    if r <= r_inner:
        return f0
    elif r >= r_outer:
        return f_min
    t = (r - r_inner) / (r_outer - r_inner)
    return f0 * (1 - t**alpha) + f_min * t**alpha

@njit(parallel=True, fastmath=True)
def compute_gyroid_field(X, Y, Z, r_inner, r_outer, f0, f_min, alpha):
    G = np.empty(X.shape, dtype=np.float32)
    for i in prange(X.shape[0]):
        for j in prange(X.shape[1]):
            for k in prange(X.shape[2]):
                x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                r = np.sqrt(x**2 + y**2)
                freq = calculate_frequency(r, r_inner, r_outer, f0, f_min, alpha)
                G[i,j,k] = gyroid_value(x, y, z, freq)
    return G

@njit(parallel=True, fastmath=True)
def compute_helical_channels_field(X, Y, Z, r_inner, r_outer, tube_diameter, tube_turns, num_tubes, rings, height):
    H = np.full(X.shape, 1e9, dtype=np.float32)
    tube_radius = tube_diameter / 2
    ring_radii = np.linspace(r_inner + tube_radius, r_outer - tube_radius, rings)
    k_helix = (2 * np.pi * tube_turns) / height
    tubes_per_ring = num_tubes // rings

    for i in prange(X.shape[0]):
        for j in prange(X.shape[1]):
            for k in prange(X.shape[2]):
                x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                min_dist = 1e9
                for R0 in ring_radii:
                    for t in range(tubes_per_ring):
                        phi0 = (2 * np.pi * t) / tubes_per_ring
                        dist = distance_to_helix(x, y, z, R0, phi0, k_helix, tube_radius)
                        if dist < min_dist:
                            min_dist = dist
                H[i,j,k] = min_dist
    return H

def main():
    parser = argparse.ArgumentParser(description="NRX Cryo-Gyroid with Helical Channels")
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--chunk_size", type=int, default=64)
    args = parser.parse_args()

    # Parameters
    GRID_RES = args.resolution
    WALL_THICKNESS = 0.5
    TUBE_DIAMETER = 4.0
    TUBE_TURNS = 9.5
    NUM_TUBES = 48
    RINGS = 4
    r_inner, r_outer = 77.03, 84.14
    height = 300.0
    f0, f_min, alpha = 0.7022, 0.2, 0.5

    # Generate grid
    x = np.linspace(-r_outer, r_outer, GRID_RES, dtype=np.float32)
    y = x.copy()
    z = np.linspace(0, height, GRID_RES, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Compute fields
    G = compute_gyroid_field(X, Y, Z, r_inner, r_outer, f0, f_min, alpha)
    H = compute_helical_channels_field(X, Y, Z, r_inner, r_outer, TUBE_DIAMETER, TUBE_TURNS, NUM_TUBES, RINGS, height)

    # Combine fields
    F = np.maximum(np.abs(G) - WALL_THICKNESS, -H)  # Correct blending

    # Create mesh
    grid = pv.StructuredGrid(X, Y, Z)
    grid["F"] = F.ravel(order='F')
    mesh = grid.contour([0], scalars="F")

    # Visualize
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', smooth_shading=True)
    plotter.show()

if __name__ == "__main__":
    main()