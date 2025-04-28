# 1. Overall Geometry and Core Structure

## 1.1 3D Cylindrical Layout

The NRX-Cryo Cylindrical Heat Exchanger is designed as a cylindrical structure with an embedded TPMS-based gyroid architecture. The gyroid structure provides an optimal balance between surface area, flow characteristics, and mechanical strength, making it ideal for cryogenic heat exchange applications.

The cylindrical geometry allows for efficient integration with existing propulsion systems while maximizing the heat transfer surface area within a compact volume. The gyroid architecture creates a continuous, self-supporting network of channels that enables efficient heat transfer between the solid structure and the fluid medium.

## 1.2 Final Confirmed Dimensions

| Parameter | Value | Unit |
|-----------|-------|------|
| Inner Radius | 77.03 | mm |
| Outer Radius | 84.14 | mm |
| Wall Thickness | 7.11 | mm |
| Total Height/Length | 300.0 | mm |
| Total Volume | 1,324,212 | mm³ |
| Gyroid Cell Size | 7.5 | mm |
| Gyroid Wall Thickness | 0.3 | mm |

## 1.3 Structural Boundaries

The heat exchanger is bounded by:

- Inner cylindrical boundary at r = 77.03 mm
- Outer cylindrical boundary at r = 84.14 mm
- Bottom planar boundary at z = 0 mm
- Top planar boundary at z = 300.0 mm

A smooth transition region is implemented at all boundaries to ensure manufacturability and structural integrity. The transition width is set to 1.0 mm from each boundary, creating a gradual blend between the gyroid structure and the boundary surfaces.

## 1.4 Gyroid Architecture

The core structure utilizes a gyroid-based Triply Periodic Minimal Surface (TPMS) architecture. The gyroid is a minimal surface with triple periodicity that divides space into two intertwined but separate labyrinths.

Key characteristics of the gyroid architecture:

- Continuous, self-supporting structure without intersections
- High surface-to-volume ratio for efficient heat transfer
- Smooth curvature that minimizes flow resistance
- Balanced mechanical properties in all directions
- Manufacturability via additive manufacturing techniques

The gyroid structure is mathematically defined using an implicit function where the surface is represented by the isosurface G(x,y,z) = 0. The solid region is defined by G(x,y,z) < 0, and the fluid region by G(x,y,z) > 0.

## 1.5 Frequency Gradient

A key innovation in this design is the implementation of a spatially varying frequency gradient that optimizes the gyroid cell size based on radial position. The frequency varies according to a power law with exponent α = 0.3:

f(r) = f_min + (f_0 - f_min) × ((R_outer - r)/(R_outer - R_inner))^α

Where:
- f_0 = 2π/cell_size × 0.8378 = 0.7019 rad/mm (reference frequency)
- f_min = f_0 × 0.1 = 0.0702 rad/mm (minimum frequency)
- α = 0.3 (frequency gradient exponent)

This gradient creates smaller cells near the inner radius and larger cells near the outer radius, optimizing flow characteristics and heat transfer efficiency throughout the structure.
