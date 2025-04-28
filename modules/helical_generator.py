""" 
Industrial-grade helical cooling channels field generator for NRX-Cryo system
Enhanced Version with Continuous Helix Modeling (Optimized)
"""

import numpy as np
import pyvista as pv
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HelicalFieldGenerator:
    """
    Generates 3D scalar distance field for helical cooling channels in cylindrical geometry
    
    Attributes:
        inner_radius (float): Inner radius of cylindrical domain (mm)
        outer_radius (float): Outer radius of cylindrical domain (mm)
        height (float): Total height of cylinder (mm)
        resolution (int): Grid resolution in all dimensions
        pipe_radius (float): Radius of individual cooling channels (mm)
        num_rings (int): Number of concentric radial rings
        pipes_per_ring (int): Number of pipes per radial ring
        pitch (float): Helical pitch (mm per full turn)
        helical_direction (int): Helix winding direction (+1 = right-handed, -1 = left-handed)
    """
    
    def __init__(self,
                 inner_radius: float = 77.03,
                 outer_radius: float = 84.14,
                 height: float = 300.0,
                 resolution: int = 100,
                 pipe_radius: float = 2.0,
                 num_rings: int = 4,
                 pipes_per_ring: int = 12,
                 pitch: float = 31.58,
                 helical_direction: int = 1):
        """
        Initialize generator with NRX-Cryo geometric parameters
        
        Args:
            inner_radius: Inner radius of cylindrical domain (mm)
            outer_radius: Outer radius of cylindrical domain (mm)
            height: Total height of cylinder (mm)
            resolution: Grid resolution in all dimensions
            pipe_radius: Radius of cooling channels (mm)
            num_rings: Number of concentric radial rings
            pipes_per_ring: Number of pipes per radial ring
            pitch: Helical pitch (mm)
            helical_direction: Helix winding direction (+1/-1)
            
        Raises:
            ValueError: For invalid geometric parameters
        """
        
        # ==========================================
        # Section 1: Enhanced Parameter Validation
        # ==========================================
        logger.info("Initializing HelicalFieldGenerator")
        
        # Core parameter validation
        if not (inner_radius < outer_radius):
            raise ValueError("Inner radius must be smaller than outer radius")
        if num_rings <= 0 or pipes_per_ring <= 0:
            raise ValueError("Number of rings and pipes must be positive")
        if helical_direction not in (1, -1):
            raise ValueError("Helical direction must be +1 or -1")
        
        # Advanced geometric validation
        radial_spacing = (outer_radius - inner_radius) / (num_rings + 1)
        if radial_spacing < 2 * pipe_radius:
            raise ValueError(f"Radial spacing ({radial_spacing:.2f} mm) insufficient for pipe radius {pipe_radius} mm")
        
        if pitch < 2 * pipe_radius:
            raise ValueError(f"Pitch ({pitch} mm) must be ≥ 2×pipe radius ({2*pipe_radius} mm)")
        
        # Store parameters
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.height = height
        self.resolution = resolution
        self.pipe_radius = pipe_radius
        self.num_rings = num_rings
        self.pipes_per_ring = pipes_per_ring
        self.pitch = pitch
        self.helical_direction = helical_direction
        
        # Precompute radial positions and angular offsets
        self.radial_positions = np.linspace(inner_radius, outer_radius, num_rings+2, dtype=np.float32)[1:-1]
        self.angular_offsets = np.linspace(0, 2*np.pi, pipes_per_ring, endpoint=False, dtype=np.float32)
        
        # Generate coordinate grids
        self._initialize_grids()
        
        logger.info("Generator initialized with %d rings and %d total pipes", 
                   num_rings, num_rings*pipes_per_ring)

    def _initialize_grids(self):
        """Initialize 3D coordinate grids using vectorized operations"""
        # Cylindrical coordinates with memory optimization
        r = np.linspace(self.inner_radius, self.outer_radius, self.resolution, dtype=np.float32)
        z = np.linspace(0, self.height, self.resolution, dtype=np.float32)
        
        # Create 3D grid using broadcasting
        R, Z = np.meshgrid(r, z, indexing='ij')
        Theta = self.helical_direction * (Z / self.pitch) * 2 * np.pi
        
        # Convert to Cartesian coordinates
        self.X = R * np.cos(Theta)
        self.Y = R * np.sin(Theta)
        self.Z = Z

    def _compute_helix_distance(self, r_center: float) -> np.ndarray:
        """
        Vectorized distance computation for a complete radial ring
        
        Mathematical Basis:
        For each point (X, Y, Z) in grid:
        1. Calculate angular offset for each pipe: θ_pipe = angular_offset + (2πZ/pitch)*direction
        2. Compute ideal pipe position: (r_center*cos(θ_pipe), r_center*sin(θ_pipe), Z)
        3. Find minimum Euclidean distance to any pipe in the ring
        
        Args:
            r_center: Radial position of the ring center (mm)
            
        Returns:
            3D array of minimum distances for all pipes in ring (mm)
        """
        # Expand dimensions for vectorized operations
        theta_offsets = self.angular_offsets.reshape(1, 1, -1)
        
        # Calculate helix trajectories for all pipes
        helix_theta = theta_offsets + (self.Z[..., None] / self.pitch) * 2 * np.pi * self.helical_direction
        
        # Calculate pipe coordinates
        x_pipes = r_center * np.cos(helix_theta)
        y_pipes = r_center * np.sin(helix_theta)
        
        # Optimized distance calculation using hypot
        dx = self.X[..., None] - x_pipes
        dy = self.Y[..., None] - y_pipes
        distances = np.hypot(dx, dy)  # Optimized replacement for sqrt(dx² + dy²)
        
        # Find minimum distance across all pipes
        return np.min(distances, axis=3)

    def generate_field(self) -> np.ndarray:
        """
        Optimized field generation using vectorized operations
        
        Returns:
            3D numpy array of signed distance values (mm)
        """
        logger.info("Starting field generation...")
        
        # Initialize distance field with memory optimization
        min_distances = np.full_like(self.X, np.inf, dtype=np.float32)
        
        # Process each radial ring
        for r_center in self.radial_positions:
            ring_dist = self._compute_helix_distance(r_center)
            np.minimum(min_distances, ring_dist, out=min_distances)
        
        # Apply pipe radius threshold
        signed_field = min_distances - self.pipe_radius
        
        logger.info("Field generation completed")
        return signed_field

    def generate_structured_grid(self, field: np.ndarray) -> pv.StructuredGrid:
        """
        Create optimized PyVista StructuredGrid with corrected indexing
        
        Args:
            field: 3D numpy array of scalar values
            
        Returns:
            PyVista StructuredGrid with helical field
        """
        logger.info("Creating structured grid...")
        
        # Ensure grid dimensions match field shape
        if field.shape != self.X.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid dimensions {self.X.shape}")
        
        grid = pv.StructuredGrid(self.X, self.Y, self.Z)
        grid["helical_field"] = field.ravel(order='F')
        
        logger.info("Grid created with %d points", grid.n_points)
        return grid

# ==========================================
# Validation and Example Usage
# ==========================================
if __name__ == "__main__":
    # NRX-Cryo standard parameters with helical direction
    params = {
        'inner_radius': 77.03,
        'outer_radius': 84.14,
        'height': 300.0,
        'resolution': 100,
        'pipe_radius': 2.0,
        'num_rings': 4,
        'pipes_per_ring': 12,
        'pitch': 31.58,
        'helical_direction': -1  # Left-handed helix
    }
    
    try:
        # Initialize generator
        hfg = HelicalFieldGenerator(**params)
        
        # Generate field
        field = hfg.generate_field()
        
        # Create and save grid
        grid = hfg.generate_structured_grid(field)
        grid.save("nrx_helical_field_optimized.vti")
        
        print("Field generation successful!")
        print(f"Field range: [{np.min(field):.2f}, {np.max(field):.2f}] mm")
        print(f"Negative values indicate pipe interior")
        
    except Exception as e:
        logger.error("Generation failed: %s", str(e))