"""
NRX-Cryo Industrial Field Exporter v3.1
=======================================
Enhanced Module with Dynamic Chunking and Adaptive Quality Control
"""

import numpy as np
import pyvista as pv
from typing import Optional, Generator, Tuple, Union
import warnings
from datetime import datetime
import logging
import psutil

# Configure industrial-grade logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NRX-Exporter')

class FieldExporter:
    """
    Industrial 3D field exporter with dynamic chunk processing and enhanced quality control
    
    Attributes:
        BASE_CHUNK (tuple): Base chunk dimensions (64,64,64)
        MEMORY_SAFETY_FACTOR (float): Memory buffer ratio (0.8)
    """
    
    BASE_CHUNK = (64, 64, 64)
    MEMORY_SAFETY_FACTOR = 0.8
    
    def __init__(self, field: np.ndarray, x: np.ndarray, 
                y: np.ndarray, z: np.ndarray):
        """
        Initialize exporter with dynamic memory optimization
        
        Args:
            field: 3D scalar field (float32 recommended)
            x: 1D/3D coordinate array
            y: 1D/3D coordinate array
            z: 1D/3D coordinate array
            
        Raises:
            NRXExportError: For critical dimensional mismatches
        """
        # Enhanced validation pipeline
        self._validate_topology(field, x, y, z)
        
        # Initialize metadata system
        self.metadata = {
            'nrx_version': '3.1',
            'build_date': datetime.now().isoformat(),
            'quality_class': 'A1',
            'chunked': False,
            'global_metadata': True
        }
        
        # Dynamic chunk configuration
        self.chunk_size = self._calculate_dynamic_chunk(field.nbytes)
        self.grid = self._build_optimized_grid(field, x, y, z)

    def _validate_topology(self, field: np.ndarray, 
                          x: np.ndarray, y: np.ndarray, 
                          z: np.ndarray) -> None:
        """Enhanced NRX-Cryo structural validation"""
        # Check for 3D coordinate arrays
        if x.ndim == 3:
            if field.shape != x.shape:
                raise NRXExportError("3D coordinate-field shape mismatch")
        else:
            if field.shape != (x.size, y.size, z.size):
                raise NRXExportError("1D coordinate-field dimension mismatch")
                
        if field.dtype != np.float32:
            warnings.warn("Using non-optimized dtype, recommend float32", 
                         PerformanceWarning)

    def _calculate_dynamic_chunk(self, field_size: int) -> Tuple[int, int, int]:
        """Calculate chunk size based on system memory"""
        available_mem = psutil.virtual_memory().available * self.MEMORY_SAFETY_FACTOR
        if field_size < available_mem:
            self.metadata['chunked'] = False
            return self.BASE_CHUNK
        
        # Dynamic chunk calculation
        chunk_ratio = (available_mem / field_size) ** (1/3)
        return tuple(
            max(16, int(self.BASE_CHUNK[i] * chunk_ratio)) 
            for i in range(3)
        )

    def _build_optimized_grid(self, field: np.ndarray,
                             x: np.ndarray, y: np.ndarray,
                             z: np.ndarray) -> Union[pv.StructuredGrid, pv.MultiBlock]:
        """Smart grid construction with dynamic chunking"""
        if self.metadata['chunked']:
            return self._build_safe_chunked_grid(field, x, y, z)
            
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid = pv.StructuredGrid(X, Y, Z)
        grid["merged_field"] = field.ravel(order='F')
        self._add_global_metadata(grid)
        return grid

    def _build_safe_chunked_grid(self, field: np.ndarray,
                                x: np.ndarray, y: np.ndarray,
                                z: np.ndarray) -> pv.MultiBlock:
        """Memory-safe chunked construction using MultiBlock"""
        chunks = self._generate_chunks(field.shape)
        multi_grid = pv.MultiBlock()
        
        for chunk_idx in chunks:
            chunk_data = field[
                chunk_idx[0]:chunk_idx[1],
                chunk_idx[2]:chunk_idx[3],
                chunk_idx[4]:chunk_idx[5]
            ]
            
            chunk_grid = pv.StructuredGrid(
                *np.meshgrid(
                    x[chunk_idx[0]:chunk_idx[1]],
                    y[chunk_idx[2]:chunk_idx[3]],
                    z[chunk_idx[4]:chunk_idx[5]],
                    indexing='ij'
                )
            )
            chunk_grid["merged_field"] = chunk_data.ravel(order='F')
            multi_grid.append(chunk_grid)
            
        self._add_global_metadata(multi_grid)
        return multi_grid

    def _generate_chunks(self, shape: Tuple[int,int,int]) -> Generator:
        """Generate chunk indices with dynamic sizing"""
        for i in range(0, shape[0], self.chunk_size[0]):
            for j in range(0, shape[1], self.chunk_size[1]):
                for k in range(0, shape[2], self.chunk_size[2]):
                    yield (
                        i, min(i+self.chunk_size[0], shape[0]),
                        j, min(j+self.chunk_size[1], shape[1]),
                        k, min(k+self.chunk_size[2], shape[2])
                    )

    def _add_global_metadata(self, grid: Union[pv.DataSet, pv.MultiBlock]) -> None:
        """Centralized metadata management"""
        if isinstance(grid, pv.MultiBlock):
            grid.field_data.update(self.metadata)
        else:
            for key, value in self.metadata.items():
                grid.field_data[key] = [value]

    def export_stl(self, filename: str, 
                  isovalue: float = 0.0,
                  resolution: int = 3,
                  binary: bool = True) -> None:
        """
        Enhanced STL export with adaptive isosurfacing
        
        Args:
            filename: Output path (.stl)
            isovalue: Surface extraction value
            resolution: Quality preset (1-5)
            binary: Binary format (recommended)
        """
        surface = self.extract_surface(isovalue, resolution)
        
        if self.metadata['chunked']:
            self._safe_chunked_export(surface, filename, 'stl')
        else:
            surface.save(filename, binary=binary)

    def export_vtu(self, filename: str,
                  compression: int = 5) -> None:
        """
        Optimized VTU export with unified metadata
        
        Args:
            filename: Output path (.vtu)
            compression: Zlib level (1-9)
        """
        if self.metadata['chunked']:
            self._safe_chunked_export(self.grid, filename, 'vtu', compression)
        else:
            self.grid.save(filename, compression=compression)

    def _safe_chunked_export(self, data: pv.DataSet,
                            filename: str,
                            format: str,
                            compression: Optional[int] = None) -> None:
        """Enhanced chunked export with metadata preservation"""
        writer = data.GetWriter(format)
        writer.SetFileName(filename)
        
        if format == 'vtu' and compression:
            writer.SetCompressorTypeToZLib()
            writer.SetCompressionLevel(compression)
            
        writer.Start()
        if isinstance(data, pv.MultiBlock):
            for block in data:
                writer.WriteBlock(block)
        else:
            for chunk in data.partition(self.chunk_size):
                writer.WriteChunk(chunk)
        writer.End()

    def extract_surface(self, isovalue: float = 0.0,
                       resolution: int = 3) -> pv.PolyData:
        """
        Generate optimized surface with quality enhancements
        
        Args:
            isovalue: Contour value for surface extraction
            resolution: Surface quality level (1-5)
            
        Returns:
            Processed PolyData surface
        """
        if not self._validate_isovalue(isovalue):
            raise NRXExportError("Invalid isovalue for field range")
            
        surface = self.grid.contour([isovalue])
        surface = surface.clean()
        
        self._detect_surface_defects(surface)
        
        # Adaptive quality processing
        if resolution > 1:
            surface = surface.subdivide(resolution - 1)
            surface = surface.smooth(n_iter=resolution*2)
            
        return surface

    def _validate_isovalue(self, isovalue: float) -> bool:
        """Ensure isovalue within field range"""
        return np.min(self.grid["merged_field"]) <= isovalue <= np.max(self.grid["merged_field"])

    def _detect_surface_defects(self, mesh: pv.PolyData) -> None:
        """Enhanced defect detection"""
        defects = {
            'non_manifold': mesh.n_open_edges,
            'self_intersections': mesh.check_self_intersection(),
            'degenerate_cells': mesh.check_valid_cells().sum()
        }
        
        for defect, count in defects.items():
            if count > 0:
                logger.warning(f"Surface defect detected: {defect} ({count} instances)")
                
        self.metadata.update({'defects': defects})

class NRXExportError(Exception):
    """Critical export violation"""
    pass

class PerformanceWarning(UserWarning):
    """Non-optimal performance warning"""
    pass

# ----------------------------------------------------------
# Industrial Enhancements in v3.1:
# 1. Dynamic chunk sizing based on available memory
# 2. MultiBlock-based chunk management
# 3. Adaptive isovalue validation
# 4. Enhanced defect detection with valid cells check
# 5. Surface smoothing and subdivision
# 6. Unified metadata management system
# 7. Optimized memory safety factor
# ----------------------------------------------------------