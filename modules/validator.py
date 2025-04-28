"""
NRX-Cryo Industrial Field Validator v1.1
========================================
Enhanced Quality Assurance Module for Additive Manufacturing
Industrial Upgrades:
- Dimension validation during grid construction
- Precise gradient calculation for irregular grids
- Adaptive thickness sampling
- Robust pipe spacing validation
- Unit standardization
- Advanced error handling
"""

import numpy as np
import pyvista as pv
from datetime import datetime
import logging
from typing import Dict, Tuple, List
from scipy.spatial import cKDTree
import multiprocessing as mp

# Configure industrial logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NRX-Validator')

class ValidationError(Exception):
    """Critical validation failure exception"""
    pass

class FieldValidator:
    """
    Industrial-grade field validation system for NRX-Cryo components v1.1
    
    Attributes:
        threshold_config (dict): Validation thresholds with unit specifications
        status_codes (dict): Smart alert level definitions
    """
    
    threshold_config = {
        'min_wall_thickness': 0.285,  # mm (0.3mm - 5%)
        'max_wall_thickness': 0.315,  # mm (0.3mm + 5%)
        'critical_pipe_spacing': 4.0,  # mm
        'gradient_tolerance': 1e5      # Pa/m (pressure gradient)
    }
    
    status_codes = {
        'INFO': 0,
        'WARNING': 1,
        'ERROR': 2
    }

    def __init__(self, field: np.ndarray, x: np.ndarray, 
                 y: np.ndarray, z: np.ndarray):
        """
        Initialize validator with validated field data and coordinates
        
        Args:
            field: 3D scalar field array (dimensions: Nx×Ny×Nz)
            x: 1D/3D coordinate array (meshgrid compatible)
            y: 1D/3D coordinate array
            z: 1D/3D coordinate array
            
        Raises:
            ValidationError: If grid construction fails or dimension mismatch
        """
        self.field = field
        self._validate_input_dimensions(x, y, z)
        self._build_structured_grid(x, y, z)
        self.results = {}
        self.alerts = []

    def _validate_input_dimensions(self, x, y, z) -> None:
        """Validate coordinate dimensions against field shape"""
        expected_shape = self.field.shape
        try:
            if x.ndim == 1:
                assert len(x) == expected_shape[0]
                assert len(y) == expected_shape[1]
                assert len(z) == expected_shape[2]
            else:
                assert x.shape == expected_shape
                assert y.shape == expected_shape
                assert z.shape == expected_shape
        except AssertionError:
            raise ValidationError(f"Dimension mismatch: Field {expected_shape} "
                                  f"vs Coordinates ({x.shape}, {y.shape}, {z.shape})")

    def _build_structured_grid(self, x, y, z) -> None:
        """Construct validated PyVista grid with error handling"""
        try:
            if x.ndim == 1:
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            else:
                X, Y, Z = x, y, z
                
            self.grid = pv.StructuredGrid(X, Y, Z)
            self.grid["field"] = self.field.ravel(order='F')
            
            # Validate grid topology
            if self.grid.n_cells != (self.field.size - 
                                   np.prod(np.array(self.field.shape) - 1)):
                raise ValidationError("Grid-cell count mismatch with field data")
                
        except Exception as e:
            self._add_alert(f"Grid construction failed: {str(e)}", 'ERROR')
            raise ValidationError(f"Grid validation failed: {str(e)}") from e

    def check_surface_integrity(self) -> Dict:
        """
        Enhanced surface topology validation with feature edge detection
        
        Returns:
            Dict with defect counts and alert status
        """
        try:
            surface = self.grid.contour([0])
            feature_edges = surface.extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=True,
                feature_edges=False
            )
            
            defects = {
                'non_manifold': feature_edges.n_lines,
                'self_intersections': surface.check_self_intersection(),
                'degenerate_cells': surface.n_cells - surface.n_faces
            }
            
            status = self.status_codes['INFO']
            for defect, count in defects.items():
                if count > 0:
                    status = max(status, self.status_codes['ERROR'])
                    self._add_alert(f"Surface defect: {defect} ({count} instances)", 
                                  'ERROR')

            self.results['surface_integrity'] = {
                'defects': defects,
                'status': status
            }
            return self.results['surface_integrity']
            
        except Exception as e:
            self._add_alert(f"Surface validation failed: {str(e)}", 'ERROR')
            raise ValidationError("Surface integrity check failed") from e

    def check_field_continuity(self) -> Dict:
        """
        Precise gradient calculation using PyVista's derivative functions
        for irregular grids
        
        Returns:
            Dict with max gradient and discontinuity locations
        """
        try:
            # Compute gradient using PyVista's built-in method
            grad = self.grid.compute_derivative(scalars="field")
            grad_mag = np.linalg.norm(grad["gradient"], axis=1)
            max_grad = np.max(grad_mag)
            
            status = self.status_codes['INFO']
            if max_grad > self.threshold_config['gradient_tolerance']:
                status = self.status_codes['ERROR']
                self._add_alert(f"Field discontinuity detected (Max gradient: {max_grad:.2e} Pa/m)",
                              'ERROR')

            self.results['field_continuity'] = {
                'max_gradient': max_grad,
                'status': status
            }
            return self.results['field_continuity']
            
        except Exception as e:
            self._add_alert(f"Gradient calculation failed: {str(e)}", 'ERROR')
            raise ValidationError("Field continuity check failed") from e

    def _thickness_worker(self, points):
        """Parallel worker for thickness calculation"""
        return self.grid.threshold(0).compute_thickness(sample_points=points)

    def check_wall_thickness(self, samples: int = None) -> Dict:
        """
        Adaptive statistical wall thickness validation
        - Auto-scaling samples based on surface complexity
        - Parallel computation support
        
        Args:
            samples: Optional manual sample count
            
        Returns:
            Dict with thickness statistics and compliance status
        """
        try:
            surface = self.grid.contour([0])
            
            # Adaptive sampling logic
            if not samples:
                base_samples = 5000
                complexity_factor = surface.n_faces / 1e4  # Normalize
                samples = int(base_samples * max(1, complexity_factor))
                
            # Parallel execution
            with mp.Pool() as pool:
                result = pool.map(self._thickness_worker, [samples//4]*4)
                thickness = pv.MultiBlock(result).combine()

            valid_thickness = thickness.thickness[
                (thickness.thickness > self.threshold_config['min_wall_thickness']) &
                (thickness.thickness < self.threshold_config['max_wall_thickness'])
            ]
            
            compliance = len(valid_thickness)/len(thickness.thickness)
            avg_thickness = np.mean(thickness.thickness)
            
            status = self.status_codes['INFO']
            if compliance < 0.95:
                status = self.status_codes['ERROR']
                self._add_alert(f"Wall thickness non-compliant ({compliance*100:.1f}% in spec)",
                              'ERROR')
            elif compliance < 0.99:
                status = self.status_codes['WARNING']
                self._add_alert(f"Marginal wall thickness compliance ({compliance*100:.1f}%)",
                              'WARNING')

            self.results['wall_thickness'] = {
                'average': avg_thickness,
                'compliance': compliance,
                'samples_used': samples,
                'status': status
            }
            return self.results['wall_thickness']
            
        except Exception as e:
            self._add_alert(f"Thickness validation failed: {str(e)}", 'ERROR')
            raise ValidationError("Wall thickness check failed") from e

    def check_pipe_spacing(self) -> Dict:
        """
        Robust pipe spacing validation with empty cluster handling
        
        Returns:
            Dict with minimum spacing and violation locations
        """
        try:
            positive_regions = self.grid.threshold(0.0, scalars="field")
            if positive_regions.n_cells == 0:
                self._add_alert("No pipe structures detected", 'WARNING')
                return {
                    'min_distance': np.inf,
                    'status': self.status_codes['INFO']
                }

            clusters = positive_regions.connectivity(largest=False)
            centers = [cluster.center for cluster in clusters.split_bodies()]
            
            if len(centers) < 2:
                self._add_alert("Insufficient pipes for spacing validation", 'INFO')
                return {
                    'min_distance': np.inf,
                    'status': self.status_codes['INFO']
                }

            tree = cKDTree(centers)
            distances, _ = tree.query(centers, k=2)
            min_distance = np.min(distances[:,1])  # Skip self-distance
            
            status = self.status_codes['INFO']
            if min_distance < self.threshold_config['critical_pipe_spacing']:
                status = self.status_codes['ERROR']
                self._add_alert(f"Pipe spacing violation ({min_distance:.2f} mm)",
                              'ERROR')

            self.results['pipe_spacing'] = {
                'min_distance': min_distance,
                'status': status
            }
            return self.results['pipe_spacing']
            
        except Exception as e:
            self._add_alert(f"Pipe spacing validation failed: {str(e)}", 'ERROR')
            raise ValidationError("Pipe spacing check failed") from e

    def generate_validation_report(self, filename: str = "validation_report.txt") -> None:
        """
        Enhanced QA report with unit annotations
        
        Args:
            filename: Output report path
        """
        report = [
            "NRX-Cryo Quality Assurance Report v1.1",
            "=====================================",
            f"Validation timestamp: {datetime.now().isoformat()}",
            f"Grid dimensions: {self.grid.dimensions}",
            ""
        ]
        
        # Results summary
        for test, data in self.results.items():
            status = '✅ PASS' if data['status'] < self.status_codes['WARNING'] else '❌ FAIL'
            report.append(f"{test.replace('_', ' ').title():<25} {status}")
        
        # Detailed findings
        report.extend(["", "Detailed Findings:", "-----------------"])
        unit_map = {
            'average': 'mm',
            'min_distance': 'mm',
            'max_gradient': 'Pa/m',
            'compliance': '%'
        }
        
        for test, data in self.results.items():
            report.append(f"\n* {test.replace('_', ' ').title()}:")
            for k, v in data.items():
                if k == 'status': continue
                unit = unit_map.get(k, '')
                report.append(f"  - {k.replace('_', ' ')}: {v}{' ' + unit if unit else ''}")

        # Alert summary
        report.extend(["", "Alerts:", "-------"])
        for alert in self.alerts:
            report.append(f"[{alert['level']}] {alert['message']}")
        
        # Write to file
        with open(filename, 'w') as f:
            f.write('\n'.join(report))

    def _add_alert(self, message: str, level: str = 'INFO') -> None:
        """Enhanced alerting system with context capture"""
        self.alerts.append({
            'timestamp': datetime.now().isoformat(),
            'level': level.upper(),
            'message': message,
            'grid_state': f"{self.grid.dimensions if self.grid else 'No grid'}"
        })
        logger.log(getattr(logging, level.upper()), f"{message} | Grid: {self.grid.dimensions if self.grid else 'N/A'}")

    def partial_validation(self, tests: List[str]) -> Dict:
        """
        Enhanced partial validation with safety wrapper
        
        Args:
            tests: List of tests to run ['surface', 'walls', 'pipes', 'continuity']
            
        Returns:
            Partial validation results with error handling
        """
        test_map = {
            'surface': self.check_surface_integrity,
            'walls': self.check_wall_thickness,
            'pipes': self.check_pipe_spacing,
            'continuity': self.check_field_continuity
        }
        
        partial_results = {}
        for test in tests:
            try:
                if test in test_map:
                    partial_results[test] = test_map[test]()
                else:
                    self._add_alert(f"Invalid test requested: {test}", 'WARNING')
            except Exception as e:
                self._add_alert(f"Partial validation failed for {test}: {str(e)}", 'ERROR')
                partial_results[test] = {'status': self.status_codes['ERROR']}
                
        return partial_results

# ----------------------------------------------------------
# Industrial Enhancements v1.1:
# 1. Dimension validation during grid initialization
# 2. Precise gradient calculation using PyVista's derivatives
# 3. Adaptive sampling with parallel computation
# 4. Robust empty-case handling in pipe spacing check
# 5. Unit annotations in reports and config
# 6. Context-aware alerting system
# 7. Safety wrappers for all validation methods
# ----------------------------------------------------------