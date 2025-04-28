"""
NRX-Cryo Gyroid Field Generator v2.1
====================================
Industrial-Grade Implementation with Numba Acceleration
**تم التحديث وفق التوصيات الهندسية**
"""

import math
import numpy as np
import pyvista as pv
from numba import njit, prange
from typing import NamedTuple

# ==========================================
# Section 1: Numba-Compatible Data Structure
# ==========================================

class GyroidParameters(NamedTuple):
    """Numba-compatible parameter container"""
    inner_radius: float
    outer_radius: float
    height: float
    resolution: int
    wall_thickness: float
    frequency_exponent: float
    cell_size: float
    f0: float
    f_min: float

# ==========================================
# Section 2: Numba Accelerated Core Functions (Corrected)
# ==========================================

@njit(fastmath=True)
def numba_calculate_gyroid(x: float, y: float, z: float, p: GyroidParameters) -> float:
    """Numba-optimized gyroid surface calculation with corrected radial gradient"""
    r = math.hypot(x, y)
    
    # التحقق من الحدود الشعاعية والارتفاع
    if not (p.inner_radius <= r <= p.outer_radius) or z < 0 or z > p.height:
        return 0.0
    
    # حساب التدرج الترددي الصحيح حسب المواصفات
    normalized = (r - p.inner_radius) / (p.outer_radius - p.inner_radius)
    freq = p.f_min + (p.f0 - p.f_min) * math.pow(normalized, p.frequency_exponent)
    
    theta = math.atan2(y, x)
    
    # المعادلة المعدلة مع تطبيق القيمة المطلقة
    t1 = math.sin(freq * r * math.cos(theta)) * math.cos(freq * r * math.sin(theta))
    t2 = math.sin(freq * r * math.sin(theta)) * math.cos(freq * z)
    t3 = math.sin(freq * z) * math.cos(freq * r * math.cos(theta))
    
    return abs(t1 + t2 + t3) - p.wall_thickness  # تطبيق سماكة الجدار بشكل صحيح

@njit(parallel=True, fastmath=True)
def numba_generate_field(p: GyroidParameters, r_space: np.ndarray, 
                        theta_space: np.ndarray, z_space: np.ndarray) -> np.ndarray:
    """Numba-optimized parallel field generation with cylindrical grid"""
    field = np.zeros((p.resolution, p.resolution, p.resolution), dtype=np.float64)  # تحسين الدقة
    
    for i in prange(p.resolution):
        z = z_space[i]
        for j in prange(p.resolution):
            theta = theta_space[j]
            for k in prange(p.resolution):
                r = r_space[k]
                # تحويل إحداثيات أسطوانية إلى ديكارتية
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                field[i, j, k] = numba_calculate_gyroid(x, y, z, p)
    
    return field

# ==========================================
# Section 3: Main Generator Class (Optimized)
# ==========================================

class GyroidFieldGenerator:
    """Industrial Gyroid Generator with Corrected Cylindrical Grid"""
    
    def __init__(self, params: GyroidParameters):
        self.params = params
        self._precompute_cylindrical_grid()  # إنشاء شبكة أسطوانية أولاً
    
    def _precompute_cylindrical_grid(self):
        """إنشاء شبكة إحداثيات أسطوانية"""
        # محاور إحداثيات أسطوانية
        self.r_space = np.linspace(self.params.inner_radius, 
                                  self.params.outer_radius, 
                                  self.params.resolution,
                                  dtype=np.float64)
        
        self.theta_space = np.linspace(0, 2*np.pi, 
                                      self.params.resolution,
                                      dtype=np.float64)
        
        self.z_space = np.linspace(0, self.params.height,
                                  self.params.resolution,
                                  dtype=np.float64)
        
        # إنشاء شبكة ثلاثية الأبعاد
        R, Theta, Z = np.meshgrid(self.r_space, self.theta_space, self.z_space, indexing='ij')
        
        # تحويل إلى إحداثيات ديكارتية
        self.X = R * np.cos(Theta)
        self.Y = R * np.sin(Theta)
        self.Z = Z

    def generate_field(self) -> np.ndarray:
        """Generate optimized 3D scalar field"""
        return numba_generate_field(self.params, 
                                   self.r_space, 
                                   self.theta_space, 
                                   self.z_space)

    def generate_structured_grid(self) -> pv.StructuredGrid:
        """Create visualization-ready structured grid"""
        field = self.generate_field()
        grid = pv.StructuredGrid(self.X, self.Y, self.Z)
        grid["gyroid_field"] = field.ravel(order='F')
        return grid

# ==========================================
# Section 4: Factory Functions (Updated)
# ==========================================

def create_official_generator(resolution: int = 300) -> GyroidFieldGenerator:
    """Create production-ready generator instance with validated parameters"""
    cell_size = 7.5
    f0 = (2 * math.pi / cell_size) * 0.8378
    f_min = f0 * 0.1
    
    params = GyroidParameters(
        inner_radius=77.03,
        outer_radius=84.14,
        height=300.0,
        resolution=resolution,
        wall_thickness=0.3,
        frequency_exponent=0.3,
        cell_size=cell_size,
        f0=f0,
        f_min=f_min
    )
    
    return GyroidFieldGenerator(params)

# ==========================================
# التحديثات الرئيسية:
# 1. تصحيح التدرج الترددي الشعاعي وفق المواصفات
# 2. إعادة بناء الشبكة باستخدام إحداثيات أسطوانية
# 3. تحسين دقة الحساب باستخدام float64
# 4. تطبيق سماكة الجدار عبر القيمة المطلقة
# 5. إضافة تحقق من حدود الارتفاع (z)
# ==========================================