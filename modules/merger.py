"""
NRX-Cryo Field Merger v3.0
==========================
الإصدار النهائي مع طبقة الانتقال والشبكة الأسطوانية
"""

import math
import numpy as np
import pyvista as pv
from numba import njit, prange
from typing import NamedTuple

# ==========================================
# البنية الأساسية للمعاملات (محدثة)
# ==========================================

class FieldParameters(NamedTuple):
    """حاوية معاملات متوافقة مع Numba"""
    inner_radius: float    # نصف القطر الداخلي (مم)
    outer_radius: float    # نصف القطر الخارجي (مم)
    height: float          # الارتفاع الكلي (مم)
    resolution: int        # دقة الشبكة (نقطة/محور)
    wall_thickness: float  # سماكة الجدار (مم)
    gyroid_frequency: float  # تردد الجايرويد (راديان/مم)
    helical_frequency: float # تردد الحلزوني (راديان/مم)
    cell_size: float         # حجم الخلية الأساسي (مم)
    phase_shift_factor: float # معامل تحويل الطور
    frequency_exponent: float # أس التردد
    transition_width: float = 1.0  # عرض طبقة الانتقال (مم)

# ==========================================
# الدوال الأساسية المعجلة (محدثة)
# ==========================================

@njit(fastmath=True)
def calculate_transition_layer(r: float, params: FieldParameters) -> float:
    """حساب معامل الانتقال B(r)"""
    d_inner = r - params.inner_radius
    d_outer = params.outer_radius - r
    d_transition = params.transition_width
    
    inner_factor = 0.5 * (1 + math.tanh(d_inner / d_transition))
    outer_factor = 0.5 * (1 + math.tanh(d_outer / d_transition))
    return inner_factor * outer_factor

@njit(fastmath=True)
def gyroid_function(x: float, y: float, z: float, params: FieldParameters) -> float:
    r = math.hypot(x, y)
    if not (params.inner_radius <= r <= params.outer_radius):
        return 0.0
    
    theta = math.atan2(y, x)
    radial_factor = (params.outer_radius - r) / (params.outer_radius - params.inner_radius)
    freq = params.gyroid_frequency * (1 + math.pow(radial_factor, params.frequency_exponent))
    
    return (math.sin(freq * x) * math.cos(freq * y) +
            math.sin(freq * y) * math.cos(freq * z) +
            math.sin(freq * z) * math.cos(freq * x)) - params.wall_thickness

@njit(fastmath=True)
def helical_function(x: float, y: float, z: float, params: FieldParameters) -> float:
    r = math.hypot(x, y)
    if r < params.inner_radius or r > params.outer_radius:
        return 0.0
    
    phase_shift = params.phase_shift_factor * (z / params.height)
    return math.sin(params.helical_frequency * (x + z * phase_shift))

@njit(parallel=True)
def generate_gyroid(x_space: np.ndarray, y_space: np.ndarray, 
                   z_space: np.ndarray, params: FieldParameters) -> np.ndarray:
    field = np.zeros((params.resolution, params.resolution, params.resolution), dtype=np.float32)
    for i in prange(params.resolution):
        z = z_space[i]
        for j in range(params.resolution):
            y = y_space[j]
            for k in range(params.resolution):
                x = x_space[k]
                field[k, j, i] = gyroid_function(x, y, z, params)
    return field

@njit(parallel=True)
def generate_helical(x_space: np.ndarray, y_space: np.ndarray,
                    z_space: np.ndarray, params: FieldParameters) -> np.ndarray:
    field = np.zeros((params.resolution, params.resolution, params.resolution), dtype=np.float32)
    for i in prange(params.resolution):
        z = z_space[i]
        for j in range(params.resolution):
            y = y_space[j]
            for k in range(params.resolution):
                x = x_space[k]
                field[k, j, i] = helical_function(x, y, z, params)
    return field

@njit(parallel=True)
def apply_transition_layer(field: np.ndarray, x_space: np.ndarray,
                          y_space: np.ndarray, params: FieldParameters) -> np.ndarray:
    transition = np.zeros_like(field)
    for i in prange(params.resolution):
        for j in range(params.resolution):
            for k in range(params.resolution):
                x = x_space[k]
                y = y_space[j]
                r = math.hypot(x, y)
                transition[k, j, i] = calculate_transition_layer(r, params)
    return field * transition

# ==========================================
# الفئة الرئيسية مع التحسينات النهائية
# ==========================================

class FieldMerger:
    def __init__(self, params: FieldParameters):
        self._validate_parameters(params)
        self.params = params
        self._generate_cylindrical_grid()
    
    def _validate_parameters(self, params: FieldParameters):
        if params.inner_radius >= params.outer_radius:
            raise ValueError("نصف القطر الداخلي يجب أن يكون أصغر من الخارجي")
        if params.resolution <= 0:
            raise ValueError("الدقة يجب أن تكون موجبة")
    
    def _generate_cylindrical_grid(self):
        """إنشاء شبكة أسطوانية وتحويلها إلى كارتيزية"""
        r = np.linspace(self.params.inner_radius, self.params.outer_radius,
                       self.params.resolution, dtype=np.float32)
        theta = np.linspace(0, 2*np.pi, self.params.resolution, dtype=np.float32)
        z = np.linspace(0, self.params.height, self.params.resolution, dtype=np.float32)
        
        R, Theta, Z = np.meshgrid(r, theta, z, indexing='ij')
        self.X = R * np.cos(Theta)
        self.Y = R * np.sin(Theta)
        self.Z = Z
        
        self.x_space = np.unique(self.X)
        self.y_space = np.unique(self.Y)
        self.z_space = np.unique(self.Z)
    
    def generate_merged_field(self) -> np.ndarray:
        """دمج الحقول مع تطبيق طبقة الانتقال"""
        gyroid = generate_gyroid(self.x_space, self.y_space, self.z_space, self.params)
        helical = generate_helical(self.x_space, self.y_space, self.z_space, self.params)
        merged = np.maximum(np.abs(gyroid) - self.params.wall_thickness, -helical)
        return apply_transition_layer(merged, self.x_space, self.y_space, self.params)
    
    def generate_structured_grid(self) -> pv.StructuredGrid:
        grid = pv.StructuredGrid(self.X, self.Y, self.Z)
        grid["merged_field"] = self.generate_merged_field().ravel(order='F')
        return grid

# ==========================================
# وظيفة الإنشاء المعيارية (محدثة)
# ==========================================

def create_official_merger(resolution: int = 512) -> FieldMerger:
    params = FieldParameters(
        inner_radius=77.03,
        outer_radius=84.14,
        height=300.0,
        resolution=resolution,
        wall_thickness=0.3,
        gyroid_frequency=2 * math.pi / 7.5 * 0.8378,
        helical_frequency=2 * math.pi / 4.2,
        cell_size=7.5,
        phase_shift_factor=0.1,
        frequency_exponent=0.3,
        transition_width=1.0
    )
    return FieldMerger(params)

# ==========================================
# مثال استخدام
# ==========================================
if __name__ == "__main__":
    merger = create_official_merger(resolution=100)
    grid = merger.generate_structured_grid()
    grid.save("nrx_final_field.vti")
    print("تم إنشاء الحقل بنجاح!")