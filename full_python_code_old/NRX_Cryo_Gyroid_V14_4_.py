#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NRX Cryo-Gyroid Generator - الإصدار النهائي المعدل
"""

import argparse
import numpy as np
import pyvista as pv
from numba import njit, prange
import warnings
import tracemalloc
from math import sqrt

# ================================================================
# الدوال الرياضية الأساسية مع تحسينات Numba
# ================================================================

@njit(fastmath=True, cache=True, inline='always')
def gyroid(x, y, z, freq):
    """حساب قيمة الجايرويد مع تحسين دقة الحساب"""
    return (
        np.sin(freq*x)*np.cos(freq*y) + 
        np.sin(freq*y)*np.cos(freq*z) + 
        np.sin(freq*z)*np.cos(freq*x)
    )

@njit(fastmath=True, cache=True)
def compute_frequency(r, r_inner, r_outer, f0, f_min, alpha):
    """حساب التردد مع حدود أمان ديناميكية"""
    r_clamped = min(max(r, r_inner), r_outer)
    ratio = (r_outer - r_clamped)/(r_outer - r_inner)
    return f_min + (f0 - f_min)*(ratio**alpha)

# ================================================================
# محرك الحساب المتوازي مع إدارة الذاكرة المتقدمة
# ================================================================

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def parallel_compute(X, Y, Z, params):
    """الحساب المتوازي مع معالجة ذكية للحدود"""
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
# النظام الرئيسي مع واجهة تحكم متكاملة
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
        """إنشاء شبكة ثلاثية الأبعاد متوافقة الأبعاد"""
        buffer_factor = 1 + (buffer_percent/100)
        x = np.linspace(
            -self.params['r_outer']*buffer_factor,
            self.params['r_outer']*buffer_factor,
            resolution,
            dtype=np.float32
        )
        y = x.copy()  # التأكد من تطابق الأبعاد
        z = np.linspace(
            0, 
            self.params['height'], 
            resolution,  # نفس الدقة لجميع المحاور
            dtype=np.float32
        )
        
        # إنشاء الشبكة بدون sparse=True
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return X, Y, Z
    
    def generate(self, args):
        """العملية الرئيسية للتوليد"""
        tracemalloc.start()
        
        try:
            # إنشاء الشبكة
            X, Y, Z = self.create_grid(args.resolution, args.buffer)
            
            # حساب الحقل
            params = (
                self.params['r_inner'],
                self.params['r_outer'],
                self.params['f0'],
                self.params['f_min'],
                self.params['alpha'],
                args.buffer
            )
            
            G = parallel_compute(X, Y, Z, params)
            F = np.abs(G) - args.wall_thickness
            
            # توليد الشبكة
            grid = pv.StructuredGrid(X, Y, Z)
            grid["F"] = F.ravel(order='F')
            mesh = grid.contour([0.0])
            
            if mesh.n_points == 0:
                raise RuntimeError("فشل في توليد السطح - تحقق من المعلمات")
            
            return mesh
        
        except MemoryError:
            peak = tracemalloc.get_traced_memory()[1]
            print(f"خطأ: تجاوز سقف الذاكرة ({peak/1e9:.2f} GB)")
            print("الحلول المقترحة:")
            print(f"1. تقليل الدقة (الحالية: {args.resolution})")
            print(f"2. تقليل العازل (الحالية: {args.buffer}%)")
            print("3. تفعيل الوضع الآمن (--safe)")
            return None
        
        finally:
            tracemalloc.stop()

# ================================================================
# واجهة المستخدم والتحكم البصري
# ================================================================

class VisualizationEngine:
    def __init__(self):
        self.plotter = pv.Plotter()
        self.plotter.set_background("#1f1f1f")
        self.plotter.enable_anti_aliasing('ssaa')
    
    def add_controls(self, mesh):
        """إضافة عناصر التحكم التفاعلية"""
        # شريط التحكم في الشفافية
        self.plotter.add_slider_widget(
            lambda v: self.plotter.actor.GetProperty().SetOpacity(v),
            [0.3, 1.0],
            title='الشفافية'
        )
        
        # شريط التحكم في الإضاءة
        self.plotter.add_slider_widget(
            lambda v: self.plotter.actor.GetProperty().SetSpecularPower(v),
            [0.1, 5.0],
            title='قوة الإضاءة'
        )
        
        # معلومات الأداء
        self.plotter.add_text(
            f"النقاط: {mesh.n_points:,}\nالخلايا: {mesh.n_cells:,}",
            position="lower_right",
            font_size=12,
            color="white"
        )
    
    def show(self, mesh, clip_half=False):
        """عرض النموذج"""
        if clip_half:
            mesh = mesh.clip(normal='x', invert=False)
        
        self.plotter.add_mesh(
            mesh,
            scalars="F",
            cmap="coolwarm",
            specular=0.8,
            diffuse=0.8,
            smooth_shading=True,
            show_edges=False,
            edge_color="white",
            scalar_bar_args={'title': "قيمة الحقل"}
        )
        
        self.add_controls(mesh)
        self.plotter.show()

# ================================================================
# الإعدادات الرئيسية ومعالجة الأوامر
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NRX Cryo-Gyroid Generator - الإصدار النهائي",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-r", "--resolution", type=int, default=400,
                      help="الدقة الثلاثية الأبعاد")
    parser.add_argument("-w", "--wall-thickness", type=float, default=0.3,
                      help="سماكة الجدار بالميليمتر")
    parser.add_argument("-b", "--buffer", type=float, default=5.0,
                      help="النسبة المئوية للمنطقة العازلة")
    parser.add_argument("--safe", action='store_true',
                      help="وضع الذاكرة الآمن")
    parser.add_argument("--clip", action='store_true',
                      help="عرض النصف الأمامي فقط")
    
    args = parser.parse_args()
    
    # تطبيق قيود الأمان
    if args.safe:
        args.resolution = min(args.resolution, 350)
        args.buffer = min(args.buffer, 10.0)
        warnings.warn(f"الوضع الآمن: الدقة {args.resolution}، العازل {args.buffer}%")
    
    # التنفيذ الرئيسي
    generator = GyroidGenerator()
    engine = VisualizationEngine()
    
    try:
        mesh = generator.generate(args)
        if mesh is not None:
            engine.show(mesh, args.clip)
    except Exception as e:
        print(f"خطأ غير متوقع: {str(e)}")
    finally:
        print("العملية اكتملت")

if __name__ == "__main__":
    main()