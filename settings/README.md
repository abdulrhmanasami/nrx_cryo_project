# Settings Manager - NRX-Cryo

Official settings JSON files for managing NRX-Cryo configuration parameters.

## 📄 Parameter Overview

| Parameter          | Min  | Max   | Unit | Notes                          |
|--------------------|------|-------|------|--------------------------------|
| resolution         | 50   | 1024  | -    | Affects processing time        |
| wall_thickness     | 0.1  | 20    | mm   | Must fit between radii         |
| export_precision   | 0.001| 1.0   | mm   | Applies to VTU/3MF exports only |

## 🚀 Usage Example

```python
from settings_manager import SettingsManager
settings = SettingsManager.for_environment("production")
resolution = settings.get_setting("resolution")