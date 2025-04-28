
# NRX-Cryo Gyroid Generator

---

## 📖 Overview

This project implements the NRX-Cryo cylindrical gyroid heat exchanger model based on industrial specifications.  
The model uses a mathematically precise generation of the gyroid structure with radial frequency gradient, prepared for additive manufacturing.

---

## 📂 Project Structure

```
nrx_cryo_project/
│
├── main.py            # Main execution entry point
├── README.md          # Project documentation
├── modules/           # Modular Python scripts (generation, export, validation, etc.)
├── logs/              # Execution log files
├── exports/           # Exported 3D models (STL, VTK)
├── settings/          # Saved settings (JSON)
```

---

## 🚀 How to Run

```bash
python main.py --resolution 400 --wall-thickness 0.3
```

(Additional CLI options will be documented as modules are completed.)

---

## ⚡ Current Status

- Project is under **active modular construction**.
- Each module is being implemented step-by-step according to strict industrial guidelines.

---

## 🛠️ Note for Developers

- No shortcuts, no assumptions.
- Strictly follow the provided work instructions per module.
- Always maintain modular independence and single-responsibility per file.
- Do not modify the project structure or architecture without explicit approval.

---
