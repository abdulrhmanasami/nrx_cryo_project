#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NRX-Cryo Generator - Core Control System
========================================
Version: 2.1.0  # Updated version
Description: Central entry point for industrial gyroid generation pipeline
"""

import argparse
import sys
import logging
import json
from datetime import datetime
from modules import core_generator, industrial_checks, file_export

# Global configuration
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
CONFIG_VERSION = "2.1"  # Added versioning for configs

def validate_settings(settings: dict):
    """Validate loaded JSON settings against industrial standards"""
    if not 50 <= settings.get('resolution', 0) <= 500:
        raise ValueError(f"Invalid resolution: {settings['resolution']}. Must be 50-500.")
    
    if settings.get('wall_thickness', 0) <= 0:
        raise ValueError(f"Wall thickness must be positive. Got: {settings['wall_thickness']}")
    
    if settings.get('buffer', 0) < 0:
        raise ValueError(f"Buffer zone cannot be negative. Got: {settings['buffer']}")

def configure_logging(log_file=None):
    """Initialize industrial-grade logging system"""
    logger = logging.getLogger('NRX_Main')
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    
    return logger

def main():
    """Main execution pipeline"""
    parser = argparse.ArgumentParser(
        description="NRX-Cryo Industrial Gyroid Generator",
        epilog=f"Version: {CONFIG_VERSION} | Engineering Team"
    )
    
    # Core parameters (Added input validation)
    parser.add_argument('-r', '--resolution', type=int, default=100,
                      choices=range(50, 501),  # Enforce valid range
                      metavar="[50-500]",
                      help='Grid resolution (50-500 points)')
    
    parser.add_argument('-w', '--wall-thickness', type=float, default=0.3,
                      help='Structural wall thickness (mm)')
    parser.add_argument('-b', '--buffer', type=float, default=2.0,
                      help='Buffer zone extension (mm)')
    
    # Operational modes
    parser.add_argument('--safe', action='store_true',
                      help='Enable memory-safe mode')
    parser.add_argument('--clip', action='store_true',
                      help='Enable model clipping for preview')
    
    # I/O configurations
    parser.add_argument('--export-stl', type=str,
                      help='STL export filename')
    parser.add_argument('--export-vtk', type=str,
                      help='VTK export filename')
    parser.add_argument('--save-settings', type=str,
                      help='Save configuration to JSON file')
    parser.add_argument('--load-settings', type=str,
                      help='Load configuration from JSON file')
    
    # Advanced controls
    parser.add_argument('--log-file', type=str,
                      help='Execution log filename')
    parser.add_argument('--chunk-size', type=int, default=1024,
                      help='Processing chunk size')
    parser.add_argument('--transition-width', type=float, default=1.0,
                      help='Boundary transition width')
    parser.add_argument('--apply-thickness', action='store_true',
                      help='Enable true wall thickness generation')
    parser.add_argument('--skip-industrial-checks', action='store_true',
                      help='Bypass industrial validation checks')
    
    args = parser.parse_args()
    logger = configure_logging(args.log_file)
    
    try:
        logger.info("=== Starting industrial generation process ===")
        
        # Configuration management
        settings = {}
        if args.load_settings:
            with open(args.load_settings, 'r') as f:
                settings = json.load(f)
                logger.info(f"Loaded configuration from: {args.load_settings}")
                
                # Validate loaded settings
                try:
                    validate_settings(settings)
                    logger.debug(f"Config validation passed: {settings}")
                except ValueError as ve:
                    logger.critical(f"Invalid configuration: {str(ve)}")
                    sys.exit(1)
                
                # Version compatibility check
                if settings.get('version', '1.0') != CONFIG_VERSION:
                    logger.warning("Config version mismatch! May cause unexpected behavior")
        
        # Initialize core components with priority to loaded settings
        generator = core_generator.IndustrialGenerator(
            resolution=settings.get('resolution', args.resolution),
            wall_thickness=settings.get('wall_thickness', args.wall_thickness),
            buffer_zone=settings.get('buffer', args.buffer)
        )
        
        # Safety validations
        if not args.skip_industrial_checks:
            try:
                industrial_checks.run_safety_checks(
                    generator, 
                    memory_safe=args.safe
                )
            except industrial_checks.IndustrialSafetyError as ise:
                logger.critical(f"Safety checks failed: {str(ise)}")
                sys.exit(1)
        
        # Generation pipeline
        try:
            generated_model = generator.generate(
                apply_thickness=args.apply_thickness,
                chunk_size=args.chunk_size
            )
        except MemoryError as me:
            logger.critical(f"Memory overload: {str(me)}. Try smaller chunk-size or enable safe-mode.")
            sys.exit(1)
        except core_generator.GenerationError as ge:
            logger.critical(f"Generation failed: {str(ge)}")
            sys.exit(1)
        
        # Export operations
        try:
            if args.export_stl:
                file_export.export_to_stl(
                    generated_model, 
                    args.export_stl,
                    clip_model=args.clip
                )
                logger.info(f"Exported STL: {args.export_stl}")
                
            if args.export_vtk:
                file_export.export_to_vtk(
                    generated_model, 
                    args.export_vtk,
                    clip_model=args.clip
                )
                logger.info(f"Exported VTK: {args.export_vtk}")
        except file_export.ExportError as ee:
            logger.critical(f"Export failed: {str(ee)}")
            sys.exit(1)
        
        # Configuration persistence (with versioning)
        if args.save_settings:
            settings = {
                'version': CONFIG_VERSION,  # Added version tracking
                'resolution': args.resolution,
                'wall_thickness': args.wall_thickness,
                'buffer': args.buffer,
                'timestamp': datetime.now().isoformat(),
                '_comment': "TIMESTAMP format: ISO 8601 (YYYY-MM-DDTHH:MM:SS.mmmmmm)"  # Added documentation
            }
            with open(args.save_settings, 'w') as f:
                json.dump(settings, f, indent=4)
            logger.info(f"Saved configuration: {args.save_settings}")
        
        logger.info("=== Process completed successfully ===")
        
    except (ValueError, json.JSONDecodeError) as ve:
        logger.critical(f"Configuration error: {str(ve)}")
        sys.exit(1)
    except IOError as ioe:
        logger.critical(f"I/O operation failed: {str(ioe)}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()