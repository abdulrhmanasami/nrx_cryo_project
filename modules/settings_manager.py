"""
NRX-Cryo Project - Industrial-Grade Settings Manager (Enhanced)
Version: 1.3.1
Author: NRX Engineering Team
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

class SettingsError(Exception):
    """Base exception for settings management errors"""
    pass

class MissingSettingError(SettingsError):
    """Raised when a required setting is missing"""
    pass

class InvalidSettingTypeError(SettingsError):
    """Raised when a setting has invalid type"""
    pass

class SettingsValidationError(SettingsError):
    """Raised when settings fail validation checks"""
    pass

class UnknownSettingError(SettingsError):
    """Raised when encountering unknown settings"""
    pass

class SettingsManager:
    """
    Industrial-grade settings manager with enhanced validation
    """
    
    _TYPE_CHECKS = {
        "inner_radius": (float, int),
        "outer_radius": (float, int),
        "height": (float, int),
        "wall_thickness": (float, int),
        "resolution": int,
        "frequency_exponent": (float, int),
        "output_format": str,
        "validation_level": str,
        "smoothing_iterations": int
    }

    _DEFAULT_SETTINGS = {
        "inner_radius": 50.0,
        "outer_radius": 60.0,
        "height": 200.0,
        "wall_thickness": 0.5,
        "resolution": 100,
        "frequency_exponent": 0.3,
        "output_format": "stl",
        "validation_level": "strict",
        "smoothing_iterations": 3
    }

    def __init__(self, initial_settings: Dict[str, Any] = None):
        self._settings = self._DEFAULT_SETTINGS.copy()
        if initial_settings:
            self.update_settings(initial_settings)

    @property
    def current_settings(self) -> Dict[str, Any]:
        """Get read-only copy of current settings"""
        return self._settings.copy()

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load settings from JSON file with enhanced error handling
        """
        path = Path(file_path)
        try:
            with path.open('r') as f:
                file_settings = json.load(f)
        except FileNotFoundError as e:
            raise SettingsError(f"Configuration file not found: {path.resolve()}") from e
        except PermissionError as e:
            raise SettingsError(f"Permission denied: {path.resolve()}") from e
        except json.JSONDecodeError as e:
            raise SettingsError(f"Invalid JSON syntax: {e.msg}") from e
        except Exception as e:
            raise SettingsError(f"Unexpected error loading file: {str(e)}") from e

        self.update_settings(file_settings)

    def save_to_file(self, file_path: Union[str, Path], indent: int = 4) -> None:
        """Save settings to file with atomic write pattern"""
        path = Path(file_path)
        temp_path = path.with_suffix(".tmp")
        
        try:
            with temp_path.open('w') as f:
                json.dump(self._settings, f, indent=indent)
            temp_path.replace(path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise SettingsError(f"Failed to save settings: {str(e)}") from e

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Enhanced update with comprehensive validation"""
        # Validate unknown keys first
        unknown_keys = set(new_settings.keys()) - set(self._TYPE_CHECKS.keys())
        if unknown_keys:
            raise UnknownSettingError(f"Unknown settings detected: {list(unknown_keys)}")

        # Validate individual settings
        validation_errors = []
        for key, value in new_settings.items():
            try:
                self._validate_setting(key, value)
            except SettingsError as e:
                validation_errors.append(str(e))

        if validation_errors:
            raise SettingsValidationError("\n".join(validation_errors))

        # Apply updates and validate relationships
        self._settings.update(new_settings)
        try:
            self._validate_relationships()
        except SettingsValidationError as e:
            # Rollback changes if relationship validation fails
            self._settings = self._DEFAULT_SETTINGS.copy()
            raise

    def _validate_setting(self, key: str, value: Any) -> None:
        """Type and value validation with enhanced error messages"""
        # Type check
        expected_type = self._TYPE_CHECKS[key]
        if not isinstance(value, expected_type):
            raise InvalidSettingTypeError(
                f"Invalid type for '{key}'. Expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

        # Value validation
        if key == "resolution" and value < 50:
            raise SettingsValidationError(f"Resolution must be ≥50 (got {value})")
        if key == "validation_level" and value not in {"strict", "relaxed"}:
            raise SettingsValidationError(
                f"Invalid validation level '{value}'. Must be 'strict' or 'relaxed'"
            )

    def _validate_relationships(self) -> None:
        """Enhanced relationship validation with clear error messages"""
        if self._settings["inner_radius"] >= self._settings["outer_radius"]:
            raise SettingsValidationError(
                f"Invalid radii: Inner ({self._settings['inner_radius']}) "
                f"must be < Outer ({self._settings['outer_radius']})"
            )

        max_wall = (self._settings["outer_radius"] - self._settings["inner_radius"]) / 2
        if self._settings["wall_thickness"] > max_wall:
            raise SettingsValidationError(
                f"Wall thickness {self._settings['wall_thickness']} exceeds "
                f"maximum allowed {max_wall:.2f}"
            )

    @classmethod
    def for_environment(cls, env: str = "production") -> 'SettingsManager':
        """Environment loader with enhanced error reporting"""
        env_files = {
            "production": "settings_default.json",
            "development": "settings_dev.json",
            "validation": "settings_validation.json",
            "test": "settings_test.json"
        }

        if env not in env_files:
            raise SettingsError(
                f"Unknown environment '{env}'. Valid options: {list(env_files.keys())}"
            )

        config_path = Path(__file__).parent / "settings" / env_files[env]
        manager = cls()
        try:
            manager.load_from_file(config_path)
        except SettingsError as e:
            raise SettingsError(
                f"Failed to load {env} environment config: {str(e)}"
            ) from e
        return manager

    def __str__(self) -> str:
        return json.dumps(self._settings, indent=2)

# Example Usage
if __name__ == "__main__":
    try:
        manager = SettingsManager.for_environment("development")
        manager.set_setting("resolution", 200)
        print("Current configuration:")
        print(manager)
        manager.save_to_file("modified_settings.json")
        
        try:
            manager.set_setting("resolution", "high")
        except InvalidSettingTypeError as e:
            print(f"\nError: {str(e)}")

    except SettingsError as e:
        print(f"Configuration Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")