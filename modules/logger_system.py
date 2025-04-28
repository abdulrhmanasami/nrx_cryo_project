# logger_system.py
"""
NRX-Cryo Industrial-Grade Logging System v4.1
=============================================
Enterprise logging system with enhanced security and dynamic partitioning
**تم التحديث وفق التوصيات الأمنية والهندسية**
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict
from logging.handlers import RotatingFileHandler
import traceback
from cryptography.fernet import Fernet, InvalidToken

# Conditional imports and initialization
try:
    from colorama import init, Fore, Style
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Safe fallback for color attributes
    Fore = type('', (), {'__getattr__': lambda *_: ''})()
    Style = type('', (), {'__getattr__': lambda *_: ''})()

if sys.platform.startswith('win') and COLORAMA_AVAILABLE:
    init(convert=True)

class ColoredFormatter(logging.Formatter):
    """Advanced ANSI formatter with dynamic color handling"""
    
    COLOR_MAP = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT
    } if COLORAMA_AVAILABLE else {}
    
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''

    def format(self, record: logging.LogRecord) -> str:
        """Apply full message coloring safely"""
        if COLORAMA_AVAILABLE:
            color = self.COLOR_MAP.get(record.levelname, '')
            record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

class JsonFormatter(logging.Formatter):
    """Secure JSON formatter with AES-256 encryption"""
    
    def __init__(self, encrypt_key: Optional[str] = None):
        super().__init__()
        self.cipher = None
        if encrypt_key:
            self._validate_encryption_key(encrypt_key)
            self.cipher = Fernet(encrypt_key)
    
    def _validate_encryption_key(self, key: str):
        """Validate Fernet key format (32-byte URL-safe base64)"""
        if len(key) != 44:
            raise ValueError("Invalid encryption key: Must be 44 URL-safe base64 bytes")
        try:
            Fernet(key)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid encryption key: {str(e)}") from e

    def format(self, record: logging.LogRecord) -> str:
        """Generate encrypted or plain JSON log entry"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        json_data = json.dumps(log_entry, ensure_ascii=False)
        return self._encrypt(json_data) if self.cipher else json_data

    def _encrypt(self, data: str) -> str:
        """Encrypt data with error handling"""
        try:
            return self.cipher.encrypt(data.encode()).decode()
        except InvalidToken as e:
            raise RuntimeError(f"Encryption failed: {str(e)}") from e

class LoggerSystem:
    """Industrial logging system with enhanced features"""
    
    def __init__(self,
                 existing_logger: Optional[logging.Logger] = None,
                 log_level: int = logging.INFO,
                 log_dir: Union[str, Path] = "./logs",
                 log_prefix: str = "nrx_cryo_log",
                 enable_colors: bool = True,
                 console_output: bool = True,
                 output_format: str = "plain",
                 max_log_size: int = 10,
                 size_unit: str = 'MB',
                 backup_count: int = 5,
                 secure_mode: bool = False,
                 encryption_key: Optional[str] = None):
        """
        Initialize next-gen logging system
        
        Args:
            size_unit: File size unit ('MB' or 'GB')
            encryption_key: 44-byte URL-safe base64 key
        """
        self.log_stats: Dict[str, int] = {
            'DEBUG': 0, 'INFO': 0, 'WARNING': 0,
            'ERROR': 0, 'CRITICAL': 0
        }
        
        self._configure_system(
            existing_logger=existing_logger,
            log_level=log_level,
            log_dir=log_dir,
            log_prefix=log_prefix,
            enable_colors=enable_colors,
            console_output=console_output,
            output_format=output_format,
            max_log_size=max_log_size,
            size_unit=size_unit,
            backup_count=backup_count,
            secure_mode=secure_mode,
            encryption_key=encryption_key
        )

    def _configure_system(self, **kwargs) -> None:
        """Core system initialization"""
        self.logger = kwargs['existing_logger'] or logging.getLogger('NRX-Cryo')
        self.logger.setLevel(kwargs['log_level'])
        self.log_dir = Path(kwargs['log_dir'])
        self.log_prefix = kwargs['log_prefix']

        try:
            self._create_log_directory()
            self._configure_handlers(**kwargs)
            self._install_statistics_hook()
        except Exception as e:
            self._handle_critical_failure(e)

    def _handle_critical_failure(self, error: Exception) -> None:
        """Emergency error handling"""
        sys.stderr.write(f"CRITICAL LOGGER FAILURE: {str(error)}\n")
        traceback.print_exc()
        sys.exit(1)

    def _create_log_directory(self) -> None:
        """Create monthly directory with dynamic timestamp"""
        current_date = datetime.now()
        self.month_dir = self.log_dir / f"{current_date.year}-{current_date.month:02d}"
        self.month_dir.mkdir(parents=True, exist_ok=True)

    def _configure_handlers(self, **kwargs) -> None:
        """Configure all logging handlers"""
        self._add_file_handler(
            output_format=kwargs['output_format'],
            max_log_size=kwargs['max_log_size'],
            size_unit=kwargs['size_unit'],
            backup_count=kwargs['backup_count'],
            secure_mode=kwargs['secure_mode'],
            encryption_key=kwargs['encryption_key']
        )
        if kwargs['console_output']:
            self._add_console_handler(
                enable_colors=kwargs['enable_colors'],
                output_format=kwargs['output_format']
            )

    def _add_file_handler(self, **kwargs) -> None:
        """Configure secure rotating file handler"""
        # Calculate file size multiplier
        size_units = {'MB': 1024**2, 'GB': 1024**3}
        if kwargs['size_unit'] not in size_units:
            raise ValueError(f"Invalid size unit: {kwargs['size_unit']}. Use 'MB' or 'GB'")
        
        multiplier = size_units[kwargs['size_unit']]
        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
        log_file = self.month_dir / f"{self.log_prefix}_{timestamp}.log"
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=kwargs['max_log_size'] * multiplier,
            backupCount=kwargs['backup_count'],
            encoding='utf-8'
        )
        
        formatter = JsonFormatter(
            encrypt_key=kwargs['encryption_key']
        ) if kwargs['secure_mode'] else logging.Formatter(
            '[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _add_console_handler(self, **kwargs) -> None:
        """Configure intelligent console output"""
        console_handler = logging.StreamHandler()
        
        if kwargs['output_format'] == "json":
            formatter = JsonFormatter()
        else:
            formatter = ColoredFormatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            ) if kwargs['enable_colors'] else logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _install_statistics_hook(self) -> None:
        """Install logging statistics tracker"""
        def stats_hook(record: logging.LogRecord):
            self.log_stats[record.levelname] += 1
        self.logger.addFilter(stats_hook)

    def export_summary_report(self, filepath: Union[str, Path]) -> None:
        """Export operational report with error handling"""
        try:
            report = self._generate_report()
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        except (PermissionError, OSError, json.JSONEncodeError) as e:
            self.logger.error(f"Failed to export report: {str(e)}")
            raise

    def _generate_report(self) -> Dict:
        """Generate comprehensive system report"""
        return {
            "log_statistics": self.log_stats,
            "system_info": {
                "log_directory": str(self.log_dir.resolve()),
                "active_handlers": len(self.logger.handlers),
                "encryption_enabled": any(
                    isinstance(h.formatter, JsonFormatter) and h.formatter.cipher
                    for h in self.logger.handlers
                ),
                "current_partition": str(self.month_dir.name)
            }
        }

# Setup.py reference (for deployment)
"""
from setuptools import setup, find_packages

setup(
    name='nrx-logger',
    version='4.1',
    packages=find_packages(),
    install_requires=[
        'colorama',
        'cryptography>=3.4.7'
    ],
    entry_points={
        'console_scripts': [
            'nrx-logger-config = logger_system:main'
        ]
    }
)
"""