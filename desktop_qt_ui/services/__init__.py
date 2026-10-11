"""
Service initialisation module.
Manages and initialises all service components in one place
"""
import logging
import os
from typing import Any, Dict, Optional

# Import the editor core module (absolute import)
from desktop_qt_ui.editor.core import ResourceManager

from .async_service import AsyncService

# Import all services
from .config_service import ConfigService
from .file_service import FileService
from .history_service import EditorStateManager as HistoryService
from .i18n_service import I18nManager
from .log_service import LogService, setup_logging
from .ocr_service import OcrService
from .preset_service import PresetService
from .render_parameter_service import RenderParameterService
from .state_manager import StateManager
from .translation_service import TranslationService


class ServiceContainer:
    """Service container - the dependency injection container"""
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.services: Dict[str, Any] = {}
        self.initialized = False
        self._root_widget = None
        
        # Initialise logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self):
        """Set up logging"""
        log_dir = os.path.join(self.root_dir, "logs")
        setup_logging(log_dir, "MangaTranslatorUI")
        
    def initialize_services(self, root_widget=None) -> bool:
        """Fast service initialisation - asynchronous loading in stages"""
        try:
            self._root_widget = root_widget
            # All services are now initialized synchronously
            self._init_essential_services()
            self._init_heavy_services()
            
            # UI services can still be deferred if they depend on a fully-drawn widget
            if self._root_widget:
                # In Qt, we might not need .after(), can be called directly
                # For now, keeping the structure but it might be simplified
                self._init_ui_services()
            
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            return False
    
    def _init_essential_services(self):
        """Initialise the essential base services"""
        self.services['log'] = LogService()
        self.services['state'] = StateManager()
        self.services['config'] = ConfigService(self.root_dir)
        
        # Initialise the i18n service - the language setting is read from the configuration
        locale_dir = os.path.join(self.root_dir, "desktop_qt_ui", "locales")
        config = self.services['config'].get_config()
        ui_language = config.app.ui_language if hasattr(config.app, 'ui_language') else "auto"
        self.services['i18n'] = I18nManager(locale_dir=locale_dir, fallback_locale="zh_CN", config_language=ui_language)
        
        # Initialise the preset service
        self.services['preset'] = PresetService(config_service=self.services['config'])
        
        # Set the log level from the configuration
        try:
            config = self.services['config'].get_config()
            if hasattr(config, 'cli') and hasattr(config.cli, 'verbose'):
                verbose = config.cli.verbose
                self.services['log'].set_console_log_level(verbose)
        except Exception as e:
            self.logger.warning(f"Failed to set log level: {e}")
        
        self.services['state'].set_app_ready(True)
    
    def _init_heavy_services(self):
        """Initialise the heavy non-UI services in a background thread"""
        try:
            
            self.services['file'] = FileService()
            self.services['translation'] = TranslationService()
            self.services['ocr'] = OcrService()
            self.services['async'] = AsyncService()
            self.services['history'] = HistoryService()
            self.services['render_parameter'] = RenderParameterService()
            self.services['resource_manager'] = ResourceManager()  # The new resource manager

            
        except Exception as e:
            self.logger.error(f"Background initialization of heavyweight services failed: {e}")

    def _init_ui_services(self):
        """Initialise the UI-related services on the UI main thread"""
        # This method is now mostly obsolete as ShortcutManager and DragDropService are removed.
        # Kept for potential future UI-specific services.
    
    def _default_drop_callback(self, files):
        """Default drag-and-drop callback"""
        state_manager = self.get_service('state')
        if state_manager:
            current_files = state_manager.get_current_files()
            current_files.extend(files)
            state_manager.set_current_files(current_files)
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service instance"""
        return self.services.get(service_name)
    
    def register_service(self, name: str, service_instance: Any):
        """Register a new service"""
        self.services[name] = service_instance

    def _call_service_hook(self, service_name: str, *hook_names: str):
        """Call the shutdown hooks the services support, in order."""
        service = self.get_service(service_name)
        if not service:
            return

        for hook_name in hook_names:
            hook = getattr(service, hook_name, None)
            if callable(hook):
                hook()
                return

        self.logger.debug(f"Service {service_name} has no available shutdown hook: {hook_names}")
    
    def shutdown_services(self):
        """Shut down all services"""
        
        shutdown_steps = [
            ("async", ("shutdown",)),
            ("resource_manager", ("cleanup_all", "shutdown", "cleanup")),
            ("translation", ("cleanup", "shutdown", "close")),
            ("log", ("shutdown", "cleanup", "close")),
        ]

        for service_name, hook_names in shutdown_steps:
            try:
                self._call_service_hook(service_name, *hook_names)
            except Exception as e:
                self.logger.error(f"Error shutting down service {service_name}: {e}", exc_info=True)
        
        self.services.clear()
        self.initialized = False
        print("所有服务已关闭")

class ServiceManager:
    """Service manager - the global access point for services"""
    
    _instance = None
    _container = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, root_dir: str, root_widget=None) -> bool:
        """Initialise the service manager"""
        if cls._container is None:
            cls._container = ServiceContainer(root_dir)
            return cls._container.initialize_services(root_widget)
        return True
    
    @classmethod
    def get_service(cls, service_name: str) -> Optional[Any]:
        """Get a service"""
        if cls._container:
            return cls._container.get_service(service_name)
        return None

    @classmethod
    def register_service(cls, name: str, service_instance: Any):
        """Register a new service."""
        if cls._container:
            cls._container.register_service(name, service_instance)

    
    @classmethod
    def get_config_service(cls) -> Optional[ConfigService]:
        """Get the configuration service"""
        return cls.get_service('config')
    
    @classmethod
    def get_translation_service(cls) -> Optional[TranslationService]:
        """Get the translation service"""
        return cls.get_service('translation')
    
    @classmethod
    def get_file_service(cls) -> Optional[FileService]:
        """Get the file service"""
        return cls.get_service('file')
    
    @classmethod
    def get_state_manager(cls) -> Optional[StateManager]:
        """Get the state manager"""
        return cls.get_service('state')
    
    @classmethod
    def get_log_service(cls) -> Optional[LogService]:
        """Get the log service"""
        return cls.get_service('log')
    
    @classmethod
    def get_ocr_service(cls) -> Optional[OcrService]:
        """Get the OCR service"""
        return cls.get_service('ocr')

    @classmethod
    def get_render_parameter_service(cls) -> Optional[RenderParameterService]:
        """Get the render parameter service"""
        return cls.get_service('render_parameter')

    @classmethod
    def get_async_service(cls) -> Optional[AsyncService]:
        """Get the async service"""
        return cls.get_service('async')

    @classmethod
    def get_history_service(cls) -> Optional[HistoryService]:
        """Get the history service"""
        return cls.get_service('history')
    
    @classmethod
    def get_resource_manager(cls) -> Optional[ResourceManager]:
        """Get the resource manager"""
        return cls.get_service('resource_manager')
    
    @classmethod
    def get_i18n_manager(cls) -> Optional[I18nManager]:
        """Get the internationalisation manager"""
        return cls.get_service('i18n')
    
    @classmethod
    def get_preset_service(cls) -> Optional[PresetService]:
        """Get the preset service"""
        return cls.get_service('preset')
    
    @classmethod
    def shutdown(cls):
        """Shut down the service manager"""
        if cls._container:
            cls._container.shutdown_services()
            cls._container = None

# Convenience functions
def init_services(root_dir: str, root_widget=None) -> bool:
    """Convenience function that initialises the services"""
    return ServiceManager.initialize(root_dir, root_widget)

def get_config_service() -> Optional[ConfigService]:
    """Convenience function that gets the configuration service"""
    return ServiceManager.get_config_service()

def get_translation_service() -> Optional[TranslationService]:
    """Convenience function that gets the translation service"""
    return ServiceManager.get_translation_service()

def get_file_service() -> Optional[FileService]:
    """Convenience function that gets the file service"""
    return ServiceManager.get_file_service()

def get_state_manager() -> Optional[StateManager]:
    """Convenience function that gets the state manager"""
    return ServiceManager.get_state_manager()

def get_logger(name: str = None) -> logging.Logger:
    """Convenience function that gets the logger"""
    log_service = ServiceManager.get_log_service()
    if log_service:
        return log_service.get_logger(name)
    return logging.getLogger(name or __name__)

def get_ocr_service() -> Optional[OcrService]:
    """Convenience function that gets the OCR service"""
    return ServiceManager.get_ocr_service()

def get_render_parameter_service() -> Optional[RenderParameterService]:
    """Convenience function that gets the render parameter service"""
    return ServiceManager.get_render_parameter_service()

def get_async_service() -> Optional[AsyncService]:
    """Convenience function that gets the async service"""
    return ServiceManager.get_async_service()

def get_history_service() -> Optional[HistoryService]:
    """Convenience function that gets the history service"""
    return ServiceManager.get_history_service()

def get_resource_manager() -> Optional[ResourceManager]:
    """Convenience function that gets the resource manager"""
    return ServiceManager.get_resource_manager()

def get_i18n_manager() -> Optional[I18nManager]:
    """Convenience function that gets the internationalisation manager"""
    return ServiceManager.get_i18n_manager()

def get_preset_service() -> Optional[PresetService]:
    """Convenience function that gets the preset service"""
    return ServiceManager.get_preset_service()

def shutdown_services():
    """Convenience function that shuts down the services"""
    ServiceManager.shutdown()

# Dependency injection decorator
def inject_service(service_name: str):
    """Service injection decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            service = ServiceManager.get_service(service_name)
            return func(*args, **kwargs, **{service_name: service})
        return wrapper
    return decorator

# Service health check
def check_services_health() -> Dict[str, bool]:
    """Check the health of all services"""
    health_status = {}
    
    if ServiceManager._container:
        for service_name, service in ServiceManager._container.services.items():
            try:
                # Basic health check
                if hasattr(service, 'is_healthy'):
                    health_status[service_name] = service.is_healthy()
                else:
                    health_status[service_name] = service is not None
            except Exception:
                health_status[service_name] = False
    
    return health_status
