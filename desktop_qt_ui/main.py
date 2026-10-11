from contextlib import redirect_stdout
from io import StringIO

import logging
import os
import sys
import warnings

# Silence warnings of third-party libraries (must be set before the other libraries are imported)
warnings.filterwarnings('ignore', message='.*Triton.*')
warnings.filterwarnings('ignore', message='.*triton.*')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*pynvml package is deprecated.*', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ctranslate2')
warnings.filterwarnings('ignore', module='xformers')

# Set GPU memory options before PyTorch initialises, allowing shared GPU memory
# expandable_segments reduces GPU memory fragmentation and avoids OOM errors
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# Let the desktop app load long images whose decoded size exceeds Qt's default limit of 256 MiB.
os.environ.setdefault('QT_IMAGEIO_MAXALLOC', '1024')

# Fix for the path of portable Python: put the script folder at the start of sys.path
# Portable Python uses a ._pth file, which turns off the default of adding the script folder automatically
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Let runtime modules read the .env the desktop app actually uses at import time as well.
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    env_dir = os.path.dirname(sys.executable)
else:
    env_dir = project_root
os.environ.setdefault('MANGA_TRANSLATOR_ENV_PATH', os.path.join(env_dir, '.env'))

# Fix for loading the onnxruntime DLLs in a PyInstaller build
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running inside a PyInstaller build
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        # Only set the DLL search path; nothing is preloaded
        # Let Python's import machinery load the DLLs in its own way
        os.add_dll_directory(sys._MEIPASS)
        onnx_capi_dir = os.path.join(sys._MEIPASS, 'onnxruntime', 'capi')
        if os.path.exists(onnx_capi_dir):
            os.add_dll_directory(onnx_capi_dir)

# Load PyTorch before PyQt6, so the Qt DLL path of PyQt6 does not interfere with loading c10.dll
# See: https://github.com/pytorch/pytorch/issues/166628
try:
    import torch  # noqa: F401
except ImportError:
    pass

# qfluentwidgets prints a promotional message unconditionally on import; the desktop entry point silences just this one import.
with redirect_stdout(StringIO()):
    import qfluentwidgets  # noqa: F401

from ui.main_window import MainWindow
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from services import init_services
from utils.app_version import get_app_version
from utils.resource_helper import iter_existing_resource_paths, load_icon_from_resources
from ui.secondary_pages.themed_message_box import install_themed_message_boxes


# Global exception handler: catches unhandled exceptions and writes them to the log
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """全局异常处理器，防止程序静默崩溃"""
    import traceback
    
    # Ignore KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Format the exception
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    # Write to the log (goes to result/log_*.txt)
    logging.critical(f"Unhandled exception caused the application to crash:\n{error_msg}")
    
    # Print to the console as well (so it is certainly seen)
    print(f"\n{'='*60}", file=sys.stderr)
    print("❌ 程序发生未捕获的异常:", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(error_msg, file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

# Install the global exception handler
sys.excepthook = global_exception_handler


def _set_windows_app_user_model_id():
    """确保 Windows 将直接脚本启动识别为独立应用，而不是 python.exe。"""
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'manga.translator.ui.1.0'
        )
    except Exception:
        logging.exception("Failed to set Windows AppUserModelID")

def _apply_windows_window_class_icon(window, icon_path: str):
    """在首次显示前设置窗口类图标，供任务栏初始化时读取。"""
    if not icon_path:
        return False

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.LoadImageW.argtypes = [
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            wintypes.UINT,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.SetClassLongPtrW.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        user32.SetClassLongPtrW.restype = ctypes.c_ssize_t

        image_icon = 1
        lr_loadfromfile = 0x0010
        big_icon = user32.LoadImageW(
            None, icon_path, image_icon, 256, 256, lr_loadfromfile
        )
        small_icon = user32.LoadImageW(
            None, icon_path, image_icon, 32, 32, lr_loadfromfile
        )

        hwnd = wintypes.HWND(int(window.winId()))
        if big_icon:
            user32.SetClassLongPtrW(hwnd, -14, big_icon)  # GCLP_HICON
        if small_icon:
            user32.SetClassLongPtrW(hwnd, -34, small_icon)  # GCLP_HICONSM

        if big_icon or small_icon:
            # Keep the native handles alive for the whole window lifetime.
            window._native_class_icon_handles = (big_icon, small_icon)
            return True

        logging.warning(f"Failed to load Windows window class icon: {icon_path}")
    except Exception:
        logging.exception("Failed to set Windows window class icon")
    return False


def _apply_macos_native_app_icon(icon_path: str):
    """为 macOS Dock/原生应用层设置 .icns 图标。"""
    if not icon_path:
        return False

    try:
        from AppKit import NSApplication, NSImage

        image = NSImage.alloc().initWithContentsOfFile_(icon_path)
        if not image:
            logging.warning(f"Failed to load macOS native application icon: {icon_path}")
            return False

        NSApplication.sharedApplication().setApplicationIconImage_(image)
        logging.info(f"macOS native application icon set: {icon_path}")
        return True
    except ImportError:
        logging.info("PyObjC/AppKit is not installed; skipping macOS native Dock icon setup")
    except Exception:
        logging.exception("Failed to set macOS native application icon")
    return False


def main():
    """
    应用主入口
    """
    # --- Logging setup: all formatting and all console/file/recent writes happen in the listener thread ---
    import atexit
    from services.log_service import configure_queue_logging, shutdown_queue_logging

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    
    # --- Log file setup ---
    from datetime import datetime
    
    # The log folder is result/, next to app.exe
    if getattr(sys, 'frozen', False):
        log_dir = os.path.join(os.path.dirname(sys.executable), 'result')
    else:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    log_dir = os.path.normpath(os.path.abspath(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    
    # Build a log file name with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file_path = os.path.normpath(os.path.abspath(os.path.join(log_dir, f'log_{timestamp}.txt')))
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8', delay=False)
    file_handler.setLevel(logging.DEBUG)  # Always at DEBUG level
    file_handler.setFormatter(log_formatter)
    configure_queue_logging((console_handler, file_handler), queue_size=10_000)
    atexit.register(shutdown_queue_logging)
    
    logging.info(f"UI log file: {log_file_path}")
    
    # --- Make sure the configuration files exist ---
    try:
        from manga_translator.runtime_files import ensure_runtime_files
        ensure_runtime_files(logging.getLogger("manga_translator"))
    except Exception as e:
        logging.warning(f"Failed to create configuration file: {e}")
    
    # --- Crash capture (faulthandler) ---
    # Enable faulthandler to catch crashes at the C++ level (segmentation faults and the like)
    # The crash information is written straight to the same log file
    import faulthandler
    # Use the stream object of file_handler
    # all_threads=False: while a native file dialog is open, Windows raises the harmless
    # 0x8001010e (RPC_E_WRONG_THREAD) very often; each time faulthandler walks the frame stacks of all
    # running threads without a lock, which races with the OCR/inpainting threads and ends in an access violation and a crash
    faulthandler.enable(file=file_handler.stream, all_threads=False)

    # --- Environment setup ---
    # Windows only: the AppUserModelID has to be set before QApplication is created
    if sys.platform == 'win32':
        _set_windows_app_user_model_id()

        # qframelesswindow#185: opening a FramelessDialog must not force its
        # sibling widgets to become native, or maximize/restore can duplicate
        # and offset their rendered surfaces.
        QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings,
            True,
        )
    
    # 1. Create the QApplication instance
    app = QApplication(sys.argv)
    app.setApplicationName("MangaX")
    app.setOrganizationName("Manga Translator UI")
    app_version = get_app_version()
    if app_version != "unknown":
        app.setApplicationVersion(app_version)
        logging.info(f"UI version: {app_version}")
    install_themed_message_boxes()
    
    # Install the Qt message handler (catches exceptions in signals and slots)
    def qt_message_handler(mode, context, message):
        """Qt 消息处理器，捕获 Qt 内部错误"""
        from PyQt6.QtCore import QtMsgType
        if mode == QtMsgType.QtFatalMsg:
            logging.critical(f"Qt Fatal: {message} (file: {context.file}, line: {context.line})")
        elif mode == QtMsgType.QtCriticalMsg:
            logging.error(f"Qt Critical: {message}")
        elif mode == QtMsgType.QtWarningMsg:
            # Filter out some common harmless warnings
            if "QWindowsWindow::setGeometry" not in message:
                logging.warning(f"Qt Warning: {message}")
        # Debug and Info levels are not recorded, to keep the log small
    
    from PyQt6.QtCore import qInstallMessageHandler
    qInstallMessageHandler(qt_message_handler)
    
    if sys.platform == 'darwin':
        icon_relative_path = os.path.join('doc', 'images', 'icon.icns')
    elif sys.platform == 'win32':
        icon_relative_path = os.path.join('desktop_qt_ui', 'ui', 'icons', 'icon.ico')
    else:
        icon_relative_path = os.path.join('doc', 'images', 'icon.png')

    # One icon source; both the Qt application icon and the Windows window class icon use it.
    app_icon, icon_source = load_icon_from_resources([icon_relative_path])
    if app_icon and not app_icon.isNull():
        app.setWindowIcon(app_icon)
        logging.info(f"UI icon set: {icon_source}")
    else:
        logging.warning(f"Failed to load UI icon: {icon_relative_path}")

    if sys.platform == 'darwin':
        native_macos_icon_path = next(
            iter_existing_resource_paths([os.path.join('doc', 'images', 'icon.icns')]),
            None,
        )
        if native_macos_icon_path:
            _apply_macos_native_app_icon(native_macos_icon_path)
        else:
            logging.warning("macOS native application icon not found: doc/images/icon.icns")


    # 2. Initialise all services
    # When packaged, the resource root is the folder of app.exe; _internal only holds dependencies.
    if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # Development: the resources are in the project root
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if not init_services(root_dir):
        logging.fatal("Fatal: Service initialization failed.")
        sys.exit(1)

    # 3. Create and show the main window
    main_window = MainWindow()

    # FluentTitleBar only listens to windowIconChanged and does not read the inherited application icon.
    if app_icon and not app_icon.isNull():
        main_window.setWindowIcon(app_icon)

    # On first show the Windows taskbar may read the window class icon instead of WM_GETICON.
    if sys.platform == 'win32':
        _apply_windows_window_class_icon(main_window, icon_relative_path)
    

    main_window.show()

    # Avoid processing events synchronously inside the initial show sequence on Windows.
    # That would start re-entrant message handling in Qt/Windows and may cause RPC_E_CANTCALLOUT_ININPUTSYNCCALL.
    from PyQt6.QtCore import QTimer

    def finalize_window_activation():
        """启动置前的最小集合。

        Windows 上普通进程直接调 SetForegroundWindow 常被系统拒绝
        （前台锁定），因此保留 AttachThreadInput 技巧：临时挂接到当前
        前台窗口所在线程的输入队列后再置前。TOPMOST/NOTOPMOST 往返、
        重复 ShowWindow、SetActiveWindow/SetFocus 等冗余调用已移除——
        它们对已完成首帧的窗口只产生一轮 z-order 抖动（启动闪烁）。"""
        try:
            if main_window.isMinimized():
                main_window.showNormal()

            main_window.raise_()
            main_window.activateWindow()

            if sys.platform == 'win32':
                try:
                    import ctypes
                    from ctypes import wintypes

                    user32 = ctypes.windll.user32
                    kernel32 = ctypes.windll.kernel32

                    user32.GetForegroundWindow.restype = wintypes.HWND
                    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
                    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
                    user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
                    user32.AttachThreadInput.restype = wintypes.BOOL
                    user32.BringWindowToTop.argtypes = [wintypes.HWND]
                    user32.BringWindowToTop.restype = wintypes.BOOL
                    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
                    user32.SetForegroundWindow.restype = wintypes.BOOL
                    kernel32.GetCurrentThreadId.restype = wintypes.DWORD

                    hwnd = int(main_window.winId())
                    if hwnd:
                        foreground_hwnd = user32.GetForegroundWindow()
                        current_thread_id = kernel32.GetCurrentThreadId()
                        foreground_thread_id = 0
                        if foreground_hwnd:
                            foreground_thread_id = user32.GetWindowThreadProcessId(
                                wintypes.HWND(foreground_hwnd),
                                None,
                            )

                        attached = False
                        if foreground_thread_id and foreground_thread_id != current_thread_id:
                            attached = bool(
                                user32.AttachThreadInput(
                                    wintypes.DWORD(foreground_thread_id),
                                    wintypes.DWORD(current_thread_id),
                                    True,
                                )
                            )

                        try:
                            user32.BringWindowToTop(wintypes.HWND(hwnd))
                            user32.SetForegroundWindow(wintypes.HWND(hwnd))
                        finally:
                            if attached:
                                user32.AttachThreadInput(
                                    wintypes.DWORD(foreground_thread_id),
                                    wintypes.DWORD(current_thread_id),
                                    False,
                                )
                except Exception as exc:
                    logging.debug(f"Failed to bring Windows application to foreground: {exc}")
        except Exception as exc:
            logging.debug(f"Failed to activate main window: {exc}")

    # Scheduled only once: a second full activation sequence after 250ms is pointless for a window that is already shown,
    # and it was the cause of the window flicker at start-up
    QTimer.singleShot(0, finalize_window_activation)

    # 4. Start the event loop
    ret = app.exec()

    # Persist the latest coalesced config/.env snapshots before services vanish.
    try:
        from services import get_config_service
        config_service = get_config_service()
        if config_service is not None and not config_service.shutdown():
            logging.error("Configuration service could not save all pending writes before shutdown")
    except Exception as e:
        logging.error(f"Error shutting down configuration service: {e}", exc_info=True)

    try:
        from services import shutdown_services
        shutdown_services()
    except Exception as e:
        logging.error(f"Error shutting down services: {e}", exc_info=True)

    try:
        faulthandler.disable()
        shutdown_queue_logging()
    except Exception as e:
        print(f"关闭日志处理器时出错: {e}", file=sys.stderr)
    return ret

if __name__ == '__main__':
    # Set the DPI policy before QApplication is created; another reliable way to deal with DPI problems
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    raise SystemExit(main())
