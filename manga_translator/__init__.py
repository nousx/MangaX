import colorama

from .utils.dotenv_utils import load_app_dotenv

colorama.init(autoreset=True)
load_app_dotenv(override=False)

# Lazy import, so the large libraries are not loaded for --help
def __getattr__(name):
    """Import MangaTranslator and the other classes lazily"""
    if name in ['MangaTranslator', 'Config', 'Context']:
        from .manga_translator import Config, Context, MangaTranslator
        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
