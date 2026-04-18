"""Configuración de pytest."""
import sys
from pathlib import Path

# Añadir el directorio raíz al path para imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
