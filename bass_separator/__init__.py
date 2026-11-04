"""bass-separator: aislamiento y realce de contrabajo en grabaciones de jazz.

Flujo principal (ver :mod:`bass_separator.pipeline`):

1. Restauración previa de la grabación (de-click, filtrado, reducción de ruido suave).
2. Separación de fuentes con Demucs (Meta AI, pesos oficiales verificados por hash)
   o con un backend DSP sin redes neuronales.
3. Seguimiento de la línea de bajo (f0) y segmentación en notas.
4. Realce guiado por la altura: máscara armónica, reconstrucción de la fundamental,
   reducción de ruido/colado, ecualización y dinámica.
5. Resíntesis opcional del contrabajo (modelo físico de cuerda pulsada) alineada con
   las notas detectadas para obtener un sonido de "micrófono dedicado".
"""

__version__ = "0.1.0"

from .config import Settings, PRESETS  # noqa: F401
