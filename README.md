# bass-separator

Aplicación con inteligencia artificial que **aísla y realza la pista de contrabajo
(bajo acústico)** de una grabación de jazz en formato WAV. Está pensada para
grabaciones antiguas de big band (Glenn Miller, Count Basie, Benny Goodman,
Charlie Christian, Duke Ellington…), transferidas desde 78 rpm, mono, con poco
ancho de banda y mucho ruido, en las que la línea de bajo apenas se intuye.

La salida es la pista de contrabajo sola, limpiada y realzada como si el
bajista hubiera tenido un micrófono dedicado, conservando sus fraseos y su
*walking bass*; además se entrega la línea de bajo transcrita en MIDI y JSON.

## Cómo funciona

```
WAV  ─►  Restauración previa  ─►  Separación (Demucs)  ─►  Seguimiento de f0 y notas
            de-click, DC,            stem "bass"            salience armónica + Viterbi,
            siseo suave                                     ataques, segmentación
                                                                   │
         WAV contrabajo  ◄─  Dinámica/EQ  ◄─  Resíntesis (opcional)  ◄─  Realce guiado por f0
         MIDI + JSON         limitador        modelo físico de cuerda      máscara armónica, puerta
                                              pulsada + cuerpo             espectral, HPSS, graves
```

1. **Restauración previa** (perfil `vintage`): filtro de DC/subsónicos,
   eliminación de clics y crepitar por interpolación cúbica, reducción suave del
   siseo y, opcionalmente, del zumbido de red (50/60 Hz) si se detecta.
2. **Separación de fuentes** con [Demucs](https://github.com/facebookresearch/demucs)
   (Hybrid Transformer Demucs, Meta AI Research, licencia MIT). Por defecto se
   usa `htdemucs_ft`; como solo se necesita el bajo, se carga únicamente el
   sub-modelo entrenado para esa fuente (4× más rápido, resultado idéntico).
   Se pueden promediar varios modelos (`--model htdemucs_ft,mdx_extra`).
3. **Seguimiento de la línea de bajo**: un seguidor de f0 propio basado en
   *salience* armónica con decodificación de Viterbi, diseñado para grabaciones
   en las que la fundamental (40-80 Hz) se perdió: la altura se deduce de los
   armónicos y se comprueba la ausencia de sub-armónicos para evitar errores de
   octava. Las notas se segmentan con los ataques y los cambios de altura.
4. **Realce guiado por la altura**: en una sola pasada STFT se combinan una
   máscara armónica adaptativa (deja pasar k·f0 con tolerancia a la
   inarmonicidad de la cuerda), una puerta espectral que nunca toca los bins
   armónicos, protección de los ataques y reducción del colado de batería
   (HPSS). Después se **reconstruye la fundamental y el 2.º armónico** que la
   grabación perdió, con un oscilador síncrono con la f0 y nivel deducido de los
   armónicos presentes. Finalmente ecualización de cuerpo/presencia, compresión
   y limitación.
5. **Resíntesis** (modo `hybrid`, por defecto): cada nota detectada se vuelve a
   tocar con un modelo físico de contrabajo pizzicato (Karplus-Strong extendido
   + resonancias de cuerpo) respetando instante, duración, afinación real y
   dinámica, y se mezcla a bajo nivel bajo el stem realzado. Esto aporta el
   "cuerpo" de micrófono cercano sin inventar notas: la capa sigue
   exactamente la línea transcrita.

### Sobre los modelos y la seguridad

* Solo se usan dependencias de primera línea: PyTorch, Demucs (Meta), librosa,
  SciPy, NumPy, libsndfile. No hay plugins ni binarios de terceros.
* Los pesos de Demucs se descargan de los servidores oficiales de Meta
  (`dl.fbaipublicfiles.com/demucs/`) y `torch.hub` verifica su SHA-256 contra el
  hash incluido en el nombre del archivo (`check_hash=True`); un archivo
  manipulado se rechaza. Se guardan en la caché de `torch.hub`
  (`~/.cache/torch/hub/checkpoints/`). Para uso sin conexión: `bass-separator
  download-models` o `--model-repo CARPETA`.
* **No se entrenó ninguna red neuronal**: no existe un conjunto de datos de
  contrabajo de jazz aislado con el que superar a Demucs (entrenado con cientos
  de horas de multipistas), y un modelo entrenado solo con datos sintéticos
  sería peor. Todo el conocimiento específico del contrabajo (rango E1-G3,
  armónicos, inarmonicidad, cuerpo, articulación del walking bass) está en los
  módulos DSP, que se validan con pruebas automáticas.

## Instalación

Requiere Python 3.10–3.12 (3.13 también funciona si hay ruedas de PyTorch para
tu plataforma).

```bash
git clone https://github.com/pxjacomex/bass-separator.git
cd bass-separator
python -m venv .venv && source .venv/bin/activate     # en Windows: .venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # o la variante CUDA
pip install -e .
# opcional: interfaz web
pip install -e ".[ui]"
```

Con GPU NVIDIA instala la variante CUDA de PyTorch (ver pytorch.org); el
proceso es 10-30× más rápido. En CPU, un tema de 3 minutos tarda del orden de
1-3 minutos con `htdemucs_ft` (solo sub-modelo de bajo).

## Uso

```bash
# descarga y verificación de los pesos (opcional, se hace sola la primera vez)
bass-separator download-models

# caso típico: grabación antigua de big band
bass-separator separate "Count Basie - Jumpin at the Woodside.wav" -o salida

# varios archivos, más calidad (TTA), promediando dos modelos
bass-separator separate *.wav -o salida --shifts 2 --model htdemucs_ft,mdx_extra

# grabación moderna, solo realce DSP sin capa resintetizada
bass-separator separate tema.wav --profile modern --mode enhance

# solo la línea resintetizada (contrabajo "nuevo" tocando la misma línea)
bass-separator separate tema.wav --mode resynth

# informe rápido del archivo (mono, ancho de banda, zumbido, graves)
bass-separator analyze tema.wav

# interfaz web local
bass-separator ui
```

Salida (en la carpeta `-o`, por defecto `salida/`):

| archivo | contenido |
|---|---|
| `<nombre>_contrabajo.wav` | contrabajo aislado y realzado (24 bits, misma frecuencia que la entrada) |
| `<nombre>_contrabajo.mid` | línea de bajo transcrita (General MIDI *Acoustic Bass*) |
| `<nombre>_contrabajo_notas.json` | notas (inicio, fin, nota, f0 real, velocidad) + metadatos del proceso |
| `<nombre>_contrabajo_sin_realce.wav` | con `--keep-raw`: el stem de Demucs sin procesar |
| `<nombre>_drums.wav`, `_other.wav`, `_vocals.wav` | con `--keep-stems`: el resto de fuentes |

### Modos y perfiles

| opción | valores | descripción |
|---|---|---|
| `--profile` | `vintage` (defecto), `modern` | ajusta de-click, reducción de ruido, reconstrucción de graves, EQ |
| `--mode` | `hybrid` (defecto), `enhance`, `resynth`, `raw` | realce + capa resintetizada / solo realce / solo resíntesis / stem sin tocar |
| `--resynth-mix` | 0..1 | nivel de la capa resintetizada (0.35) |
| `--sub-gain` | 0..2 | cuánto se reconstruye la fundamental perdida (1.0) |
| `--denoise` | 0..1 | puerta espectral sobre el stem (0.7) |
| `--harmonic-strength` | 0..1 | atenuación de lo que no es armónico de la nota (0.8) |
| `--bleed-reduction` | 0..1 | reducción del colado de batería (0.6) |
| `--quantize-pitch` | | afina la capa resintetizada a semitonos exactos |
| `--pitch-method` | `salience`, `pyin` | seguidor de f0 (el propio o el de librosa) |
| `--backend` | `demucs`, `dsp` | `dsp` no usa red neuronal: solo para pruebas o sin conexión |

Consejos para 78 rpm: si el resultado suena "hueco", sube `--sub-gain 1.5`;
si quedan golpes de bombo, sube `--bleed-reduction 0.9`; si la línea de bajo
transcrita tiene errores de octava en pasajes muy ruidosos, prueba
`--pitch-method pyin` o acota `--fmax 250`.

## Uso desde Python

```python
from bass_separator import Settings
from bass_separator.config import apply_preset
from bass_separator.pipeline import process_file

settings = apply_preset(Settings(), "vintage")
res = process_file("tema.wav", "salida", settings)
print(res.output_path, res.midi_path, res.n_notes)
```

## Pruebas

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Las pruebas usan una big band sintética (línea de walking bass con modelo físico,
piano, batería y metales) degradada a 78 rpm, y comprueban entre otras cosas que
el seguidor de f0 supera el 95 % de precisión con la fundamental ausente, que la
transcripción recupera más del 80 % de las notas con colado y ruido fuertes, que
el realce mejora la SI-SDR del stem y que el pipeline completo funciona en los
cuatro modos. La integración con Demucs se valida con el modelo de prueba del
propio paquete (sin descargar pesos).

Generar una demo: `bass-separator demo -o demo_bigband.wav --duration 20`.

## Limitaciones

* Demucs se entrenó con música moderna en estéreo; en 78 rpm mono el stem de
  bajo puede llevar colado de bombo, piano o trombón. La cadena de realce lo
  reduce, pero en pasajes de tutti con el bajo enterrado el resultado depende de
  la grabación.
* La transcripción asume una sola nota de bajo a la vez (walking bass, no
  acordes) y el rango E1–E4.
* El procesamiento mantiene el audio completo en memoria: ~1 GB por cada
  3 minutos de audio.

## Licencia

MIT. Demucs © Meta Platforms, Inc. (MIT).
