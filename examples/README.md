# Ejemplos

```bash
# 1) Generar una "big band" sintética degradada a 78 rpm (20 s) y su referencia limpia
bass-separator demo -o demo_bigband.wav --duration 20

# 2) Separar y realzar el contrabajo (descarga los pesos oficiales de Demucs la primera vez)
bass-separator separate demo_bigband.wav -o salida

# 3) Comparar con la referencia:
#    salida/demo_bigband_contrabajo.wav  vs  demo_bigband_contrabajo_referencia.wav
```

`salida/demo_bigband_contrabajo_notas.json` contiene la línea de bajo transcrita
(instante, duración, nota, afinación real y dinámica) y
`salida/demo_bigband_contrabajo.mid` la misma línea en MIDI.
