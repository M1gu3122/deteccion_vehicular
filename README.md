# Deteccion vehicular Kia

Clasificacion de 11 modelos Kia, deteccion de vehiculos y placas, OCR y zonas de video.

## Instalar en otro equipo (Windows)

Instala Git y Python 3.12 de 64 bits. Las versiones de referencia son las del equipo original (Python 3.12.10); no se ha comprobado una instalacion limpia en otros sistemas.

```powershell
git clone https://github.com/M1gu3122/deteccion_vehicular.git
cd deteccion_vehicular
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verificar_instalacion.py
```

Los pesos del proyecto, exportaciones OpenVINO y `runs/auto_cls_v2/` estan incluidos en Git. No necesitas copiar `myvenv/` ni entrenar de nuevo. La primera lectura OCR necesita Internet para descargar sus dos modelos PaddleOCR; despues se reutiliza la cache local. Esta configuracion ejecuta la inferencia en CPU; no exige una GPU.

Para analizar una imagen o video propio:

```powershell
.\.venv\Scripts\python.exe analizar_vehiculo.py "C:\fotos\mi_kia.jpg" --salida resultados/prueba1 --mostrar
.\.venv\Scripts\python.exe analizar_vehiculo.py "C:\videos\mi_kia.mp4" --salida resultados/prueba2 --mostrar
```

Cada salida debe ser una carpeta nueva. Las fotos, videos y datasets no se incluyen: usa archivos de tu equipo. Para los descargadores instala tambien `requirements-descargas.txt`. Para entrenar o evaluar necesitas obtener el dataset por separado; los checkpoints y registros no sustituyen las imagenes de entrenamiento. `requeriments.txt` se mantiene como alias del archivo de dependencias correctamente escrito.

## Modelos necesarios

- `modelos/auto_cls_v2.pt`: clasificador propio de 11 modelos Kia. Su exportacion `auto_cls_v2_openvino_model/` acelera la inferencia.
- `modelos/best.pt`: detector independiente de placas (clase 0) y vehiculos (clase 1). Se conserva junto a `best_openvino_model/` porque el clasificador no localiza placas.
- `runs/auto_cls_v2/`: historial y checkpoints de tu entrenamiento. Se conserva para futuras consultas o recuperacion.

## Uso

### Analizador unificado: foto o video, modelo, color y placa

```powershell
python analizar_vehiculo.py "imagenes/mi_auto.jpg" --salida resultados/foto1 --mostrar
python analizar_vehiculo.py "videos/mi_video.mp4" --salida resultados/video1 --mostrar
```

No necesita una zona delimitada. Cada salida debe ser una carpeta nueva. Guarda `imagen.jpg` y `datos.json` para fotos, o `video.mp4` sin audio y `datos.jsonl` para videos. Pulsa Q para detener un video. Los registros son por fotograma, sin identificadores persistentes de seguimiento.

`--cada 5` analiza uno de cada cinco fotogramas; los restantes se guardan sin anotaciones. `--sin-ocr` permite probar deteccion, clasificacion y color sin inicializar PaddleOCR, pero no lee las placas. Sin esa opcion se usa el OCR existente; su primera inicializacion puede requerir descargar sus modelos.

La marca Kia es una suposicion del proyecto. El modelo reconoce solo sus 11 clases entrenadas. El color es una estimacion por bandas de carroceria, no segmentacion: sombras, fondo y reflejos pueden alterarlo. Gris/plateado se agrupan; `color.proporcion` representa la fraccion de pixeles del color dominante, no una probabilidad calibrada. La placa se vincula a un solo vehiculo que la contiene; las no asociadas aparecen por separado. Revisa visualmente los resultados.

Pruebas de color y asociacion: `python -m unittest discover -s pruebas -p test_analisis_vehiculo.py`.

El lector OCR filtra la matricula con el patron de tres letras y tres numeros (por ejemplo, `FSN819`), descartando leyendas como la ciudad. Otros formatos, lecturas incompletas o ambiguas se devuelven sin lectura. `texto_original` conserva el reconocimiento completo para diagnostico. Pruebas: `python -m unittest discover -s pruebas -p test_lector_ocr.py`.

Activa el entorno existente desde PowerShell:

```powershell
.\myvenv\Scripts\Activate.ps1
python main.py
python reconocer_auto_simple.py --imagen imagenes/auto2.jpg
python delimitar_zona.py
python video_placas.py
```

Configura las imagenes en `config.py` y el video en cada script. La zona debe corresponder al mismo video y resolucion. `clasificador_vehiculo.py` permite integrar la clasificacion en otros programas; `reconocer_auto.py` combina reconocimiento de vehiculo y placa.

## Descargar, preparar y entrenar

Los descargadores permanecen en `descargar imagenes/` y `descargar_amanti.py`. Consulta `--help` en el descargador elegido. Los que usan Serper reciben la clave por `SERPER_API_KEY` o `--api-key`; no la guardes en Git.

El ultimo entrenamiento utilizo `dataset_autos_v2/`, segun `runs/auto_cls_v2/args.yaml` y los metadatos del modelo. Este es el dataset que se conserva: 11 clases, 1996 imagenes entre train/ y val/. El script de limpieza elimina `dataset_autos/` y `dataset_autos_full/`. Para volver a preparar datos desde cero, primero descarga nuevamente las fuentes; la preparacion reconstruye el destino.

```powershell
python preparar_dataset_v2.py --help
python entrenar_clasificador_v2.py --nombre kia_nuevo --solo-comando
python entrenar_clasificador_v2.py --nombre kia_nuevo
python evaluar_modelo.py --modelo modelos/kia_nuevo.pt
```

El entrenamiento parte de tu checkpoint local. Para comenzar con una arquitectura sin pesos preentrenados, usa `--modelo yolo11n-cls.yaml`. Elige un nombre nuevo para conservar el modelo actual. Para activar otro resultado, actualiza las rutas en `clasificador_vehiculo.py`, `reconocer_auto.py` y `reconocer_auto_simple.py`; no reutilices una exportacion OpenVINO antigua.

## Limpieza

`limpiar_proyecto.ps1` muestra la lista de pesos genericos, versiones antiguas, diagnosticos obsoletos y datasets descartados. Aplica esa lista con `./limpiar_proyecto.ps1 -Eliminar`. Conserva `dataset_autos_v2/`, medios de entrada, el entorno virtual y los dos modelos necesarios. Los cambios son locales; no se publican automaticamente en GitHub.
