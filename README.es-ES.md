# PAPR: Representación de Puntos con Atención de Proximidad (NeurIPS 2023 Spotlight 🤩)
[Yanshu Zhang*](https://zvict.github.io/), [Shichong Peng*](https://sites.google.com/view/niopeng/home), [Alireza Moazeni](https://amoazeni75.github.io/), [Ke Li](https://www.sfu.ca/~keli/) (* indica contribución igualitaria)<br>

<img src="./images/SFU_AI.png" height=100px /><img src="images/APEX_lab.png" height=120px />

[Sitios del proyecto](https://zvict.github.io/papr)
 | [Artículo](https://arxiv.org/abs/2307.11086) |
Contacto principal: [Yanshu Zhang](https://zvict.github.io/)

Proximity Attention Point Rendering (PAPR) es un nuevo método que permite la síntesis conjunta de vistas noveles y la reconstrucción 3D. Aprende simultáneamente, desde cero, una representación precisa de la nube de puntos de la superficie de la escena y una red neuronal basada en atención que renderiza dicha nube de puntos desde vistas novedosas.

<!-- <img src="./images/pipeline.png" /> -->

[![Presentación en NeurIPS 2023](https://github.com/zvict/papr/blob/main/images/papr_video_cover.png)](https://youtu.be/1atBGH_pDHY)

## Cita BibTeX
 <strong>PAPR: Proximity Attention Point Rendering</strong>.  &nbsp;&nbsp;&nbsp; 
```
@inproceedings{zhang2023papr,
    title={PAPR: Proximity Attention Point Rendering},
    author={Yanshu Zhang and Shichong Peng and Seyed Alireza Moazenipourasil and Ke Li},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}
```

## Instalación
```
git clone git@github.com:zvict/papr.git   # o 'git clone https://github.com/zvict/papr'
cd papr
conda env create -f papr.yml
conda activate papr
```
O utiliza un entorno virtual con `python=3.9`:
```
python -m venv path/to/<env_name>
source path/to/<env_name>/bin/activate
pip install -r requirements.txt
```

## Preparación de datos

La estructura del dataset esperada en el directorio de origen es la siguiente:
```
papr
├── data
│   ├── nerf_synthetic
│   │   ├── chair
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   │   ├── transforms_train.json
│   │   │   ├── transforms_val.json
│   │   │   ├── transforms_test.json
│   │   ├── ... 
│   ├── tanks_temples
│   │   ├── Barn
│   │   │   ├── pose
│   │   │   ├── rgb
│   │   │   ├── intrinsics.txt
│   │   ├── ... 
```
### Dataset NeRF Synthetic
Descarga el dataset NeRF Synthetic desde [aquí](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) y colócalo en `data/nerf_synthetic/`


### Tanks & Temples
Descarga [Tanks&Temples](https://www.tanksandtemples.org/) desde [aquí](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip) y colócalo en:
`data/tanks_temples/`

### Uso de tus propios datos
Puedes consultar este [issue](https://github.com/zvict/papr/issues/3#issuecomment-1907260683) para obtener instrucciones sobre cómo preparar el dataset.

Debes crear un nuevo archivo de configuración para tu propio dataset y colocarlo en `configs`. El parámetro `dataset.type` en el archivo de configuración especifica el tipo de dataset. Si tu dataset tiene el mismo formato que el dataset NeRF Synthetic, puedes establecer directamente `dataset.type` a `"synthetic"`. De lo contrario, debes implementar tu propio script en Python para cargar el dataset en la carpeta `dataset`, y agregarlo a la función `load_meta_data` en `dataset/utils.py`.

La mayoría de los parámetros predeterminados en `configs/default.yml` son generales y pueden ser utilizados para tu propio dataset. Puedes especificar los parámetros específicos de tu dataset en el archivo de configuración que creaste, de manera similar a los archivos de configuración para el dataset NeRF Synthetic y el dataset Tanks and Temples.

## Descripción general

El código tiene dos componentes principales: la parte de carga de datos en `dataset/` y los modelos en `models/`. La clase `PAPR` en `models/model.py` define nuestro modelo principal. Todas las configuraciones están en `configs/`, y `configs/demo.yml` es un archivo de configuración de demostración que incluye comentarios sobre los argumentos importantes.

Proporcionamos un notebook `demo.ipynb` que demuestra cómo entrenar y probar el modelo con el archivo de configuración de demostración, así como cómo utilizar el control de exposición para mejorar la calidad de renderizado en escenas del mundo real capturadas con autoexposición activada.

## Entrenamiento
```
python train.py --opt configs/nerfsyn/chair.yml
```

## Finetuning con [cIMLE](https://arxiv.org/abs/2004.03590) (opcional)

Para escenas del mundo real donde la exposición puede variar entre vistas, podemos introducir un código latente adicional como entrada en nuestro modelo y finetunear el modelo utilizando una técnica llamada [Conditional Implicit Maximum Likelihood Estimation (cIMLE)](https://arxiv.org/abs/2004.03590) para controlar el nivel de exposición de la imagen renderizada, como se describe en la Sección 4.4 y el Apéndice A.8 del artículo. Se requiere un modelo preentrenado para finetunear con control de exposición, entrenando con `train.py` utilizando configuraciones predeterminadas. Proporcionamos un archivo de configuración de demostración para la escena Caterpillar del dataset Tanks and Temples en `configs/t2/Caterpillar_exposure_control.yml`.

Para finetunear un modelo preentrenado con control de exposición, ejecuta:
```
python exposure_control_finetune.py --opt configs/t2/Caterpillar_exposure_control.yml
```

## Evaluación
Para evaluar tu modelo entrenado sin finetuning para control de exposición, ejecuta:
```
python test.py --opt configs/nerfsyn/chair.yml
```
Esto te proporcionará imágenes renderizadas y métricas en el conjunto de prueba.

Con un modelo finetunado, puedes renderizar todas las vistas de prueba con un solo nivel de exposición aleatorio, ejecutando:
```
python test.py --opt configs/t2/Caterpillar_exposure_control.yml --exp
```
Para generar imágenes con diferentes niveles de exposición aleatorios para una sola vista, ejecuta:
```
python test.py --opt configs/t2/Caterpillar_exposure_control.yml --exp --random --view 0
```
Ten en cuenta que durante la prueba, la escala de los códigos latentes debería aumentarse para generar imágenes con exposiciones más diversas, por ejemplo:
```
python test.py --opt configs/t2/Caterpillar_exposure_control.yml --exp --random --view 0 --scale 8
```
Una vez que generes imágenes con diferentes niveles de exposición, puedes interpolar dos niveles de exposición seleccionados especificando su índice, por ejemplo:
```
python test.py --opt configs/t2/Caterpillar_exposure_control.yml --exp --intrp --view 0 --start_index 0 --end_index 1
```

## Modelos preentrenados

Proporcionamos modelos preentrenados en los datasets NeRF Synthetic y Tanks&Temples aquí (sin finetuning): [Google Drive](https://drive.google.com/drive/folders/1HSNlMu6Uup9o5hqi7T0hgDf63yR9W90s?usp=sharing). También proporcionamos un modelo preentrenado con control de exposición en la escena Caterpillar en Google Drive. Para cargar los modelos preentrenados, colócalos en `checkpoints/` y modifica el `test.load_path` en el archivo de configuración.

## Agradecimiento
Esta investigación fue posible en parte gracias al apoyo brindado por NSERC, el Grupo BC DRI y el Digital Research Alliance of Canada.
