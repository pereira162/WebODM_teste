# CAPÍTULO 3: GESTÃO DE MEMÓRIA E TILING

## Parte 1: Estratégias de Memória e Point Cloud Processing

---

## 3.1 VISÃO GERAL DO GERENCIAMENTO DE MEMÓRIA

O processamento fotogramétrico é extremamente intensivo em memória. Um dataset típico de 100 imagens 4K pode requerer 16-32GB de RAM no processamento tradicional. O ODM implementa várias estratégias para gerenciar essa demanda.

### 3.1.1 Consumo de Memória por Estágio

```
┌────────────────────────────────────────────────────────────────────────┐
│                    PERFIL DE MEMÓRIA DO ODM                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  RAM (GB)                                                              │
│    32 ┤                         ████████                               │
│    28 ┤                        ██████████                              │
│    24 ┤                       ████████████                             │
│    20 ┤                      ██████████████                            │
│    16 ┤          ████       ████████████████        ████              │
│    12 ┤         ██████     ██████████████████      ██████             │
│     8 ┤████    ████████   ████████████████████    ████████    ████   │
│     4 ┤██████████████████████████████████████████████████████████████│
│     0 ┼────────────────────────────────────────────────────────────── │
│       Load  Features Match   SfM   OpenMVS  Mesh  Texture  DEM Ortho  │
│                                                                        │
│  Legenda: ████ = RAM utilizada em cada estágio                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1.2 Requisitos de Memória por Dataset

| Tamanho Dataset | Imagens | RAM Mínima | RAM Recomendada | Disco Temp |
|-----------------|---------|------------|-----------------|------------|
| Pequeno | 20-50 | 4 GB | 8 GB | 10 GB |
| Médio | 50-200 | 8 GB | 16 GB | 50 GB |
| Grande | 200-500 | 16 GB | 32 GB | 100 GB |
| Muito Grande | 500-1000 | 32 GB | 64 GB | 200 GB |
| Massivo | 1000+ | 64 GB+ | 128 GB+ | 500 GB+ |

---

## 3.2 ESTRATÉGIAS DE GERENCIAMENTO DE MEMÓRIA

### 3.2.1 GDAL Cache Management

O ODM utiliza extensivamente o cache do GDAL para operações raster:

```python
# Configuração do cache GDAL (opendm/system.py)
import os
from osgeo import gdal

def configure_gdal_cache():
    """
    Configurar cache GDAL baseado na memória disponível
    """
    # Obter memória total do sistema
    total_memory_mb = get_total_memory_mb()
    
    # Alocar 25-50% para GDAL
    gdal_cache_mb = int(total_memory_mb * 0.25)
    
    # Limitar a um máximo razoável
    gdal_cache_mb = min(gdal_cache_mb, 8192)  # Max 8GB
    
    # Configurar
    gdal.SetCacheMax(gdal_cache_mb * 1024 * 1024)  # Em bytes
    
    # Configurações adicionais
    gdal.SetConfigOption('GDAL_CACHEMAX', str(gdal_cache_mb))
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.SetConfigOption('VSI_CACHE', 'TRUE')
    gdal.SetConfigOption('VSI_CACHE_SIZE', str(gdal_cache_mb * 1024 * 1024))
    
    return gdal_cache_mb
```

### 3.2.2 Streaming de Imagens

Para evitar carregar todas as imagens na memória:

```python
# Processamento em streaming (app/models/task.py)
def process_images_streaming(images_path, process_func, batch_size=10):
    """
    Processar imagens em lotes para economizar memória
    """
    images = list_images(images_path)
    total = len(images)
    
    for i in range(0, total, batch_size):
        batch = images[i:i+batch_size]
        
        for image_path in batch:
            # Carregar imagem
            img = Image.open(image_path)
            
            # Processar
            result = process_func(img)
            
            # Liberar memória explicitamente
            img.close()
            del img
            
            yield result
        
        # Forçar garbage collection após cada lote
        import gc
        gc.collect()
```

### 3.2.3 Memory-Mapped Files

Para grandes arrays, usar arquivos mapeados em memória:

```python
import numpy as np
import tempfile

def create_memory_mapped_array(shape, dtype=np.float32):
    """
    Criar array usando arquivo temporário memory-mapped
    Permite trabalhar com arrays maiores que a RAM
    """
    # Criar arquivo temporário
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mmap')
    
    # Criar array mapeado
    arr = np.memmap(
        temp_file.name,
        dtype=dtype,
        mode='w+',
        shape=shape
    )
    
    return arr, temp_file.name

# Uso para point cloud grande
def process_large_pointcloud(las_file, output_file):
    """
    Processar point cloud grande usando memory-mapping
    """
    import laspy
    
    # Ler header para obter dimensões
    with laspy.open(las_file) as f:
        header = f.header
        point_count = header.point_count
    
    # Criar arrays mapeados
    points, points_file = create_memory_mapped_array((point_count, 3))
    colors, colors_file = create_memory_mapped_array((point_count, 3), np.uint8)
    
    try:
        # Ler em chunks
        chunk_size = 1_000_000
        with laspy.open(las_file) as f:
            for i, chunk in enumerate(f.chunk_iterator(chunk_size)):
                start = i * chunk_size
                end = start + len(chunk.points)
                
                points[start:end, 0] = chunk.x
                points[start:end, 1] = chunk.y
                points[start:end, 2] = chunk.z
                
                if hasattr(chunk, 'red'):
                    colors[start:end, 0] = chunk.red
                    colors[start:end, 1] = chunk.green
                    colors[start:end, 2] = chunk.blue
        
        # Processar (filtrar, classificar, etc.)
        # ...
        
    finally:
        # Cleanup
        del points, colors
        os.unlink(points_file)
        os.unlink(colors_file)
```

---

## 3.3 PDAL - POINT DATA ABSTRACTION LIBRARY

O PDAL é a biblioteca central para processamento de point clouds no ODM.

### 3.3.1 Arquitetura PDAL

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PDAL PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│  │ READERS │ -> │ FILTERS │ -> │ FILTERS │ -> │ WRITERS │          │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘          │
│       │              │              │              │                 │
│       ▼              ▼              ▼              ▼                 │
│  - readers.las  - filters.smrf  - filters.outlier  - writers.las   │
│  - readers.ply  - filters.elm   - filters.range    - writers.gdal  │
│  - readers.xyz  - filters.csf   - filters.crop     - writers.ply   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3.2 Pipeline JSON Típico para ODM

```python
# opendm/point_cloud.py
def create_pdal_pipeline(input_las, output_las, config):
    """
    Criar pipeline PDAL para processamento de point cloud
    """
    pipeline = {
        "pipeline": [
            # 1. Leitura
            {
                "type": "readers.las",
                "filename": input_las
            },
            
            # 2. Filtro estatístico de outliers
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": config.get('meanK', 8),
                "multiplier": config.get('standard_deviation', 2.5)
            },
            
            # 3. Classificação de ground points (SMRF)
            {
                "type": "filters.smrf",
                "scalar": config.get('smrf_scalar', 1.25),
                "slope": config.get('smrf_slope', 0.15),
                "threshold": config.get('smrf_threshold', 0.5),
                "window": config.get('smrf_window', 18.0),
                "cell": 1.0
            },
            
            # 4. Filtro de altura (opcional)
            {
                "type": "filters.hag_dem",
                "raster": config.get('dem_file')
            } if config.get('dem_file') else None,
            
            # 5. Escrita
            {
                "type": "writers.las",
                "filename": output_las,
                "compression": "laszip"
            }
        ]
    }
    
    # Remover None
    pipeline["pipeline"] = [p for p in pipeline["pipeline"] if p]
    
    return pipeline
```

### 3.3.3 Algoritmo SMRF (Simple Morphological Filter)

O SMRF é usado para classificar pontos de ground (terreno):

```
SMRF ALGORITHM:
═══════════════

1. CRIAR GRADE DE ELEVAÇÃO MÍNIMA
   ┌───┬───┬───┬───┐
   │2.1│2.3│2.5│2.4│  <- Elevação mínima em cada célula
   ├───┼───┼───┼───┤
   │2.0│2.2│2.8│2.6│
   └───┴───┴───┴───┘
   
2. APLICAR OPENING MORFOLÓGICO
   - Erosão seguida de dilatação
   - Tamanho do kernel baseado em 'window'
   
3. CALCULAR GRADIENTE
   - Diferença entre superfície original e opened
   
4. CLASSIFICAR PONTOS
   - Se gradiente < threshold: GROUND
   - Se gradiente >= threshold: NON-GROUND
   
5. ITERAR COM JANELAS CRESCENTES
   - Começar com janela pequena
   - Aumentar até 'window' máximo
```

```python
# Pseudocódigo SMRF simplificado
def smrf_classify(points, cell_size, slope, threshold, window_sizes):
    """
    Simple Morphological Filter para classificação de terreno
    """
    # Criar grade de elevação mínima
    grid = create_min_elevation_grid(points, cell_size)
    
    # Superfície de referência inicial
    surface = grid.copy()
    
    for window in window_sizes:
        # Calcular elevação máxima permitida baseada no slope
        max_height = window * cell_size * slope
        
        # Opening morfológico
        kernel = create_circular_kernel(window)
        opened = morphological_open(surface, kernel)
        
        # Identificar pontos de ground
        for point in points:
            cell = get_cell(point, cell_size)
            diff = point.z - opened[cell]
            
            if diff < max_height:
                point.classification = GROUND
            else:
                point.classification = NON_GROUND
        
        # Atualizar superfície para próxima iteração
        surface = update_surface(points, cell_size)
    
    return points
```

---

## 3.4 DECIMAÇÃO E SUBSAMPLING

### 3.4.1 Estratégias de Decimação

```python
# opendm/point_cloud.py
def decimate_pointcloud(input_las, output_las, decimation_factor):
    """
    Reduzir número de pontos mantendo distribuição espacial
    """
    if decimation_factor <= 1:
        return input_las  # Sem decimação
    
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": input_las},
            
            # Método 1: Decimação simples (cada N pontos)
            {
                "type": "filters.decimation",
                "step": decimation_factor
            },
            
            {"type": "writers.las", "filename": output_las}
        ]
    }
    
    # OU Método 2: Voxel grid (distribuição espacial uniforme)
    pipeline_voxel = {
        "pipeline": [
            {"type": "readers.las", "filename": input_las},
            {
                "type": "filters.voxeldownsize",
                "cell": 0.1,  # Tamanho do voxel em metros
                "mode": "center"  # Ponto central de cada voxel
            },
            {"type": "writers.las", "filename": output_las}
        ]
    }
    
    pdal.Pipeline(json.dumps(pipeline)).execute()
    return output_las
```

### 3.4.2 Voxel Grid Downsampling

```
VOXEL GRID DOWNSAMPLING:
════════════════════════

Antes:                        Depois:
┌─────────────────┐           ┌─────────────────┐
│  •  • •  •  •   │           │                 │
│ • •• • ••  •  • │           │    •    •    •  │
│  •• •• • • •    │    =>     │                 │
│ • •  •  • • • • │           │    •    •    •  │
│  • •  • •  •    │           │                 │
└─────────────────┘           └─────────────────┘
   100.000 pts                    4 pts/célula

- Divide espaço em voxels de tamanho fixo
- Mantém 1 ponto por voxel (centróide ou mais próximo)
- Preserva distribuição espacial
- Redução controlada pelo tamanho do voxel
```

---

## 3.5 SPLIT-MERGE PARA DATASETS GRANDES

O ODM implementa uma estratégia de Split-Merge para processar datasets que excedem a memória disponível.

### 3.5.1 Arquitetura Split-Merge

```
SPLIT-MERGE WORKFLOW:
════════════════════

                    ┌────────────────────┐
                    │   DATASET GRANDE   │
                    │   (1000+ imagens)  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │    SPLIT STAGE     │
                    │  (clustering GPS)  │
                    └─────────┬──────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
     ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
     │ SUBMODEL 1│      │ SUBMODEL 2│      │ SUBMODEL N│
     │ (~300 img)│      │ (~300 img)│      │ (~300 img)│
     └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
           │                  │                  │
     ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
     │ PROCESS   │      │ PROCESS   │      │ PROCESS   │
     │ (full ODM)│      │ (full ODM)│      │ (full ODM)│
     └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    MERGE STAGE     │
                    │  (unify outputs)   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   OUTPUTS FINAIS   │
                    │ (ortho, DEM, etc)  │
                    └────────────────────┘
```

### 3.5.2 Algoritmo de Clustering para Split

```python
# stages/splitmerge.py
def split_dataset(images, config):
    """
    Dividir dataset em submodelos baseado em GPS
    """
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Extrair coordenadas GPS
    gps_coords = []
    for img in images:
        if img.latitude and img.longitude:
            gps_coords.append([img.latitude, img.longitude])
    
    gps_coords = np.array(gps_coords)
    
    # Determinar número de submodelos
    avg_images_per_submodel = config.get('split_avg', 200)
    n_submodels = max(1, len(images) // avg_images_per_submodel)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_submodels, random_state=42)
    labels = kmeans.fit_predict(gps_coords)
    
    # Criar submodelos com overlap
    submodels = []
    overlap_distance = config.get('split_overlap', 150)  # metros
    
    for i in range(n_submodels):
        # Imagens do cluster
        cluster_images = [img for j, img in enumerate(images) if labels[j] == i]
        
        # Adicionar imagens vizinhas para overlap
        center = kmeans.cluster_centers_[i]
        for j, img in enumerate(images):
            if labels[j] != i:
                dist = haversine_distance(
                    center[0], center[1],
                    img.latitude, img.longitude
                )
                if dist < overlap_distance:
                    cluster_images.append(img)
        
        submodels.append({
            'id': i,
            'images': cluster_images,
            'center': center
        })
    
    return submodels
```

### 3.5.3 Merge de Outputs

```python
# stages/splitmerge.py
def merge_outputs(submodels, output_dir, config):
    """
    Unificar outputs dos submodelos
    """
    # 1. Merge point clouds
    point_clouds = [
        os.path.join(sm['path'], 'odm_georeferencing', 'odm_georeferenced_model.laz')
        for sm in submodels
    ]
    
    merged_pc = merge_point_clouds(point_clouds, output_dir)
    
    # 2. Merge orthophotos
    orthophotos = [
        os.path.join(sm['path'], 'odm_orthophoto', 'odm_orthophoto.tif')
        for sm in submodels
    ]
    
    merged_ortho = merge_orthophotos(orthophotos, output_dir, config)
    
    # 3. Merge DEMs
    dems = [
        os.path.join(sm['path'], 'odm_dem', 'dsm.tif')
        for sm in submodels
    ]
    
    merged_dem = merge_dems(dems, output_dir, config)
    
    return {
        'point_cloud': merged_pc,
        'orthophoto': merged_ortho,
        'dem': merged_dem
    }

def merge_point_clouds(input_files, output_dir):
    """
    Merge múltiplos point clouds usando PDAL
    """
    readers = [{"type": "readers.las", "filename": f} for f in input_files]
    
    pipeline = {
        "pipeline": readers + [
            {
                "type": "filters.merge"  # Combina todos os inputs
            },
            {
                "type": "writers.las",
                "filename": os.path.join(output_dir, "merged_pointcloud.laz"),
                "compression": "laszip"
            }
        ]
    }
    
    pdal.Pipeline(json.dumps(pipeline)).execute()

def merge_orthophotos(input_files, output_dir, config):
    """
    Merge orthophotos com feathering nas bordas
    """
    from osgeo import gdal
    
    output_file = os.path.join(output_dir, "merged_orthophoto.tif")
    
    # Usar GDAL Virtual Raster para merge
    vrt_options = gdal.BuildVRTOptions(
        resolution='highest',
        separate=False
    )
    
    vrt = gdal.BuildVRT(
        '/vsimem/merged.vrt',
        input_files,
        options=vrt_options
    )
    
    # Converter VRT para GeoTIFF
    translate_options = gdal.TranslateOptions(
        format='GTiff',
        creationOptions=[
            'COMPRESS=JPEG',
            'JPEG_QUALITY=90',
            'TILED=YES',
            'BLOCKXSIZE=512',
            'BLOCKYSIZE=512'
        ]
    )
    
    gdal.Translate(output_file, vrt, options=translate_options)
    
    return output_file
```

