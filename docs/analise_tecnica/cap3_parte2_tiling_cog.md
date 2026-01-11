# CAPÍTULO 3: GESTÃO DE MEMÓRIA E TILING

## Parte 2: Tiling System e COG

---

## 3.6 SISTEMA DE TILING

O ODM gera tiles para visualização eficiente de mapas grandes no WebODM. O sistema segue os padrões XYZ/TMS/WMTS.

### 3.6.1 Esquema de Tiles XYZ

```
TILE PYRAMID:
═════════════

Zoom 0:  1 tile (mundo inteiro)
         ┌───────┐
         │   0   │
         └───────┘

Zoom 1:  4 tiles (2x2)
         ┌───┬───┐
         │ 0 │ 1 │
         ├───┼───┤
         │ 2 │ 3 │
         └───┴───┘

Zoom 2:  16 tiles (4x4)
         ┌───┬───┬───┬───┐
         │ 0 │ 1 │ 2 │ 3 │
         ├───┼───┼───┼───┤
         │ 4 │ 5 │ 6 │ 7 │
         ├───┼───┼───┼───┤
         │ 8 │ 9 │10 │11 │
         ├───┼───┼───┼───┤
         │12 │13 │14 │15 │
         └───┴───┴───┴───┘

Zoom N:  2^N x 2^N tiles
         Cada tile: 256x256 ou 512x512 pixels

URL Pattern: /{z}/{x}/{y}.png
Exemplo: /15/16384/10922.png
```

### 3.6.2 Cálculo de Zoom Levels

```python
# opendm/tiles/tiler.py
def calculate_zoom_levels(raster_path, tile_size=256):
    """
    Calcular níveis de zoom apropriados para o raster
    """
    import rasterio
    
    with rasterio.open(raster_path) as src:
        # Resolução em metros/pixel
        res_x = abs(src.transform[0])
        res_y = abs(src.transform[4])
        resolution = (res_x + res_y) / 2
        
        # Bounds em coordenadas geográficas
        bounds = src.bounds
        width_meters = bounds.right - bounds.left
        height_meters = bounds.top - bounds.bottom
        
        # Resolução de um tile no zoom máximo
        # No zoom 18, cada tile cobre ~1.2m/pixel no equador
        METERS_PER_PIXEL_ZOOM_0 = 156543.03392804062  # Web Mercator
        
        # Zoom máximo baseado na resolução do raster
        max_zoom = int(np.log2(METERS_PER_PIXEL_ZOOM_0 / resolution))
        max_zoom = min(max_zoom, 22)  # Limitar a 22
        
        # Zoom mínimo baseado no tamanho da área
        min_tile_size = max(width_meters, height_meters) / tile_size
        min_zoom = int(np.log2(METERS_PER_PIXEL_ZOOM_0 / min_tile_size))
        min_zoom = max(min_zoom, 0)
        
        return min_zoom, max_zoom
```

### 3.6.3 Geração de Tiles (gdal2tiles)

```python
# opendm/tiles/gdal2tiles.py (adaptado)
def generate_tiles(input_raster, output_dir, config):
    """
    Gerar pirâmide de tiles a partir de raster
    """
    from osgeo import gdal
    
    # Determinar zoom levels
    min_zoom, max_zoom = calculate_zoom_levels(input_raster)
    
    # Opções de resampling
    RESAMPLING = {
        'nearest': gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'lanczos': gdal.GRA_Lanczos
    }
    
    # Abrir raster
    ds = gdal.Open(input_raster)
    
    for zoom in range(min_zoom, max_zoom + 1):
        # Calcular tiles necessários neste zoom
        tiles = get_tiles_for_zoom(ds, zoom)
        
        for tile_x, tile_y in tiles:
            # Calcular bounds do tile
            tile_bounds = get_tile_bounds(zoom, tile_x, tile_y)
            
            # Extrair região do raster
            tile_data = extract_tile(ds, tile_bounds, tile_size=256)
            
            # Salvar tile
            tile_path = os.path.join(
                output_dir,
                str(zoom),
                str(tile_x),
                f"{tile_y}.png"
            )
            
            os.makedirs(os.path.dirname(tile_path), exist_ok=True)
            save_tile_png(tile_data, tile_path)
    
    ds = None  # Fechar dataset

def get_tile_bounds(zoom, x, y):
    """
    Obter bounds geográficos de um tile (Web Mercator)
    """
    n = 2.0 ** zoom
    
    # Longitude (X)
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    
    # Latitude (Y) - fórmula Mercator
    lat_max_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    lat_min_rad = np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n)))
    
    lat_max = np.degrees(lat_max_rad)
    lat_min = np.degrees(lat_min_rad)
    
    return (lon_min, lat_min, lon_max, lat_max)
```

---

## 3.7 CLOUD OPTIMIZED GEOTIFF (COG)

O ODM gera orthophotos como COG para streaming eficiente.

### 3.7.1 Estrutura de um COG

```
COG (Cloud Optimized GeoTIFF):
══════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                         HEADER (IFD)                            │
│  - Metadados                                                    │
│  - Tags GeoTIFF                                                 │
│  - Offsets dos tiles                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   OVERVIEW 1    │ │   OVERVIEW 2    │ │   OVERVIEW N    │
│   (1/2 res)     │ │   (1/4 res)     │ │   (1/2^N res)   │
│   Tiles 512x512 │ │   Tiles 512x512 │ │   Tiles 512x512 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FULL RESOLUTION DATA                          │
│                                                                  │
│  ┌──────┬──────┬──────┬──────┐                                  │
│  │Tile 0│Tile 1│Tile 2│Tile 3│                                  │
│  ├──────┼──────┼──────┼──────┤                                  │
│  │Tile 4│Tile 5│Tile 6│Tile 7│   ...                            │
│  ├──────┼──────┼──────┼──────┤                                  │
│  │Tile 8│Tile 9│Tile10│Tile11│                                  │
│  └──────┴──────┴──────┴──────┘                                  │
│                                                                  │
│  Tiles internos: 512x512 pixels                                  │
│  Compressão: DEFLATE, JPEG, ou LZW                              │
└─────────────────────────────────────────────────────────────────┘

VANTAGENS:
- HTTP Range Requests para leitura parcial
- Sem necessidade de servidor de tiles dedicado
- Overviews embutidos para zoom rápido
- Compatível com GDAL/Rasterio
```

### 3.7.2 Criação de COG

```python
# app/cogeo.py
def assure_cogeo(input_tiff, output_tiff=None):
    """
    Converter GeoTIFF para Cloud Optimized GeoTIFF
    """
    from rio_cogeo.cogeo import cog_translate
    from rio_cogeo.profiles import cog_profiles
    
    if output_tiff is None:
        output_tiff = input_tiff
    
    # Verificar se já é COG
    if is_valid_cog(input_tiff):
        return input_tiff
    
    # Perfil de compressão
    profile = cog_profiles.get("deflate")  # ou "jpeg", "lzw"
    
    # Opções de criação
    config = {
        'GDAL_TIFF_INTERNAL_MASK': True,
        'GDAL_TIFF_OVR_BLOCKSIZE': 512
    }
    
    # Converter
    temp_output = output_tiff + '.tmp.tif'
    
    cog_translate(
        input_tiff,
        temp_output,
        profile,
        config=config,
        overview_level=5,  # Níveis de overview
        overview_resampling='lanczos',
        web_optimized=True,  # Otimizar para streaming
        in_memory=False
    )
    
    # Substituir original
    os.replace(temp_output, output_tiff)
    
    return output_tiff

def is_valid_cog(filepath):
    """
    Verificar se arquivo é um COG válido
    """
    from rio_cogeo.cogeo import cog_validate
    
    is_valid, errors, warnings = cog_validate(filepath)
    return is_valid
```

### 3.7.3 Opções de Compressão COG

| Compressão | Ratio | Velocidade | Uso Ideal |
|------------|-------|------------|-----------|
| DEFLATE | ~3:1 | Médio | Dados genéricos, lossless |
| LZW | ~2:1 | Rápido | Dados com muitos padrões |
| JPEG | ~10:1 | Rápido | Orthophotos RGB |
| JPEG2000 | ~15:1 | Lento | Arquivamento |
| ZSTD | ~4:1 | Muito rápido | Dados grandes |

```python
# Configurações por tipo de dado
COMPRESSION_PROFILES = {
    'orthophoto': {
        'compression': 'JPEG',
        'jpeg_quality': 90,
        'photometric': 'YCBCR'
    },
    'dem': {
        'compression': 'DEFLATE',
        'predictor': 2,  # Horizontal differencing
        'zlevel': 9
    },
    'multispectral': {
        'compression': 'LZW',
        'predictor': 2
    }
}
```

---

## 3.8 ENTWINE POINT TILES (EPT)

Para visualização de point clouds grandes, o ODM usa EPT (Entwine Point Tiles).

### 3.8.1 Estrutura EPT

```
EPT STRUCTURE:
══════════════

project/entwine_pointcloud/
├── ept.json                 # Metadata
├── ept-data/                # Tiles de pontos
│   ├── 0-0-0-0.laz         # Root tile
│   ├── 1-0-0-0.laz         # Level 1
│   ├── 1-0-0-1.laz
│   ├── 1-0-1-0.laz
│   ├── 1-0-1-1.laz
│   ├── 1-1-0-0.laz
│   └── ...
├── ept-hierarchy/           # Índice de tiles
│   └── 0-0-0-0.json
└── ept-sources/             # Metadados dos sources
    └── list.json

Naming: {depth}-{x}-{y}-{z}.laz
        Octree indexing
```

### 3.8.2 Geração de EPT

```python
# opendm/entwine.py
def build_ept(input_pointcloud, output_dir, config):
    """
    Gerar Entwine Point Tiles a partir de point cloud
    """
    import subprocess
    
    # Comando entwine
    cmd = [
        'entwine', 'build',
        '--input', input_pointcloud,
        '--output', output_dir,
        '--threads', str(config.get('threads', 4)),
        '--tmp', config.get('tmp_dir', '/tmp')
    ]
    
    # Opções adicionais
    if config.get('reprojection'):
        cmd.extend(['--srs', config['reprojection']])
    
    subprocess.run(cmd, check=True)
    
    return output_dir
```

### 3.8.3 Estrutura do ept.json

```json
{
  "bounds": [xmin, ymin, zmin, xmax, ymax, zmax],
  "boundsConforming": [xmin, ymin, zmin, xmax, ymax, zmax],
  "dataType": "laszip",
  "hierarchyType": "json",
  "points": 12345678,
  "schema": [
    {"name": "X", "type": "signed", "size": 4, "scale": 0.001, "offset": 1000},
    {"name": "Y", "type": "signed", "size": 4, "scale": 0.001, "offset": 2000},
    {"name": "Z", "type": "signed", "size": 4, "scale": 0.001, "offset": 100},
    {"name": "Intensity", "type": "unsigned", "size": 2},
    {"name": "Red", "type": "unsigned", "size": 2},
    {"name": "Green", "type": "unsigned", "size": 2},
    {"name": "Blue", "type": "unsigned", "size": 2},
    {"name": "Classification", "type": "unsigned", "size": 1}
  ],
  "span": 128,
  "srs": {
    "authority": "EPSG",
    "horizontal": "32632",
    "wkt": "..."
  },
  "version": "1.0.0"
}
```

---

## 3.9 ESTRATÉGIAS PARA ANDROID

### 3.9.1 Gestão de Memória em Android

```kotlin
// MemoryManager.kt
class PhotogrammetryMemoryManager(private val context: Context) {
    
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    private val memoryInfo = ActivityManager.MemoryInfo()
    
    fun getAvailableMemory(): Long {
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.availMem
    }
    
    fun getTotalMemory(): Long {
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.totalMem
    }
    
    fun isLowMemory(): Boolean {
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.lowMemory
    }
    
    fun calculateBatchSize(imageSize: Long): Int {
        val availableMb = getAvailableMemory() / (1024 * 1024)
        
        // Reservar 30% para sistema
        val usableMb = availableMb * 0.7
        
        // Cada imagem precisa de ~3x seu tamanho em memória durante processamento
        val imageMb = imageSize / (1024 * 1024) * 3
        
        return maxOf(1, (usableMb / imageMb).toInt())
    }
    
    companion object {
        // Thresholds para tablets
        const val MIN_MEMORY_MB = 2048L  // 2GB mínimo
        const val RECOMMENDED_MEMORY_MB = 4096L  // 4GB recomendado
        
        // Limites de processamento
        const val MAX_IMAGE_SIZE_LOW_MEM = 2048
        const val MAX_IMAGE_SIZE_NORMAL = 4096
        const val MAX_IMAGES_LOW_MEM = 30
        const val MAX_IMAGES_NORMAL = 100
    }
}
```

### 3.9.2 Processamento em Chunks

```kotlin
// ChunkedProcessor.kt
class ChunkedImageProcessor(
    private val memoryManager: PhotogrammetryMemoryManager,
    private val nativeLib: NativeSfMLib
) {
    suspend fun processImages(
        images: List<ImageFile>,
        onProgress: (Float) -> Unit
    ): ProcessingResult = withContext(Dispatchers.Default) {
        
        val batchSize = memoryManager.calculateBatchSize(images.first().sizeBytes)
        val batches = images.chunked(batchSize)
        
        val allFeatures = mutableListOf<FeatureSet>()
        
        batches.forEachIndexed { index, batch ->
            // Verificar memória antes de cada batch
            if (memoryManager.isLowMemory()) {
                // Forçar GC e aguardar
                System.gc()
                delay(1000)
            }
            
            // Processar batch
            val batchFeatures = batch.map { image ->
                val bitmap = loadAndResizeImage(image)
                try {
                    nativeLib.extractFeatures(bitmap)
                } finally {
                    bitmap.recycle()
                }
            }
            
            allFeatures.addAll(batchFeatures)
            
            // Atualizar progresso
            onProgress((index + 1).toFloat() / batches.size * 0.3f)
        }
        
        // Continuar com matching, etc.
        // ...
    }
    
    private suspend fun loadAndResizeImage(image: ImageFile): Bitmap {
        return withContext(Dispatchers.IO) {
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeFile(image.path, options)
            
            // Calcular sample size para fit em memória
            val maxSize = if (memoryManager.isLowMemory()) {
                MAX_IMAGE_SIZE_LOW_MEM
            } else {
                MAX_IMAGE_SIZE_NORMAL
            }
            
            options.inSampleSize = calculateInSampleSize(
                options.outWidth, options.outHeight, maxSize, maxSize
            )
            options.inJustDecodeBounds = false
            
            BitmapFactory.decodeFile(image.path, options)
        }
    }
}
```

### 3.9.3 Tiling On-Device

```kotlin
// LocalTileGenerator.kt
class LocalTileGenerator(private val context: Context) {
    
    suspend fun generateTiles(
        orthophotoFile: File,
        outputDir: File,
        minZoom: Int = 10,
        maxZoom: Int = 18
    ) = withContext(Dispatchers.IO) {
        
        // Usar GDAL Android ou implementação custom
        val raster = GdalRaster.open(orthophotoFile.path)
        
        try {
            for (zoom in minZoom..maxZoom) {
                val tilesAtZoom = calculateTilesForZoom(raster.bounds, zoom)
                
                tilesAtZoom.forEach { (x, y) ->
                    val tileBounds = getTileBounds(zoom, x, y)
                    val tileData = raster.readRegion(tileBounds, 256, 256)
                    
                    val tileFile = File(outputDir, "$zoom/$x/$y.png")
                    tileFile.parentFile?.mkdirs()
                    
                    saveTilePng(tileData, tileFile)
                }
            }
        } finally {
            raster.close()
        }
    }
}
```

---

## 3.10 CONCLUSÕES DO CAPÍTULO 3

### 3.10.1 Requisitos de Memória para Android

| Operação | RAM Mínima | RAM Recomendada | Estratégia |
|----------|------------|-----------------|------------|
| Feature Extraction | 1 GB | 2 GB | Processar 1 imagem por vez |
| Feature Matching | 2 GB | 4 GB | Limitar pares simultâneos |
| SfM Reconstruction | 2 GB | 4 GB | Incremental + local BA |
| Ortho Generation | 2 GB | 4 GB | Tile-based processing |
| **Total App** | **4 GB** | **6 GB** | Tablets modernos OK |

### 3.10.2 Limites Práticos para Tablets

```kotlin
// Configuração recomendada para tablets
object TabletLimits {
    // Tablets com 4GB RAM
    val LOW_END = ProcessingLimits(
        maxImages = 30,
        maxImageSize = 2048,
        featureQuality = "low",
        matcherNeighbors = 4,
        skipDenseReconstruction = true
    )
    
    // Tablets com 6-8GB RAM
    val MID_RANGE = ProcessingLimits(
        maxImages = 60,
        maxImageSize = 3000,
        featureQuality = "medium",
        matcherNeighbors = 6,
        skipDenseReconstruction = true
    )
    
    // Tablets com 12GB+ RAM
    val HIGH_END = ProcessingLimits(
        maxImages = 100,
        maxImageSize = 4096,
        featureQuality = "high",
        matcherNeighbors = 8,
        skipDenseReconstruction = false
    )
}
```

### 3.10.3 Otimizações Críticas

1. **Streaming de imagens** - nunca carregar todas na memória
2. **Resize agressivo** - max 2048px para processamento
3. **Memory-mapped files** - para arrays grandes
4. **Garbage collection** - forçar entre batches
5. **Native memory** - preferir alocação nativa (JNI)
6. **Tile-based output** - nunca criar raster gigante em memória

