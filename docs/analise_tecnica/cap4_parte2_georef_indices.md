# CAPÍTULO 4: FUNDAMENTOS MATEMÁTICOS DE AGRIMENSURA

## Parte 2: Georreferenciamento e Índices Vegetativos

---

## 4.7 SISTEMAS DE COORDENADAS

### 4.7.1 WGS84 (World Geodetic System 1984)

O WGS84 é o sistema de referência global usado por GPS.

**Parâmetros do elipsóide WGS84:**
- Semi-eixo maior: $a = 6,378,137.0$ m
- Achatamento: $f = 1/298.257223563$
- Semi-eixo menor: $b = a(1-f) = 6,356,752.3142$ m
- Excentricidade: $e^2 = 2f - f^2 = 0.00669437999014$

```python
# Constantes WGS84
WGS84 = {
    'a': 6378137.0,                  # Semi-eixo maior (m)
    'b': 6356752.3142,               # Semi-eixo menor (m)
    'f': 1/298.257223563,            # Achatamento
    'e2': 0.00669437999014,          # Excentricidade ao quadrado
    'e_prime2': 0.00673949674228     # Segunda excentricidade ao quadrado
}
```

### 4.7.2 Conversão Geodésica → ECEF

Converter latitude/longitude/altitude para coordenadas cartesianas ECEF:

$$X = (N + h) \cos\phi \cos\lambda$$
$$Y = (N + h) \cos\phi \sin\lambda$$
$$Z = (N(1-e^2) + h) \sin\phi$$

Onde $N$ é o raio de curvatura na vertical:

$$N = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$$

```python
def geodetic_to_ecef(lat, lon, alt):
    """
    Converter coordenadas geodésicas (WGS84) para ECEF
    
    Args:
        lat: latitude em graus
        lon: longitude em graus
        alt: altitude elipsoidal em metros
    
    Returns:
        X, Y, Z em metros
    """
    import math
    
    # Converter para radianos
    phi = math.radians(lat)
    lam = math.radians(lon)
    
    # Raio de curvatura
    N = WGS84['a'] / math.sqrt(1 - WGS84['e2'] * math.sin(phi)**2)
    
    # Coordenadas ECEF
    X = (N + alt) * math.cos(phi) * math.cos(lam)
    Y = (N + alt) * math.cos(phi) * math.sin(lam)
    Z = (N * (1 - WGS84['e2']) + alt) * math.sin(phi)
    
    return X, Y, Z
```

### 4.7.3 Projeção UTM (Universal Transverse Mercator)

A projeção UTM divide a Terra em 60 zonas de 6° de longitude cada.

**Fórmulas de conversão (simplificadas):**

```python
def latlon_to_utm(lat, lon):
    """
    Converter lat/lon (WGS84) para UTM
    """
    import pyproj
    
    # Determinar zona UTM
    zone = int((lon + 180) / 6) + 1
    
    # Determinar hemisfério
    hemisphere = 'north' if lat >= 0 else 'south'
    
    # EPSG code
    if hemisphere == 'north':
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    
    # Criar transformador
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",  # WGS84
        f"EPSG:{epsg}",
        always_xy=True
    )
    
    # Converter
    easting, northing = transformer.transform(lon, lat)
    
    return easting, northing, zone, hemisphere

def utm_to_latlon(easting, northing, zone, hemisphere='north'):
    """
    Converter UTM para lat/lon (WGS84)
    """
    import pyproj
    
    # EPSG code
    if hemisphere == 'north':
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    
    # Criar transformador
    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{epsg}",
        "EPSG:4326",
        always_xy=True
    )
    
    # Converter
    lon, lat = transformer.transform(easting, northing)
    
    return lat, lon
```

---

## 4.8 GEORREFERENCIAMENTO

### 4.8.1 Ground Control Points (GCPs)

GCPs são pontos com coordenadas conhecidas usados para melhorar a precisão do georreferenciamento.

**Formato de arquivo GCP (ODM):**

```
EPSG:32632
574370.123 4913721.456 125.789 1245.5 1678.2 DJI_0001.JPG gcp1
574380.234 4913731.567 124.567 2456.7 1234.5 DJI_0001.JPG gcp2
574380.234 4913731.567 124.567 3567.8 2345.6 DJI_0002.JPG gcp2
...
```

Formato: `X Y Z pixel_x pixel_y image_name gcp_label`

```python
# app/classes/gcp.py (simplificado)
class GCPFile:
    def __init__(self, gcp_path):
        self.gcp_path = gcp_path
        self.entries = []
        self.srs = None
        self.read()
    
    def read(self):
        with open(self.gcp_path, 'r') as f:
            lines = f.readlines()
        
        # Primeira linha: SRS
        self.srs = lines[0].strip()
        
        # Demais linhas: GCPs
        for line in lines[1:]:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 6:
                    self.entries.append(GCPEntry(
                        x=float(parts[0]),
                        y=float(parts[1]),
                        z=float(parts[2]),
                        px=float(parts[3]),
                        py=float(parts[4]),
                        filename=parts[5],
                        label=parts[6] if len(parts) > 6 else ''
                    ))
```

### 4.8.2 Transformação de Similaridade 3D (7 parâmetros)

A transformação de Helmert conecta o sistema local ao sistema de coordenadas global:

$$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix}_{global} = 
s \cdot R \cdot \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}_{local} + 
\begin{bmatrix} t_X \\ t_Y \\ t_Z \end{bmatrix}$$

Onde:
- $s$ = fator de escala
- $R$ = matriz de rotação 3x3 (3 parâmetros: $\omega, \phi, \kappa$)
- $t$ = vetor de translação (3 parâmetros)

Total: 7 parâmetros (transformação de Helmert)

```python
def helmert_transformation(local_points, global_points):
    """
    Calcular transformação de Helmert 3D (7 parâmetros)
    
    Args:
        local_points: Nx3 array de pontos no sistema local
        global_points: Nx3 array de pontos correspondentes no sistema global
    
    Returns:
        scale, rotation_matrix, translation_vector
    """
    from scipy.optimize import least_squares
    
    def residuals(params, local_pts, global_pts):
        scale = params[0]
        omega, phi, kappa = params[1:4]
        tx, ty, tz = params[4:7]
        
        # Matriz de rotação
        R = euler_to_rotation_matrix(omega, phi, kappa)
        
        errors = []
        for local, glob in zip(local_pts, global_pts):
            transformed = scale * R @ local + np.array([tx, ty, tz])
            errors.extend(transformed - glob)
        
        return np.array(errors)
    
    # Valores iniciais
    params_init = [1.0, 0, 0, 0, 0, 0, 0]  # scale, omega, phi, kappa, tx, ty, tz
    
    # Otimizar
    result = least_squares(
        residuals,
        params_init,
        args=(local_points, global_points)
    )
    
    # Extrair parâmetros
    scale = result.x[0]
    omega, phi, kappa = result.x[1:4]
    translation = result.x[4:7]
    rotation = euler_to_rotation_matrix(omega, phi, kappa)
    
    return scale, rotation, translation
```

### 4.8.3 Erro Médio Quadrático (RMSE)

Avaliar qualidade do georreferenciamento:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (predicted_i - observed_i)^2}$$

```python
def calculate_gcp_rmse(gcps, reconstruction):
    """
    Calcular RMSE dos GCPs após georreferenciamento
    """
    errors_x = []
    errors_y = []
    errors_z = []
    
    for gcp in gcps:
        # Coordenadas previstas pelo modelo
        predicted = reconstruction.get_point(gcp.label)
        
        # Coordenadas observadas (do GCP)
        observed = np.array([gcp.x, gcp.y, gcp.z])
        
        # Erros
        errors_x.append(predicted[0] - observed[0])
        errors_y.append(predicted[1] - observed[1])
        errors_z.append(predicted[2] - observed[2])
    
    rmse_x = np.sqrt(np.mean(np.array(errors_x)**2))
    rmse_y = np.sqrt(np.mean(np.array(errors_y)**2))
    rmse_z = np.sqrt(np.mean(np.array(errors_z)**2))
    rmse_3d = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)
    
    return {
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_z': rmse_z,
        'rmse_3d': rmse_3d
    }
```

---

## 4.9 GERAÇÃO DE DEM

### 4.9.1 Interpolação IDW (Inverse Distance Weighting)

O IDW interpola valores baseado na distância aos pontos conhecidos:

$$Z(x,y) = \frac{\sum_{i=1}^{n} w_i \cdot Z_i}{\sum_{i=1}^{n} w_i}$$

Onde:

$$w_i = \frac{1}{d_i^p}$$

- $d_i$ = distância ao ponto $i$
- $p$ = potência (geralmente 2)

```python
def idw_interpolation(x, y, points, values, power=2):
    """
    Interpolação IDW
    
    Args:
        x, y: Coordenadas do ponto a interpolar
        points: Nx2 array de coordenadas dos pontos conhecidos
        values: N array de valores Z conhecidos
        power: Expoente de distância
    
    Returns:
        Valor interpolado
    """
    distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
    
    # Evitar divisão por zero
    mask = distances < 1e-10
    if np.any(mask):
        return values[mask][0]
    
    weights = 1.0 / (distances ** power)
    return np.sum(weights * values) / np.sum(weights)

def create_dem_grid(points, values, resolution, bounds):
    """
    Criar grid DEM via IDW
    """
    xmin, ymin, xmax, ymax = bounds
    
    # Criar grid
    x_coords = np.arange(xmin, xmax, resolution)
    y_coords = np.arange(ymin, ymax, resolution)
    
    dem = np.zeros((len(y_coords), len(x_coords)))
    
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            dem[i, j] = idw_interpolation(x, y, points, values)
    
    return dem, x_coords, y_coords
```

### 4.9.2 Classificação de Terreno (DTM vs DSM)

```
DSM (Digital Surface Model):
- Inclui tudo: vegetação, edificações
- Usa elevação máxima em cada célula

DTM (Digital Terrain Model):
- Apenas terreno nu
- Requer classificação de ground points

┌─────────────────────────────────────────────────────┐
│                                                     │
│         ▲▲▲                     ▓▓▓▓▓             │
│        ▲▲▲▲▲                   ▓▓▓▓▓▓▓            │
│       ▲▲▲▲▲▲▲       DSM ────▶ ▓▓▓▓▓▓▓▓▓          │
│      ▲▲▲▲▲▲▲▲▲     (árvore)   ▓▓▓▓▓▓▓▓▓▓▓        │
│     ══════════════════════════════════════════    │ ← DTM (terreno)
│    /                                          \   │
│   /              TERRENO                       \  │
│  /                                              \ │
└─────────────────────────────────────────────────┘
```

---

## 4.10 ÍNDICES VEGETATIVOS

Os índices vegetativos são fórmulas aplicadas a bandas espectrais para análise de vegetação.

### 4.10.1 NDVI (Normalized Difference Vegetation Index)

$$NDVI = \frac{NIR - RED}{NIR + RED}$$

- Valores: -1 a +1
- Vegetação saudável: 0.2 a 0.9
- Solo nu: -0.1 a 0.1
- Água: < -0.1

### 4.10.2 Outros Índices (app/api/formulas.py)

```python
# Índices implementados no ODM/WebODM
VEGETATION_INDICES = {
    'NDVI': {
        'expr': '(N - R) / (N + R)',
        'range': (-1, 1),
        'bands': ['N', 'R']  # NIR, Red
    },
    'NDRE': {
        'expr': '(N - Re) / (N + Re)',  
        'range': (-1, 1),
        'bands': ['N', 'Re']  # NIR, RedEdge
    },
    'NDWI': {
        'expr': '(G - N) / (G + N)',
        'range': (-1, 1),
        'bands': ['G', 'N']  # Green, NIR
    },
    'GNDVI': {
        'expr': '(N - G) / (N + G)',
        'range': (-1, 1),
        'bands': ['N', 'G']  # NIR, Green
    },
    'EVI': {
        'expr': '2.5 * (N - R) / (N + 6*R - 7.5*B + 1)',
        'range': (-1, 1),
        'bands': ['N', 'R', 'B']
    },
    'SAVI': {
        'expr': '(1.5 * (N - R)) / (N + R + 0.5)',
        'range': (-1, 1),
        'bands': ['N', 'R']
    },
    'VARI': {
        'expr': '(G - R) / (G + R - B)',
        'range': (-1, 1),
        'bands': ['G', 'R', 'B']  # Para RGB comum
    },
    'GLI': {
        'expr': '((G * 2) - R - B) / ((G * 2) + R + B)',
        'range': (-1, 1),
        'bands': ['G', 'R', 'B']
    },
    'EXGREEN': {
        'expr': '(2 * G) - (R + B)',
        'range': None,
        'bands': ['G', 'R', 'B']
    }
}
```

### 4.10.3 Cálculo de Índices (numexpr)

```python
import numexpr as ne
import numpy as np

def calculate_index(bands, formula, band_mapping):
    """
    Calcular índice vegetativo usando numexpr
    
    Args:
        bands: Dict com arrays de cada banda {'R': array, 'G': array, ...}
        formula: Expressão do índice (ex: '(N - R) / (N + R)')
        band_mapping: Mapeamento de símbolos para índices de banda
    
    Returns:
        Array com índice calculado
    """
    # Preparar variáveis para numexpr
    local_dict = {}
    
    # Converter nomes de banda para variáveis
    expr = formula
    for symbol, band_data in bands.items():
        var_name = f'b{band_mapping[symbol]}'
        local_dict[var_name] = band_data.astype(np.float32)
        expr = expr.replace(symbol, var_name)
    
    # Calcular usando numexpr (rápido para arrays grandes)
    result = ne.evaluate(expr, local_dict=local_dict)
    
    # Tratar divisões por zero
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return result

# Exemplo de uso
def compute_ndvi(nir_band, red_band):
    """
    Calcular NDVI
    """
    bands = {'N': nir_band, 'R': red_band}
    return calculate_index(bands, '(N - R) / (N + R)', {'N': 1, 'R': 2})
```

---

## 4.11 FÓRMULAS DE ÁREA E VOLUME

### 4.11.1 Cálculo de Área em Coordenadas UTM

```python
def calculate_polygon_area_utm(coordinates):
    """
    Calcular área de polígono em coordenadas UTM (metros²)
    Fórmula de Shoelace
    """
    n = len(coordinates)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += coordinates[i][0] * coordinates[j][1]
        area -= coordinates[j][0] * coordinates[i][1]
    
    return abs(area) / 2.0

def calculate_area_hectares(coordinates_utm):
    """
    Calcular área em hectares
    """
    area_m2 = calculate_polygon_area_utm(coordinates_utm)
    return area_m2 / 10000.0
```

### 4.11.2 Cálculo de Volume (Cut & Fill)

```python
def calculate_cut_fill_volume(dem1, dem2, cell_size):
    """
    Calcular volume de corte e aterro entre dois DEMs
    
    Args:
        dem1: DEM de referência (antes)
        dem2: DEM atual (depois)
        cell_size: Tamanho da célula em metros
    
    Returns:
        cut_volume, fill_volume em metros cúbicos
    """
    diff = dem2 - dem1
    
    # Volume por célula = diferença * área da célula
    cell_area = cell_size ** 2
    
    # Corte (onde dem2 > dem1)
    cut_cells = diff[diff > 0]
    cut_volume = np.sum(cut_cells) * cell_area
    
    # Aterro (onde dem2 < dem1)
    fill_cells = diff[diff < 0]
    fill_volume = np.sum(np.abs(fill_cells)) * cell_area
    
    return cut_volume, fill_volume
```

---

## 4.12 CONCLUSÕES DO CAPÍTULO 4

### 4.12.1 Fórmulas Essenciais para Android

| Operação | Fórmula/Algoritmo | Complexidade |
|----------|-------------------|--------------|
| Projeção Perspectiva | $u = f \cdot X/Z + c_x$ | O(1) |
| Distorção Brown | $r^2 + k_1r^4 + k_2r^6$ | O(1) |
| Triangulação DLT | SVD de 4x4 | O(1) |
| Bundle Adjustment | Levenberg-Marquardt | O(n³) |
| NDVI | $(NIR-R)/(NIR+R)$ | O(pixels) |
| IDW | $\sum w_i Z_i / \sum w_i$ | O(n) |
| Área (Shoelace) | $\sum x_i y_{i+1} - x_{i+1}y_i$ | O(n) |

### 4.12.2 Bibliotecas para Implementação Android

```kotlin
// Recomendações de bibliotecas
val mathLibraries = mapOf(
    "Álgebra Linear" to "org.ejml:ejml-simple:0.41",
    "Projeções" to "org.locationtech.proj4j:proj4j:1.2.2",
    "Geometria" to "org.locationtech.jts:jts-core:1.19.0",
    "Imagens" to "org.opencv:opencv-android:4.5.0",
    "Arrays" to "nativeArray via JNI"
)
```

### 4.12.3 Precisão Numérica

| Operação | Float32 | Float64 | Recomendação |
|----------|---------|---------|--------------|
| Pixels/Features | ✅ | - | Float32 OK |
| Coordenadas Locais | ✅ | ✅ | Float32 OK |
| Coordenadas UTM | - | ✅ | Float64 necessário |
| Lat/Lon | - | ✅ | Float64 necessário |
| Bundle Adjustment | - | ✅ | Float64 necessário |
| Índices (NDVI) | ✅ | - | Float32 OK |

