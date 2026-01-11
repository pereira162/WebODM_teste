# CAPÍTULO 2: MOTOR SfM E MODO FAST ORTHOPHOTO

## Parte 1: Pipeline de Processamento e OpenSfM

---

## 2.1 VISÃO GERAL DO PIPELINE DE PROCESSAMENTO

O ODM implementa um pipeline de processamento fotogramétrico completo, organizado em **estágios sequenciais**. O arquivo `stages/odm_app.py` define a classe `ODMApp` que orquestra todo o fluxo.

### 2.1.1 Diagrama do Pipeline Completo

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ODM COMPLETO                              │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: ODMLoadDatasetStage                                             │
│ ├── Carregar imagens do diretório de entrada                             │
│ ├── Parse EXIF metadata (GPS, câmera, timestamp)                         │
│ ├── Validar formatos suportados (JPEG, TIFF, DNG)                        │
│ └── Criar objetos ODM_Photo para cada imagem                             │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: ODMSplitStage (opcional, para datasets grandes)                 │
│ ├── Dividir dataset em submodelos                                        │
│ ├── Clustering baseado em GPS                                            │
│ └── Cada submodelo processa independentemente                            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: ODMOpenSfMStage (Structure from Motion)                         │
│ ├── Extração de features (SIFT/HAHOG/DSPSIFT)                            │
│ ├── Matching de features entre imagens                                    │
│ ├── Reconstrução esparsa (triangulação)                                  │
│ ├── Bundle Adjustment (otimização câmeras + pontos)                      │
│ ├── Undistort das imagens                                                │
│ └── Exportar reconstruction.json + reconstruction.ply                    │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │ FAST ORTHOPHOTO     │         │ FULL PIPELINE       │
        │ (--fast-orthophoto) │         │ (padrão)            │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    │                               ▼
                    │               ┌──────────────────────────────────────┐
                    │               │ STAGE 4: ODMOpenMVSStage             │
                    │               │ ├── Dense point cloud generation     │
                    │               │ ├── Depth map estimation             │
                    │               │ └── Point cloud fusion               │
                    │               └──────────────────────────────────────┘
                    │                               │
                    │                               ▼
                    │               ┌──────────────────────────────────────┐
                    │               │ STAGE 5: ODMFilterPoints             │
                    │               │ ├── Statistical outlier removal      │
                    │               │ └── Noise filtering                  │
                    │               └──────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: ODMMeshingStage                                                 │
│ ├── Poisson Surface Reconstruction                                       │
│ ├── Mesh simplification                                                  │
│ └── Output: mesh.ply                                                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: ODMMvsTexStage                                                  │
│ ├── Texture mapping para mesh                                            │
│ ├── Seam leveling                                                        │
│ └── Output: textured_model.obj                                           │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 8: ODMGeoreferencingStage                                          │
│ ├── Aplicar transformação geográfica (UTM/WGS84)                         │
│ ├── Usar GCPs se disponíveis                                             │
│ └── Output: georeferenced_model.laz                                      │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 9: ODMDEMStage                                                     │
│ ├── Criar DSM (Digital Surface Model)                                    │
│ ├── Criar DTM (Digital Terrain Model) - classificação ground             │
│ └── Interpolação e gap-filling                                           │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 10: ODMOrthoPhotoStage                                             │
│ ├── Ortorretificação usando DEM                                          │
│ ├── Mosaicking de imagens                                                │
│ ├── Color balancing                                                      │
│ └── Output: odm_orthophoto.tif (GeoTIFF COG)                             │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 11: ODMMergeStage (se split foi usado)                             │
│ ├── Merge point clouds dos submodelos                                    │
│ ├── Merge orthophotos                                                    │
│ └── Unificar outputs                                                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 12: ODMReport + ODMPostProcess                                     │
│ ├── Gerar report.pdf                                                     │
│ ├── Criar tiles para visualização                                        │
│ └── Cleanup temporários                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 OPENSFM - STRUCTURE FROM MOTION ENGINE

O OpenSfM é o coração do processamento fotogramétrico no ODM. Ele transforma imagens 2D em uma reconstrução 3D esparsa.

### 2.2.1 Arquitetura OpenSfM

```
OpenSfM/
├── opensfm/
│   ├── commands/               # CLI commands
│   │   ├── extract_metadata.py
│   │   ├── detect_features.py
│   │   ├── match_features.py
│   │   ├── create_tracks.py
│   │   ├── reconstruct.py
│   │   ├── bundle.py
│   │   └── undistort.py
│   ├── features.py             # Feature detection/description
│   ├── matching.py             # Feature matching
│   ├── reconstruction.py       # SfM reconstruction
│   ├── types.py                # Data structures
│   ├── config.py               # Configuration
│   └── csfm/                   # C++ backend (Pybind11)
│       ├── src/
│       │   ├── bundle.h        # Bundle adjustment
│       │   ├── triangulation.h
│       │   └── geometry.h
│       └── CMakeLists.txt
├── bin/
│   └── opensfm                 # CLI entry point
└── data/
    └── sensor_data.json        # Camera database
```

### 2.2.2 Configuração OpenSfM (config.yaml)

O ODM gera dinamicamente o arquivo `config.yaml` para OpenSfM:

```yaml
# Configuração gerada por ODM (opendm/osfm.py)

# === FEATURE EXTRACTION ===
feature_type: dspsift           # dspsift|sift|hahog|akaze|orb
feature_process_size: 2048      # Tamanho máximo de imagem para features
feature_min_frames: 4000        # Mínimo de features por imagem
feature_quality: high           # ultra|high|medium|low|lowest

# === FEATURE MATCHING ===
matcher_type: flann             # flann|bruteforce|bow
matcher_neighbors: 0            # 0 = usar GPS para matching
matching_gps_distance: 150      # Distância GPS máxima (metros)
matching_gps_neighbors: 8       # Vizinhos GPS para matching
matcher_order: 0                # Matching por ordem de filename

# === SFM RECONSTRUCTION ===
sfm_algorithm: incremental      # incremental|triangulation|planar
retriangulation_ratio: 1.25
bundle_outlier_filtering_type: AUTO

# === BUNDLE ADJUSTMENT ===
use_hybrid_bundle_adjustment: false  # true para datasets grandes
bundle_use_gps: true
bundle_use_gcp: false

# === CAMERA ===
camera_lens: auto               # auto|brown|fisheye|spherical
optimize_camera_parameters: true
use_fixed_camera_params: false

# === UNDISTORTION ===
undistorted_image_max_size: 4000
depthmap_resolution: 640        # Para OpenMVS
```

### 2.2.3 Mapeamento de Parâmetros ODM → OpenSfM

| Parâmetro ODM | config.yaml OpenSfM | Descrição |
|---------------|---------------------|-----------|
| `--feature-type` | `feature_type` | Tipo de detector |
| `--feature-quality` | `feature_process_size` | Resolução features |
| `--min-num-features` | `feature_min_frames` | Min features/imagem |
| `--matcher-type` | `matcher_type` | Algoritmo matching |
| `--matcher-neighbors` | `matching_gps_neighbors` | Vizinhos GPS |
| `--sfm-algorithm` | `sfm_algorithm` | Tipo de SfM |
| `--camera-lens` | `camera_lens` | Modelo de lente |
| `--use-hybrid-bundle-adjustment` | `use_hybrid_bundle_adjustment` | BA híbrido |

---

## 2.3 FEATURE EXTRACTION

### 2.3.1 Tipos de Features Suportados

```python
# Configuração de feature types (OpenSfM)
FEATURE_TYPES = {
    'sift': {
        'detector': cv2.SIFT_create,
        'descriptor_size': 128,
        'float_descriptors': True,
        'pros': 'Robustez, invariância a escala/rotação',
        'cons': 'Lento, patented até 2020'
    },
    'dspsift': {
        'detector': 'Domain Size Pooling SIFT',
        'descriptor_size': 128,
        'float_descriptors': True,
        'pros': 'Mais rápido que SIFT, boa qualidade',
        'cons': 'Ainda relativamente lento'
    },
    'hahog': {
        'detector': 'Histogram of Averaged Oriented Gradients',
        'descriptor_size': 128,
        'float_descriptors': True,
        'pros': 'Muito rápido',
        'cons': 'Menos robusto que SIFT'
    },
    'akaze': {
        'detector': cv2.AKAZE_create,
        'descriptor_size': 486,
        'float_descriptors': False,  # Binary
        'pros': 'Rápido, sem patentes',
        'cons': 'Menos features que SIFT'
    },
    'orb': {
        'detector': cv2.ORB_create,
        'descriptor_size': 256,
        'float_descriptors': False,  # Binary
        'pros': 'Muito rápido, sem patentes',
        'cons': 'Menos robusto'
    },
    'sift_gpu': {
        'detector': 'CUDA SIFT',
        'descriptor_size': 128,
        'float_descriptors': True,
        'pros': 'Muito rápido com GPU',
        'cons': 'Requer CUDA'
    }
}
```

### 2.3.2 Fluxo de Extração de Features

```python
# Pseudocódigo do fluxo de extração
def extract_features(image_path, config):
    # 1. Carregar imagem
    image = load_image(image_path)
    
    # 2. Resize para feature_process_size
    if max(image.shape) > config['feature_process_size']:
        scale = config['feature_process_size'] / max(image.shape)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    # 3. Converter para grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 4. Detectar keypoints
    detector = get_detector(config['feature_type'])
    keypoints = detector.detect(gray)
    
    # 5. Filtrar por qualidade
    keypoints = filter_by_quality(keypoints, config['feature_min_frames'])
    
    # 6. Computar descriptors
    keypoints, descriptors = detector.compute(gray, keypoints)
    
    # 7. Converter coordenadas para escala original
    if scale != 1.0:
        for kp in keypoints:
            kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
    
    return keypoints, descriptors
```

### 2.3.3 Qualidade vs Performance

| Quality Level | `feature_process_size` | Features Típicos | Tempo Relativo |
|---------------|------------------------|------------------|----------------|
| `ultra` | 4096 | 15000+ | 4x |
| `high` | 2048 | 10000 | 2x |
| `medium` | 1024 | 6000 | 1x (baseline) |
| `low` | 512 | 3000 | 0.5x |
| `lowest` | 256 | 1500 | 0.25x |

---

## 2.4 FEATURE MATCHING

### 2.4.1 Estratégias de Matching

```python
# Tipos de matching disponíveis

# 1. FLANN (Fast Library for Approximate Nearest Neighbors)
class FLANNMatcher:
    """
    - Usa árvores KD para busca aproximada
    - Melhor para descriptors float (SIFT, HAHOG)
    - Parâmetros: trees=4, checks=32
    """
    def match(self, desc1, desc2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=32)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Ratio test (Lowe's)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        return good_matches

# 2. Brute Force
class BruteForceMatcher:
    """
    - Matching exato
    - Melhor para descriptors binários (ORB, AKAZE)
    - Mais lento, mais preciso
    """
    def match(self, desc1, desc2, binary=False):
        if binary:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = bf.match(desc1, desc2)
        return sorted(matches, key=lambda x: x.distance)

# 3. Bag of Words (BOW)
class BOWMatcher:
    """
    - Usa vocabulário visual pré-treinado
    - Muito rápido para datasets grandes
    - Força uso de HAHOG features
    """
    def match(self, images, vocabulary):
        # Quantizar features para visual words
        # Comparar histogramas de palavras visuais
        pass
```

### 2.4.2 Seleção de Pares para Matching

```python
def select_matching_pairs(images, config):
    pairs = []
    
    if config['matcher_neighbors'] > 0:
        # Matching baseado em GPS
        for i, img1 in enumerate(images):
            if img1.gps is None:
                continue
            
            # Encontrar vizinhos por distância GPS
            neighbors = find_gps_neighbors(
                img1, images,
                max_distance=config['matching_gps_distance'],
                max_neighbors=config['matching_gps_neighbors']
            )
            
            for j in neighbors:
                pairs.append((i, j))
    
    elif config['matcher_order'] > 0:
        # Matching por ordem sequencial
        for i in range(len(images) - 1):
            for j in range(i + 1, min(i + config['matcher_order'] + 1, len(images))):
                pairs.append((i, j))
    
    else:
        # Matching exaustivo (todas combinações)
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                pairs.append((i, j))
    
    return pairs
```

---

## 2.5 RECONSTRUÇÃO SFM

### 2.5.1 Algoritmo Incremental SfM

```python
def incremental_sfm(tracks, images, config):
    """
    Algoritmo incremental SfM (padrão do OpenSfM)
    
    1. Selecionar par inicial com boa baseline
    2. Estimar pose relativa
    3. Triangular pontos 3D
    4. Para cada nova imagem:
       a. PnP (Perspective-n-Point) para estimar pose
       b. Triangular novos pontos
       c. Bundle adjustment local
    5. Bundle adjustment global final
    """
    
    # Selecionar par inicial
    initial_pair = select_initial_pair(tracks, images)
    img1, img2 = initial_pair
    
    # Estimar Essential Matrix
    E, mask = cv2.findEssentialMat(
        points1, points2, 
        camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    
    # Decompor em R, t
    _, R, t, mask = cv2.recoverPose(E, points1, points2, camera_matrix)
    
    # Criar reconstrução inicial
    reconstruction = Reconstruction()
    reconstruction.add_camera(img1, np.eye(4))  # Origem
    reconstruction.add_camera(img2, compose_pose(R, t))
    
    # Triangular pontos iniciais
    points_3d = triangulate_points(
        reconstruction.cameras[img1],
        reconstruction.cameras[img2],
        matches
    )
    reconstruction.add_points(points_3d)
    
    # Adicionar imagens incrementalmente
    remaining = set(images) - {img1, img2}
    
    while remaining:
        # Encontrar próxima melhor imagem
        next_image = find_next_image(reconstruction, remaining, tracks)
        
        if next_image is None:
            break
        
        # Estimar pose via PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points,  # Pontos 3D conhecidos
            image_points,   # Correspondências 2D
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if success:
            reconstruction.add_camera(next_image, pose_from_rvec_tvec(rvec, tvec))
            
            # Triangular novos pontos
            new_points = triangulate_new_points(reconstruction, next_image, tracks)
            reconstruction.add_points(new_points)
            
            # Bundle adjustment local
            if config['use_hybrid_bundle_adjustment']:
                bundle_adjust_local(reconstruction, next_image, window=10)
            
            remaining.remove(next_image)
    
    # Bundle adjustment global
    bundle_adjust_global(reconstruction)
    
    return reconstruction
```

### 2.5.2 Triangulação de Pontos

```python
def triangulate_point(P1, P2, x1, x2):
    """
    Triangulação DLT (Direct Linear Transform)
    
    Dado:
    - P1, P2: Matrizes de projeção 3x4 das câmeras
    - x1, x2: Pontos correspondentes em coordenadas homogêneas
    
    Retorna:
    - X: Ponto 3D em coordenadas homogêneas
    """
    # Construir matriz A para sistema Ax = 0
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1]
    ])
    
    # Resolver via SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # Última linha de V^T
    
    # Converter para coordenadas euclidianas
    X = X[:3] / X[3]
    
    return X
```

---

## 2.6 OUTPUTS DO OPENSFM

### 2.6.1 Arquivos Gerados

```
project/opensfm/
├── config.yaml                 # Configuração usada
├── camera_models.json          # Parâmetros de câmera estimados
├── camera_models_overrides.json # Overrides manuais
├── exif/                       # Metadados EXIF extraídos
│   └── *.json
├── features/                   # Features detectadas
│   └── *.npz                   # (keypoints, descriptors)
├── matches/                    # Matches entre pares
│   └── *.matches.npz
├── tracks.csv                  # Track graph
├── reconstruction.json         # Reconstrução completa
├── reconstruction.meshed.json  # Com mesh
├── undistorted/                # Imagens undistorted
│   ├── images/
│   └── reconstruction.json
└── depthmaps/                  # Depth maps (para OpenMVS)
    └── *.exr
```

### 2.6.2 Estrutura do reconstruction.json

```json
{
  "cameras": {
    "v2 unknown unknown 4000 3000 brown 0.8": {
      "projection_type": "brown",
      "width": 4000,
      "height": 3000,
      "focal": 0.8,
      "k1": -0.05,
      "k2": 0.01,
      "p1": 0.0,
      "p2": 0.0,
      "k3": 0.0
    }
  },
  "shots": {
    "DJI_0001.JPG": {
      "camera": "v2 unknown unknown 4000 3000 brown 0.8",
      "rotation": [0.1, 0.2, 0.3],
      "translation": [10.5, 20.3, 30.1],
      "gps_position": [lat, lon, alt],
      "gps_dop": 5.0,
      "orientation": 1
    }
  },
  "points": {
    "point_id_001": {
      "coordinates": [x, y, z],
      "color": [r, g, b]
    }
  }
}
```

