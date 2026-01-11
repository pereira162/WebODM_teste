# CAPÍTULO 2: MOTOR SfM E MODO FAST ORTHOPHOTO

## Parte 2: Bundle Adjustment e Fast Orthophoto Mode

---

## 2.7 BUNDLE ADJUSTMENT

Bundle Adjustment é o processo de otimização não-linear que refina simultaneamente os parâmetros de câmera e as posições dos pontos 3D para minimizar o erro de reprojeção.

### 2.7.1 Formulação Matemática

O Bundle Adjustment resolve o seguinte problema de otimização:

$$\min_{\{R_i, t_i\}, \{X_j\}, \{K\}} \sum_{i,j} \rho\left( \|x_{ij} - \pi(K, R_i, t_i, X_j)\|^2 \right)$$

Onde:
- $R_i, t_i$ = Rotação e translação da câmera $i$
- $X_j$ = Coordenadas 3D do ponto $j$
- $K$ = Matriz de parâmetros intrínsecos da câmera
- $x_{ij}$ = Observação 2D do ponto $j$ na imagem $i$
- $\pi(\cdot)$ = Função de projeção perspectiva
- $\rho(\cdot)$ = Função de perda robusta (Huber, Cauchy)

### 2.7.2 Função de Projeção Perspectiva

```python
def project_point(K, R, t, X):
    """
    Projetar ponto 3D X para coordenadas de imagem 2D
    
    K = [[fx, 0, cx],     # Matriz intrínseca
         [0, fy, cy],
         [0,  0,  1]]
    
    R = rotation matrix 3x3
    t = translation vector 3x1
    X = 3D point [X, Y, Z]
    """
    # Transformar para coordenadas de câmera
    X_cam = R @ X + t
    
    # Projeção perspectiva
    x_norm = X_cam[0] / X_cam[2]
    y_norm = X_cam[1] / X_cam[2]
    
    # Aplicar distorção (modelo Brown)
    r2 = x_norm**2 + y_norm**2
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    
    x_dist = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
    y_dist = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
    
    # Converter para pixels
    u = fx * x_dist + cx
    v = fy * y_dist + cy
    
    return np.array([u, v])
```

### 2.7.3 Implementação com Ceres Solver

```cpp
// Bundle adjustment usando Ceres Solver (C++)
struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,  // 6 params (rotation + translation)
                    const T* const point,   // 3 params (X, Y, Z)
                    const T* const focal,   // 1 param
                    T* residuals) const {
        // Rotação via Rodrigues
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        
        // Translação
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        
        // Projeção
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        
        // Focal length
        T predicted_x = focal[0] * xp;
        T predicted_y = focal[0] * yp;
        
        // Residuals
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        
        return true;
    }

    double observed_x, observed_y;
};

// Configuração do problema
ceres::Problem problem;

for (auto& observation : observations) {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3, 1>(
            new ReprojectionError(observation.x, observation.y));
    
    problem.AddResidualBlock(cost_function,
                             new ceres::HuberLoss(1.0),  // Robust loss
                             camera_params[observation.camera_id],
                             point_params[observation.point_id],
                             &focal_length);
}

// Solver options
ceres::Solver::Options options;
options.linear_solver_type = ceres::SPARSE_SCHUR;  // Exploits BA structure
options.num_threads = 8;
options.max_num_iterations = 100;

ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
```

### 2.7.4 Bundle Adjustment Híbrido

Para datasets grandes (1000+ imagens), o BA híbrido é mais eficiente:

```python
def hybrid_bundle_adjustment(reconstruction, config):
    """
    Bundle Adjustment Híbrido:
    - BA local após adicionar cada imagem
    - BA global a cada N imagens
    """
    local_ba_images = 10      # Janela para BA local
    global_ba_interval = 100  # Intervalo para BA global
    
    for i, shot in enumerate(reconstruction.shots):
        # BA local: otimizar apenas imagens próximas
        local_shots = get_neighboring_shots(shot, reconstruction, local_ba_images)
        local_points = get_points_in_shots(local_shots, reconstruction)
        
        run_bundle_adjustment(
            cameras=local_shots,
            points=local_points,
            fix_cameras=get_fixed_cameras(local_shots)  # Fixar câmeras antigas
        )
        
        # BA global periódico
        if i > 0 and i % global_ba_interval == 0:
            run_bundle_adjustment(
                cameras=reconstruction.all_shots(),
                points=reconstruction.all_points(),
                fix_cameras=[]
            )
    
    # BA global final
    run_bundle_adjustment(
        cameras=reconstruction.all_shots(),
        points=reconstruction.all_points(),
        fix_cameras=[]
    )
```

---

## 2.8 MODO FAST ORTHOPHOTO

O modo `--fast-orthophoto` é a chave para criar um app de "Fast Stitching". Ele **pula** a reconstrução densa (OpenMVS) e gera ortofoto diretamente da reconstrução esparsa.

### 2.8.1 Pipeline Fast vs Full

```
FULL PIPELINE:
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│   OpenSfM  │ -> │  OpenMVS   │ -> │  Meshing   │ -> │  Ortophoto │
│   (sparse) │    │  (dense)   │    │  (Poisson) │    │            │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
    ~30%              ~50%              ~10%              ~10%
    tempo             tempo             tempo             tempo

FAST ORTHOPHOTO:
┌────────────┐    ┌────────────────────────────────────┐
│   OpenSfM  │ -> │  Ortophoto (direto de sparse)      │
│   (sparse) │    │  usando reconstruction.ply         │
└────────────┘    └────────────────────────────────────┘
    ~60%                        ~40%
    tempo                       tempo

GANHO: ~3-5x mais rápido
```

### 2.8.2 Como Funciona o Fast Orthophoto

```python
def fast_orthophoto_pipeline(images, config):
    """
    Pipeline simplificado para Fast Orthophoto
    """
    # 1. Executar OpenSfM normalmente (até reconstruction)
    run_opensfm_stages([
        'extract_metadata',
        'detect_features',
        'match_features', 
        'create_tracks',
        'reconstruct',      # Gera reconstruction.json
        'export_ply',       # Gera reconstruction.ply (sparse)
        'undistort'         # Prepara imagens
    ])
    
    # 2. SKIP OpenMVS (não gerar dense point cloud)
    # Isso economiza ~50% do tempo de processamento
    
    # 3. Usar reconstruction.ply diretamente para criar DEM
    sparse_ply = 'opensfm/reconstruction.ply'
    
    # Criar DEM a partir de pontos esparsos
    dem = create_dem_from_sparse(
        sparse_ply,
        resolution=config['dem_resolution'],
        interpolation='idw'  # Inverse Distance Weighting
    )
    
    # 4. Gerar ortofoto usando DEM esparso
    orthophoto = create_orthophoto(
        images='opensfm/undistorted/images/',
        cameras='opensfm/reconstruction.json',
        dem=dem,
        resolution=config['orthophoto_resolution']
    )
    
    return orthophoto
```

### 2.8.3 Diferenças na Qualidade

| Aspecto | Full Pipeline | Fast Orthophoto |
|---------|---------------|-----------------|
| **Tempo** | 100% (baseline) | 20-30% |
| **Point Cloud** | Dense (~milhões pts) | Sparse (~milhares pts) |
| **DEM Quality** | Alta resolução | Interpolado, menos detalhe |
| **Ortho Quality** | Excelente | Boa (suficiente para muitos casos) |
| **3D Model** | Mesh texturizado | Não disponível |
| **Uso de RAM** | Alto (8-32 GB) | Moderado (4-8 GB) |
| **Caso de Uso** | Levantamentos precisos | Preview rápido, áreas grandes |

### 2.8.4 Parâmetros Otimizados para Fast Mode

```bash
# Configuração otimizada para máxima velocidade
docker run opendronemap/odm \
    --fast-orthophoto \
    --feature-type hahog \           # Mais rápido que SIFT
    --feature-quality low \          # Menos features
    --matcher-neighbors 4 \          # Menos pares de matching
    --min-num-features 4000 \        # Mínimo reduzido
    --orthophoto-resolution 10 \     # 10 cm/pixel (menos detalhe)
    --skip-3dmodel \                 # Pular modelo 3D
    --skip-report \                  # Pular relatório PDF
    --pc-ept \                       # Skip EPT (point cloud tiles)
    --max-concurrency 4              # Controlar uso de recursos
```

---

## 2.9 IMPLEMENTAÇÃO NO CÓDIGO ODM

### 2.9.1 Detecção de Fast Mode (stages/odm_app.py)

```python
# Trecho de stages/odm_app.py
class ODMApp:
    def __init__(self, args):
        # ...
        
        # Definir pipeline baseado em parâmetros
        if args.fast_orthophoto:
            # Pipeline reduzido
            self.pipeline = [
                ODMLoadDatasetStage,
                ODMOpenSfMStage,
                # SKIP: ODMOpenMVSStage
                # SKIP: ODMFilterPoints
                ODMMeshingStage,     # Mesh simplificado
                ODMGeoreferencingStage,
                ODMDEMStage,
                ODMOrthoPhotoStage,
                ODMPostProcess
            ]
        else:
            # Pipeline completo
            self.pipeline = [
                ODMLoadDatasetStage,
                ODMOpenSfMStage,
                ODMOpenMVSStage,     # Dense reconstruction
                ODMFilterPoints,
                ODMMeshingStage,
                ODMMvsTexStage,
                ODMGeoreferencingStage,
                ODMDEMStage,
                ODMOrthoPhotoStage,
                ODMReport,
                ODMPostProcess
            ]
```

### 2.9.2 Geração de DEM Esparso

```python
# opendm/dem/commands.py
def create_dem(input_point_cloud, output_dem, config):
    """
    Criar DEM a partir de point cloud (esparso ou denso)
    """
    # Usar PDAL para processar point cloud
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_point_cloud
            },
            # Classificar ground points (para DTM)
            {
                "type": "filters.smrf",
                "scalar": config.smrf_scalar,
                "slope": config.smrf_slope,
                "threshold": config.smrf_threshold,
                "window": config.smrf_window
            } if config.dtm else None,
            # Gerar raster
            {
                "type": "writers.gdal",
                "filename": output_dem,
                "resolution": config.dem_resolution,
                "output_type": "idw" if config.dtm else "max",
                "radius": config.dem_resolution * 2,
                "gdaldriver": "GTiff"
            }
        ]
    }
    
    # Remover None do pipeline
    pipeline["pipeline"] = [p for p in pipeline["pipeline"] if p]
    
    # Executar PDAL
    pdal.Pipeline(json.dumps(pipeline)).execute()
```

---

## 2.10 ORTOFOTO GENERATION

### 2.10.1 Processo de Ortorretificação

```python
def create_orthophoto(images_dir, reconstruction_json, dem, output, config):
    """
    Gerar ortofoto a partir de imagens undistorted e DEM
    """
    # 1. Carregar reconstrução
    reconstruction = load_reconstruction(reconstruction_json)
    
    # 2. Calcular bounds da ortofoto
    bounds = calculate_bounds(reconstruction, dem)
    
    # 3. Criar canvas vazio
    width = int((bounds.max_x - bounds.min_x) / config.resolution)
    height = int((bounds.max_y - bounds.min_y) / config.resolution)
    orthophoto = np.zeros((height, width, 4), dtype=np.uint8)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    # 4. Para cada imagem
    for shot_id, shot in reconstruction.shots.items():
        image = load_image(os.path.join(images_dir, shot_id))
        camera = reconstruction.cameras[shot.camera]
        
        # 5. Para cada pixel do DEM
        for dem_y in range(dem.height):
            for dem_x in range(dem.width):
                # Coordenadas mundo do pixel DEM
                world_x, world_y = dem.pixel_to_world(dem_x, dem_y)
                world_z = dem.get_elevation(dem_x, dem_y)
                
                # Projetar ponto 3D para imagem
                img_x, img_y = project_to_image(
                    [world_x, world_y, world_z],
                    shot.rotation, shot.translation,
                    camera
                )
                
                # Verificar se está dentro da imagem
                if 0 <= img_x < image.width and 0 <= img_y < image.height:
                    # Verificar visibilidade (ray casting)
                    if is_visible(shot, [world_x, world_y, world_z], dem):
                        # Coordenadas na ortofoto
                        ortho_x = int((world_x - bounds.min_x) / config.resolution)
                        ortho_y = int((bounds.max_y - world_y) / config.resolution)
                        
                        # Amostrar cor (bilinear)
                        color = bilinear_sample(image, img_x, img_y)
                        
                        # Blend com peso (distância ao centro)
                        weight = calculate_weight(img_x, img_y, image.width, image.height)
                        
                        # Acumular
                        orthophoto[ortho_y, ortho_x] += color * weight
                        weight_map[ortho_y, ortho_x] += weight
    
    # 6. Normalizar pesos
    orthophoto = orthophoto / weight_map[:, :, np.newaxis]
    
    # 7. Salvar como GeoTIFF
    save_geotiff(output, orthophoto, bounds, config.crs)
    
    return output
```

### 2.10.2 Equações de Colinearidade (Ortorretificação)

As equações de colinearidade relacionam coordenadas 3D do terreno com coordenadas 2D na imagem:

$$x = x_0 - f \cdot \frac{a_1(X-X_s) + b_1(Y-Y_s) + c_1(Z-Z_s)}{a_3(X-X_s) + b_3(Y-Y_s) + c_3(Z-Z_s)}$$

$$y = y_0 - f \cdot \frac{a_2(X-X_s) + b_2(Y-Y_s) + c_2(Z-Z_s)}{a_3(X-X_s) + b_3(Y-Y_s) + c_3(Z-Z_s)}$$

Onde:
- $(x, y)$ = coordenadas imagem
- $(x_0, y_0)$ = ponto principal
- $f$ = distância focal
- $(X, Y, Z)$ = coordenadas terreno
- $(X_s, Y_s, Z_s)$ = centro de projeção da câmera
- $a_i, b_i, c_i$ = elementos da matriz de rotação

```python
# Implementação das equações de colinearidade
def collinearity_equations(world_point, camera_position, rotation_matrix, focal, principal_point):
    """
    Implementação das equações de colinearidade
    
    Args:
        world_point: [X, Y, Z] - coordenadas terreno
        camera_position: [Xs, Ys, Zs] - posição da câmera
        rotation_matrix: matriz 3x3 de rotação
        focal: distância focal em pixels
        principal_point: [x0, y0] - ponto principal
    
    Returns:
        [x, y] - coordenadas na imagem
    """
    # Diferenças de coordenadas
    dX = world_point[0] - camera_position[0]
    dY = world_point[1] - camera_position[1]
    dZ = world_point[2] - camera_position[2]
    
    # Elementos da matriz de rotação
    a1, b1, c1 = rotation_matrix[0]
    a2, b2, c2 = rotation_matrix[1]
    a3, b3, c3 = rotation_matrix[2]
    
    # Denominador comum
    den = a3*dX + b3*dY + c3*dZ
    
    # Equações de colinearidade
    x = principal_point[0] - focal * (a1*dX + b1*dY + c1*dZ) / den
    y = principal_point[1] - focal * (a2*dX + b2*dY + c2*dZ) / den
    
    return np.array([x, y])
```

---

## 2.11 CONCLUSÕES DO CAPÍTULO 2

### 2.11.1 Componentes Essenciais para Fast Stitching Android

Para implementar um app Fast Stitching, os componentes mínimos são:

```
ESSENCIAL:
├── Feature Detection (SIFT/HAHOG/ORB)
├── Feature Matching (FLANN/BruteForce)  
├── SfM Incremental (Essential Matrix, PnP)
├── Triangulação de pontos
├── Bundle Adjustment simplificado
├── Undistortion de imagens
└── Ortorretificação básica

OPCIONAL (melhora qualidade):
├── Bundle Adjustment completo (Ceres)
├── Loop closure
├── Dense reconstruction (OpenMVS)
└── Color balancing avançado
```

### 2.11.2 Simplificações para Mobile

```python
# Configurações recomendadas para Android
MOBILE_CONFIG = {
    'feature_type': 'orb',           # Mais rápido, sem patentes
    'max_features': 2000,            # Reduzido
    'feature_process_size': 1024,    # Imagens menores
    'matcher_type': 'bruteforce',    # Simples para ORB
    'use_gps_prior': True,           # Reduz matching pairs
    'skip_bundle_adjustment': False, # Essencial para qualidade
    'skip_dense_reconstruction': True,  # Fast mode
    'orthophoto_resolution': 20,     # 20 cm/pixel
}
```

### 2.11.3 Estimativa de Performance Mobile

| Operação | Desktop (i7) | Android (Snapdragon 8) | Estratégia |
|----------|--------------|------------------------|------------|
| Feature Extract (ORB) | 0.5s/img | 2s/img | GPU/NEON |
| Feature Match | 0.1s/pair | 0.5s/pair | Limit pairs |
| SfM Reconstruct | 30s | 120s | Simplify |
| Bundle Adjustment | 10s | 60s | Local BA only |
| Orthophoto | 60s | 300s | Tile-based |
| **Total (50 imgs)** | ~5 min | ~15-20 min | Aceitável |

