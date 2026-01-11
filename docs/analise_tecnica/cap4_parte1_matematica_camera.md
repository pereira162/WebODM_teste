# CAPÍTULO 4: FUNDAMENTOS MATEMÁTICOS DE AGRIMENSURA

## Parte 1: Geometria de Câmera e Projeções

---

## 4.1 MODELO PINHOLE DE CÂMERA

O modelo matemático fundamental para fotogrametria é o modelo pinhole (câmera de orifício).

### 4.1.1 Projeção Perspectiva

```
MODELO PINHOLE:
═══════════════

                    Ponto 3D (X, Y, Z)
                           •
                          /|
                         / |
                        /  |
                       /   |
            Centro    /    | Z
            Ótico    /     |
               O ───/──────┼────────▶ Eixo Z (profundidade)
                \  /       |
                 \/        |
                 /\        |
                /  ────────┼──── Plano da Imagem
               /           |     (z = f)
              /            |
             •             |
        Ponto 2D (u, v)    |
        na imagem          |
                           ▼
```

A projeção de um ponto 3D $(X, Y, Z)$ para coordenadas de imagem $(u, v)$:

$$u = f \cdot \frac{X}{Z} + c_x$$

$$v = f \cdot \frac{Y}{Z} + c_y$$

Onde:
- $f$ = distância focal (em pixels)
- $(c_x, c_y)$ = ponto principal (centro ótico projetado)

### 4.1.2 Matriz de Parâmetros Intrínsecos

A matriz de câmera $K$ encapsula os parâmetros intrínsecos:

$$K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Onde:
- $f_x, f_y$ = distâncias focais em pixels (X e Y)
- $s$ = skew (geralmente 0 para câmeras digitais)
- $c_x, c_y$ = coordenadas do ponto principal

```python
# Implementação da matriz intrínseca
import numpy as np

def create_camera_matrix(focal_length_mm, sensor_width_mm, image_width, image_height):
    """
    Criar matriz de câmera K a partir de parâmetros físicos
    """
    # Converter focal length para pixels
    focal_px = focal_length_mm * image_width / sensor_width_mm
    
    # Ponto principal (assumindo centro da imagem)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    K = np.array([
        [focal_px, 0, cx],
        [0, focal_px, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K
```

### 4.1.3 Matriz de Projeção Completa

A projeção completa inclui a pose da câmera (rotação $R$ e translação $t$):

$$P = K \cdot [R | t]$$

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z'} \cdot P \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

```python
def project_point_3d_to_2d(point_3d, K, R, t):
    """
    Projetar ponto 3D para coordenadas de imagem 2D
    
    Args:
        point_3d: numpy array [X, Y, Z]
        K: matriz intrínseca 3x3
        R: matriz de rotação 3x3
        t: vetor de translação 3x1
    
    Returns:
        numpy array [u, v] - coordenadas de imagem
    """
    # Transformar para coordenadas de câmera
    point_cam = R @ point_3d + t.flatten()
    
    # Verificar se está na frente da câmera
    if point_cam[2] <= 0:
        return None
    
    # Projeção perspectiva
    point_norm = point_cam[:2] / point_cam[2]
    
    # Aplicar matriz intrínseca
    point_px = K[:2, :2] @ point_norm + K[:2, 2]
    
    return point_px
```

---

## 4.2 MODELOS DE DISTORÇÃO DE LENTE

Lentes reais introduzem distorções que devem ser modeladas e corrigidas.

### 4.2.1 Modelo Brown-Conrady (Radial + Tangencial)

As distorções mais comuns são:

**Distorção Radial:**
$$x_{distorted} = x \cdot (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_{distorted} = y \cdot (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

**Distorção Tangencial:**
$$x_{distorted} = x + 2p_1 xy + p_2(r^2 + 2x^2)$$
$$y_{distorted} = y + p_1(r^2 + 2y^2) + 2p_2 xy$$

Onde:
- $r^2 = x^2 + y^2$ (distância ao centro)
- $k_1, k_2, k_3$ = coeficientes de distorção radial
- $p_1, p_2$ = coeficientes de distorção tangencial

```python
def apply_distortion(x_normalized, y_normalized, dist_coeffs):
    """
    Aplicar modelo de distorção Brown-Conrady
    
    Args:
        x_normalized, y_normalized: coordenadas normalizadas
        dist_coeffs: [k1, k2, p1, p2, k3]
    
    Returns:
        x_distorted, y_distorted
    """
    k1, k2, p1, p2, k3 = dist_coeffs[:5]
    
    r2 = x_normalized**2 + y_normalized**2
    r4 = r2 * r2
    r6 = r4 * r2
    
    # Distorção radial
    radial = 1 + k1*r2 + k2*r4 + k3*r6
    
    # Distorção tangencial
    x_dist = x_normalized * radial + 2*p1*x_normalized*y_normalized + p2*(r2 + 2*x_normalized**2)
    y_dist = y_normalized * radial + p1*(r2 + 2*y_normalized**2) + 2*p2*x_normalized*y_normalized
    
    return x_dist, y_dist

def undistort_point(x_distorted, y_distorted, dist_coeffs, iterations=10):
    """
    Remover distorção (processo iterativo)
    """
    x = x_distorted
    y = y_distorted
    
    for _ in range(iterations):
        x_dist, y_dist = apply_distortion(x, y, dist_coeffs)
        x = x_distorted - (x_dist - x)
        y = y_distorted - (y_dist - y)
    
    return x, y
```

### 4.2.2 Modelo Fisheye (Equidistante)

Para lentes grande angular e fisheye:

$$r_{distorted} = f \cdot \theta$$

Onde $\theta$ é o ângulo de incidência do raio.

```python
def fisheye_projection(point_3d, K, dist_coeffs):
    """
    Projeção para lente fisheye (modelo equidistante)
    """
    x, y, z = point_3d
    
    # Ângulo de incidência
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r, z)
    
    # Distorção fisheye
    theta_d = theta * (1 + dist_coeffs[0]*theta**2 + 
                       dist_coeffs[1]*theta**4 + 
                       dist_coeffs[2]*theta**6 + 
                       dist_coeffs[3]*theta**8)
    
    # Coordenadas de imagem
    if r > 1e-8:
        scale = theta_d / r
        u = K[0,0] * x * scale + K[0,2]
        v = K[1,1] * y * scale + K[1,2]
    else:
        u = K[0,2]
        v = K[1,2]
    
    return np.array([u, v])
```

---

## 4.3 GEOMETRIA EPIPOLAR

A geometria epipolar descreve a relação geométrica entre duas vistas de uma cena.

### 4.3.1 Matriz Fundamental

A matriz fundamental $F$ relaciona pontos correspondentes em duas imagens:

$$x'^T \cdot F \cdot x = 0$$

Onde $x$ e $x'$ são pontos correspondentes em coordenadas homogêneas.

```python
def compute_fundamental_matrix(points1, points2):
    """
    Calcular matriz fundamental usando algoritmo 8-point normalizado
    """
    import cv2
    
    # Normalizar pontos
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)
    
    # Construir matriz A para o sistema Af = 0
    n = len(points1)
    A = np.zeros((n, 9))
    
    for i in range(n):
        x1, y1 = points1_norm[i]
        x2, y2 = points2_norm[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Resolver via SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Forçar rank 2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    
    # Desnormalizar
    F = T2.T @ F @ T1
    
    return F / F[2, 2]  # Normalizar
```

### 4.3.2 Matriz Essencial

A matriz essencial $E$ é a versão calibrada da matriz fundamental:

$$E = K'^T \cdot F \cdot K$$

E pode ser decomposta em rotação $R$ e translação $t$:

$$E = [t]_\times \cdot R$$

Onde $[t]_\times$ é a matriz skew-symmetric de $t$.

```python
def decompose_essential_matrix(E, K1, K2, points1, points2):
    """
    Decompor matriz essencial em R e t
    """
    # SVD de E
    U, S, Vt = np.linalg.svd(E)
    
    # Garantir rotação válida
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    
    # Matriz W para decomposição
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # 4 soluções possíveis
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    
    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]
    
    # Escolher solução onde pontos estão na frente de ambas câmeras
    best_solution = None
    max_in_front = 0
    
    for R, t in solutions:
        in_front = count_points_in_front(points1, points2, K1, K2, R, t)
        if in_front > max_in_front:
            max_in_front = in_front
            best_solution = (R, t)
    
    return best_solution
```

---

## 4.4 TRIANGULAÇÃO

A triangulação determina a posição 3D de um ponto a partir de suas projeções em múltiplas imagens.

### 4.4.1 Triangulação Linear (DLT)

Dado um ponto $x = (u, v, 1)$ na imagem e a matriz de projeção $P$:

$$x \times (P \cdot X) = 0$$

Expandindo para duas vistas:

$$\begin{bmatrix} 
u(P_3^T) - P_1^T \\
v(P_3^T) - P_2^T \\
u'(P'_3^T) - P'_1^T \\
v'(P'_3^T) - P'_2^T
\end{bmatrix} \cdot X = 0$$

```python
def triangulate_point_dlt(P1, P2, x1, x2):
    """
    Triangulação DLT (Direct Linear Transform)
    
    Args:
        P1, P2: Matrizes de projeção 3x4
        x1, x2: Pontos 2D correspondentes [u, v]
    
    Returns:
        X: Ponto 3D [X, Y, Z]
    """
    # Construir matriz A
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1]
    ])
    
    # Resolver via SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    # Converter para coordenadas euclidianas
    X = X[:3] / X[3]
    
    return X

def triangulate_points(P1, P2, points1, points2):
    """
    Triangular múltiplos pontos
    """
    points_3d = []
    
    for x1, x2 in zip(points1, points2):
        X = triangulate_point_dlt(P1, P2, x1, x2)
        points_3d.append(X)
    
    return np.array(points_3d)
```

### 4.4.2 Triangulação com Múltiplas Vistas

Para $N$ vistas do mesmo ponto:

$$\begin{bmatrix} 
u_1(P_1^{(3)T}) - P_1^{(1)T} \\
v_1(P_1^{(3)T}) - P_1^{(2)T} \\
\vdots \\
u_N(P_N^{(3)T}) - P_N^{(1)T} \\
v_N(P_N^{(3)T}) - P_N^{(2)T}
\end{bmatrix} \cdot X = 0$$

```python
def triangulate_multi_view(projections, observations):
    """
    Triangulação com múltiplas vistas
    
    Args:
        projections: Lista de matrizes P 3x4
        observations: Lista de pontos 2D [u, v]
    
    Returns:
        X: Ponto 3D
    """
    n_views = len(projections)
    A = np.zeros((2 * n_views, 4))
    
    for i, (P, obs) in enumerate(zip(projections, observations)):
        u, v = obs
        A[2*i] = u * P[2] - P[0]
        A[2*i + 1] = v * P[2] - P[1]
    
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]
    
    return X
```

---

## 4.5 PERSPECTIVE-N-POINT (PnP)

O problema PnP estima a pose da câmera dados pontos 3D conhecidos e suas projeções 2D.

### 4.5.1 Formulação

Dado:
- $n$ pontos 3D $\{X_1, X_2, ..., X_n\}$ em coordenadas mundo
- Suas projeções 2D $\{x_1, x_2, ..., x_n\}$ na imagem
- Matriz intrínseca $K$

Encontrar: Rotação $R$ e translação $t$ da câmera.

### 4.5.2 Solução EPnP (Efficient PnP)

```python
def solve_pnp_epnp(points_3d, points_2d, K):
    """
    Resolver PnP usando algoritmo EPnP
    """
    import cv2
    
    # Converter para formatos OpenCV
    object_points = np.array(points_3d, dtype=np.float64)
    image_points = np.array(points_2d, dtype=np.float64)
    
    # Resolver
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        None,  # Sem distorção (ou passar coeffs)
        flags=cv2.SOLVEPNP_EPNP
    )
    
    if success:
        # Converter rotation vector para matriz
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten()
    else:
        return None, None
```

### 4.5.3 Refinamento via Levenberg-Marquardt

```python
def refine_pnp_lm(R_init, t_init, points_3d, points_2d, K):
    """
    Refinar solução PnP via otimização não-linear
    """
    from scipy.optimize import least_squares
    
    def residuals(params):
        rvec = params[:3]
        tvec = params[3:6]
        R, _ = cv2.Rodrigues(rvec)
        
        errors = []
        for X, x_obs in zip(points_3d, points_2d):
            x_proj = project_point_3d_to_2d(X, K, R, tvec)
            errors.extend(x_proj - x_obs)
        
        return np.array(errors)
    
    # Parâmetros iniciais
    rvec_init, _ = cv2.Rodrigues(R_init)
    params_init = np.concatenate([rvec_init.flatten(), t_init])
    
    # Otimizar
    result = least_squares(
        residuals,
        params_init,
        method='lm',
        ftol=1e-10
    )
    
    # Extrair resultado
    rvec = result.x[:3]
    tvec = result.x[3:6]
    R, _ = cv2.Rodrigues(rvec)
    
    return R, tvec
```

---

## 4.6 GROUND SAMPLE DISTANCE (GSD)

O GSD define a resolução espacial da ortofoto.

### 4.6.1 Cálculo do GSD

$$GSD = \frac{H \cdot S_w}{f \cdot I_w}$$

Onde:
- $H$ = altura de voo (metros)
- $S_w$ = largura do sensor (mm)
- $f$ = distância focal (mm)
- $I_w$ = largura da imagem (pixels)

```python
def calculate_gsd(flight_height_m, sensor_width_mm, focal_length_mm, image_width_px):
    """
    Calcular Ground Sample Distance
    
    Args:
        flight_height_m: Altura de voo em metros
        sensor_width_mm: Largura do sensor em mm
        focal_length_mm: Distância focal em mm
        image_width_px: Largura da imagem em pixels
    
    Returns:
        GSD em centímetros/pixel
    """
    # GSD em metros
    gsd_m = (flight_height_m * sensor_width_mm) / (focal_length_mm * image_width_px)
    
    # Converter para cm
    gsd_cm = gsd_m * 100
    
    return gsd_cm

# Exemplo: DJI Phantom 4 Pro a 100m
gsd = calculate_gsd(
    flight_height_m=100,
    sensor_width_mm=13.2,   # 1" sensor
    focal_length_mm=8.8,
    image_width_px=5472
)
# GSD ≈ 2.7 cm/pixel
```

### 4.6.2 Estimativa de GSD a partir de EXIF

```python
def estimate_gsd_from_exif(exif_data):
    """
    Estimar GSD a partir de metadados EXIF
    """
    # Extrair dados relevantes
    focal_length = exif_data.get('FocalLength', 4.0)  # mm
    image_width = exif_data.get('ImageWidth', 4000)
    image_height = exif_data.get('ImageHeight', 3000)
    
    # Altitude (pode vir de GPS ou metadados específicos do drone)
    altitude = exif_data.get('GPSAltitude', 100)
    
    # Tamanho do sensor (pode ser estimado ou de banco de dados)
    # Se não disponível, estimar baseado no modelo da câmera
    sensor_width = estimate_sensor_size(exif_data.get('Model', 'unknown'))
    
    # Calcular GSD
    gsd = (altitude * sensor_width) / (focal_length * image_width)
    
    return gsd * 100  # em cm/pixel
```

