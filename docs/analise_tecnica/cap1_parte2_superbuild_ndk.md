# CAPÍTULO 1: STACK TECNOLÓGICO E PORTABILIDADE

## Parte 2: Sistema SuperBuild e Compilação NDK

---

## 1.8 SISTEMA SUPERBUILD DO ODM

O ODM utiliza um sistema **SuperBuild** baseado em CMake para gerenciar a compilação de todas as dependências C++ de forma automatizada.

### 1.8.1 Estrutura do SuperBuild

```
ODM/SuperBuild/
├── CMakeLists.txt              # Arquivo principal
├── cmake/
│   ├── External-OpenCV.cmake   # OpenCV 4.5.0
│   ├── External-Ceres.cmake    # Ceres Solver 2.0.0
│   ├── External-OpenSfM.cmake  # OpenSfM
│   ├── External-OpenMVS.cmake  # OpenMVS
│   ├── External-PDAL.cmake     # PDAL
│   ├── External-GFlags.cmake   # Google Flags 2.2.2
│   ├── External-Eigen.cmake    # Eigen 3.4
│   ├── External-LASzip.cmake   # Compressão LAS
│   └── External-Entwine.cmake  # EPT generation
└── build/                       # Diretório de compilação
```

### 1.8.2 Configuração CMake Principal

```cmake
# SuperBuild/CMakeLists.txt (estrutura simplificada)
cmake_minimum_required(VERSION 3.10)
project(ODM-SuperBuild)

set(SB_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads")
set(SB_SOURCE_DIR "${CMAKE_BINARY_DIR}/src")
set(SB_BINARY_DIR "${CMAKE_BINARY_DIR}/build")
set(SB_INSTALL_DIR "${CMAKE_BINARY_DIR}/install")

# Ordem de compilação por dependência
include(cmake/External-GFlags.cmake)    # Sem deps
include(cmake/External-Eigen.cmake)     # Sem deps
include(cmake/External-Ceres.cmake)     # Deps: GFlags, Eigen
include(cmake/External-OpenCV.cmake)    # Deps: Eigen
include(cmake/External-OpenSfM.cmake)   # Deps: Ceres, OpenCV
include(cmake/External-OpenMVS.cmake)   # Deps: OpenCV, Ceres, Eigen
```

---

## 1.9 CONFIGURAÇÃO DE CADA DEPENDÊNCIA

### 1.9.1 OpenCV 4.5.0

```cmake
# External-OpenCV.cmake
ExternalProject_Add(opencv
    URL https://github.com/opencv/opencv/archive/4.5.0.zip
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${SB_INSTALL_DIR}
        -DBUILD_opencv_core=ON
        -DBUILD_opencv_imgproc=ON
        -DBUILD_opencv_features2d=ON
        -DBUILD_opencv_calib3d=ON
        -DBUILD_opencv_xfeatures2d=ON  # SIFT, SURF
        -DBUILD_opencv_python3=ON
        # Desabilitados para reduzir tamanho:
        -DWITH_CUDA=OFF
        -DWITH_GTK=OFF
        -DWITH_VTK=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_DOCS=OFF
)
```

**Módulos Essenciais para Stitching:**
- `opencv_core` - Estruturas de dados básicas
- `opencv_imgproc` - Processamento de imagem
- `opencv_features2d` - Detecção de features (ORB, AKAZE)
- `opencv_calib3d` - Calibração de câmera, homografia
- `opencv_xfeatures2d` - SIFT, SURF (patented)

### 1.9.2 Ceres Solver 2.0.0

```cmake
# External-Ceres.cmake
ExternalProject_Add(ceres
    URL http://ceres-solver.org/ceres-solver-2.0.0.tar.gz
    DEPENDS gflags eigen
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${SB_INSTALL_DIR}
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC
        -DBUILD_TESTING=OFF
        -DBUILD_EXAMPLES=OFF
        -DMINIGLOG=ON              # Logger embarcado
        -DGFLAGS_INCLUDE_DIR=${SB_INSTALL_DIR}/include
)
```

**Funcionalidades Ceres:**
- Non-linear least squares solver
- Bundle adjustment
- Automatic differentiation
- Sparse matrix operations (Eigen backend)

### 1.9.3 OpenSfM

```cmake
# External-OpenSfM.cmake
ExternalProject_Add(opensfm
    GIT_REPOSITORY https://github.com/OpenDroneMap/OpenSfM/
    GIT_TAG main
    DEPENDS ceres opencv gflags
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${SB_INSTALL_DIR}
        -DCERES_ROOT_DIR=${SB_INSTALL_DIR}
        -DOpenCV_DIR=${SB_INSTALL_DIR}
        -DOPENSFM_BUILD_TESTS=OFF
    UPDATE_COMMAND git submodule update --init --recursive
)
```

### 1.9.4 OpenMVS

```cmake
# External-OpenMVS.cmake
ExternalProject_Add(openmvs
    GIT_REPOSITORY https://github.com/OpenDroneMap/openMVS
    GIT_TAG master
    DEPENDS ceres opencv vcg eigen34
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DOpenCV_DIR=${SB_INSTALL_DIR}
        -DVCG_ROOT=${SB_SOURCE_DIR}/vcg
        -DEIGEN3_INCLUDE_DIR=${SB_INSTALL_DIR}/include/eigen3
        -DOpenMVS_ENABLE_TESTS=OFF
        -DOpenMVS_MAX_CUDA_COMPATIBILITY=ON
        # ARM64 específico:
        -DOpenMVS_USE_SSE=$<IF:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>,OFF,ON>
)
```

### 1.9.5 PDAL

```cmake
# External-PDAL.cmake
ExternalProject_Add(pdal
    URL https://github.com/OpenDroneMap/PDAL/archive/refs/heads/333.zip
    DEPENDS hexer laszip
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DWITH_APPS=ON
        -DWITH_GEOTIFF=ON
        -DWITH_LASZIP=ON
        -DBUILD_PLUGIN_HEXBIN=ON
        # Desabilitados:
        -DBUILD_PLUGIN_PGPOINTCLOUD=OFF
        -DBUILD_PLUGIN_PYTHON=OFF
        -DWITH_TESTS=OFF
)
```

---

## 1.10 CROSS-COMPILAÇÃO PARA ANDROID NDK

### 1.10.1 Toolchain Android

```cmake
# android-toolchain.cmake
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 24)  # Android 7.0 mínimo
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK /path/to/android-ndk-r25c)
set(CMAKE_ANDROID_STL_TYPE c++_shared)

# Flags específicos ARM64
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++17")

# Substituir SSE por NEON
add_definitions(-DEIGEN_DONT_VECTORIZE)
add_definitions(-DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
```

### 1.10.2 Script de Build Android

```bash
#!/bin/bash
# build_android.sh

export ANDROID_NDK=/opt/android-ndk-r25c
export API_LEVEL=24
export ABI=arm64-v8a

mkdir -p build_android && cd build_android

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ABI \
    -DANDROID_NATIVE_API_LEVEL=$API_LEVEL \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DWITH_CUDA=OFF \
    -DWITH_OPENCL=OFF

make -j$(nproc)
```

### 1.10.3 Problemas Conhecidos NDK

| Problema | Solução |
|----------|---------|
| SSE intrinsics não disponíveis | Usar NEON ou desabilitar vetorização |
| `aligned_alloc` não existe | Usar `posix_memalign` |
| Threads POSIX | Linkar com `-lpthread` |
| OpenMP não funciona | Desabilitar ou usar alternativas |
| Filesystem C++17 | Usar Boost.Filesystem como fallback |

---

## 1.11 BIBLIOTECAS ANDROID EQUIVALENTES

### 1.11.1 Substituições Recomendadas

```
DESKTOP → ANDROID

GDAL/Rasterio     → GeoTIFF-Android ou libgeotiff compilado
PostgreSQL        → SQLite + Room
Redis             → Não necessário (app local)
Celery            → WorkManager ou Coroutines
Django            → Não aplicável
NumPy             → Chaquopy NumPy ou implementação JNI
Pillow            → Android Bitmap + BitmapFactory
PDAL              → Implementação simplificada custom
Proj4             → proj4j (Java) ou Proj compilado
```

### 1.11.2 OpenCV Android SDK

```gradle
// build.gradle (app)
dependencies {
    implementation 'org.opencv:opencv:4.5.0'
}
```

Ou compile manualmente:
```bash
# Clone OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv/platforms/android

# Build
python build_sdk.py \
    --ndk_path $ANDROID_NDK \
    --sdk_path $ANDROID_SDK \
    --extra_modules_path ../opencv_contrib/modules \
    android_build
```

---

## 1.12 ESTIMATIVA DE TAMANHO APK

### 1.12.1 Bibliotecas Nativas (.so)

| Biblioteca | arm64-v8a | armeabi-v7a |
|------------|-----------|-------------|
| libopencv_java4.so | ~35 MB | ~25 MB |
| libceres.so | ~8 MB | ~6 MB |
| libopensfm.so | ~12 MB | ~9 MB |
| libeigen.so | Header-only | - |
| libgflags.so | ~200 KB | ~150 KB |
| **Total estimado** | **~55 MB** | **~40 MB** |

### 1.12.2 Otimizações de Tamanho

```cmake
# CMake flags para reduzir tamanho
set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG")  # Optimize for size
set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--gc-sections")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--gc-sections")

# Strip symbols
add_custom_command(TARGET mylib POST_BUILD
    COMMAND ${CMAKE_STRIP} $<TARGET_FILE:mylib>
)
```

```gradle
// build.gradle
android {
    buildTypes {
        release {
            minifyEnabled true
            shrinkResources true
            ndk {
                debugSymbolLevel 'NONE'
            }
        }
    }
    
    packagingOptions {
        exclude 'lib/*/libopencv_*.so'  // Excluir módulos não usados
    }
}
```

---

## 1.13 CONCLUSÕES DO CAPÍTULO 1

### 1.13.1 Viabilidade Técnica

| Aspecto | Avaliação | Justificativa |
|---------|-----------|---------------|
| **OpenCV** | ✅ Viável | SDK Android oficial disponível |
| **Ceres** | ✅ Viável | Cross-compile testado |
| **OpenSfM** | ⚠️ Desafiador | Híbrido Python/C++, requer adaptação |
| **OpenMVS** | ❌ Não recomendado | Dependências CUDA, complexo demais |
| **GDAL completo** | ⚠️ Desafiador | Muitas dependências |
| **Fast Stitching** | ✅ Viável | Pode usar pipeline simplificado |

### 1.13.2 Recomendação para App Android

Para um app "Fast Stitching" offline em tablet, recomenda-se:

1. **Usar OpenCV Android SDK** para feature detection/matching
2. **Compilar Ceres via NDK** para bundle adjustment
3. **Portar OpenSfM core** como biblioteca C++ pura
4. **Pular OpenMVS** - usar sparse reconstruction
5. **Simplificar output** - apenas ortofoto, sem DEM/DTM completo
6. **SQLite** para persistência local

### 1.13.3 Arquitetura Proposta Android

```
┌─────────────────────────────────────────────────────┐
│              ANDROID FAST STITCHING APP             │
├─────────────────────────────────────────────────────┤
│  UI Layer (Kotlin/Jetpack Compose)                  │
│  ├── Camera Preview/Capture                         │
│  ├── Project Management                             │
│  └── Map Visualization (OSMDroid/MapLibre)          │
├─────────────────────────────────────────────────────┤
│  Domain Layer (Kotlin)                              │
│  ├── Processing Orchestration                       │
│  ├── Progress Tracking                              │
│  └── Background WorkManager                         │
├─────────────────────────────────────────────────────┤
│  Native Layer (C++ via JNI)                         │
│  ├── libopencv_java4.so (Feature Detection)         │
│  ├── libceres.so (Bundle Adjustment)                │
│  ├── libstitcher.so (Custom SfM Pipeline)           │
│  └── libgeotiff.so (GeoTIFF Output)                 │
├─────────────────────────────────────────────────────┤
│  Data Layer                                         │
│  ├── SQLite/Room (Projects, Tasks)                  │
│  ├── File System (Images, Outputs)                  │
│  └── SharedPreferences (Settings)                   │
└─────────────────────────────────────────────────────┘
```

