# CAPÍTULO 5: BLUEPRINT DE IMPLEMENTAÇÃO ANDROID

## Parte 1: Arquitetura e JNI

---

## 5.1 VISÃO GERAL DA ARQUITETURA

### 5.1.1 Arquitetura em Camadas

```
┌─────────────────────────────────────────────────────────────────────┐
│                        UI LAYER (Kotlin/Compose)                    │
│  ┌─────────────┐  ┌────────────────┐  ┌──────────────────────────┐ │
│  │ CameraView  │  │ MapViewer      │  │ ProgressIndicators      │ │
│  │ (CameraX)   │  │ (OSMdroid)     │  │ (Material Design)       │ │
│  └─────────────┘  └────────────────┘  └──────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                      VIEWMODEL LAYER (Kotlin)                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ ProjectViewModel│  │ ProcessViewModel │  │ ResultViewModel   │  │
│  │ (StateFlow)     │  │ (WorkManager)    │  │ (LiveData)        │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                      DOMAIN LAYER (Kotlin)                          │
│  ┌──────────────────┐  ┌───────────────┐  ┌────────────────────┐   │
│  │ ProcessUseCase   │  │ ExportUseCase │  │ ValidationUseCase  │   │
│  │                  │  │ (GeoTIFF/KML) │  │ (Image/GPS Check)  │   │
│  └──────────────────┘  └───────────────┘  └────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                      DATA LAYER (Kotlin)                            │
│  ┌──────────────────┐  ┌───────────────┐  ┌────────────────────┐   │
│  │ ProjectRepository│  │ ImageRepository│  │ ResultRepository   │   │
│  │ (Room DB)        │  │ (FileSystem)   │  │ (SQLite + Files)   │   │
│  └──────────────────┘  └───────────────┘  └────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                      NATIVE LAYER (C++/JNI)                         │
│  ┌──────────────────┐  ┌───────────────┐  ┌────────────────────┐   │
│  │ SfMEngine        │  │ FeatureExtract│  │ OrthophotoGen      │   │
│  │ (OpenSfM port)   │  │ (AKAZE/ORB)   │  │ (Fast Stitching)   │   │
│  └──────────────────┘  └───────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NATIVE LIBRARIES (arm64-v8a)                     │
│  ┌────────────┐ ┌──────────────┐ ┌─────────┐ ┌───────────────────┐ │
│  │ OpenCV     │ │ Ceres Solver │ │ Eigen   │ │ libtiff/libgeotiff│ │
│  │ 4.5.0      │ │ 2.0.0        │ │ 3.4     │ │ (GeoTIFF support) │ │
│  └────────────┘ └──────────────┘ └─────────┘ └───────────────────┘ │
│  ┌────────────┐ ┌──────────────┐ ┌─────────┐ ┌───────────────────┐ │
│  │ libglog    │ │ libgflags    │ │ libproj │ │ Custom SfM Core   │ │
│  │ 0.6.0      │ │ 2.2.2        │ │ 9.x     │ │ (Simplified ODM)  │ │
│  └────────────┘ └──────────────┘ └─────────┘ └───────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.1.2 Estrutura do Projeto Android

```
FastStitching/
├── app/
│   ├── build.gradle.kts
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/faststitching/
│   │   │   │   ├── FastStitchingApp.kt
│   │   │   │   ├── di/                          # Dependency Injection
│   │   │   │   │   ├── AppModule.kt
│   │   │   │   │   ├── DatabaseModule.kt
│   │   │   │   │   └── NativeModule.kt
│   │   │   │   ├── domain/
│   │   │   │   │   ├── model/
│   │   │   │   │   │   ├── Project.kt
│   │   │   │   │   │   ├── ProcessingTask.kt
│   │   │   │   │   │   ├── GeoImage.kt
│   │   │   │   │   │   └── Orthophoto.kt
│   │   │   │   │   └── usecase/
│   │   │   │   │       ├── ProcessImagesUseCase.kt
│   │   │   │   │       ├── ExportOrthophotoUseCase.kt
│   │   │   │   │       └── ValidateImagesUseCase.kt
│   │   │   │   ├── data/
│   │   │   │   │   ├── local/
│   │   │   │   │   │   ├── AppDatabase.kt
│   │   │   │   │   │   ├── ProjectDao.kt
│   │   │   │   │   │   └── TaskDao.kt
│   │   │   │   │   └── repository/
│   │   │   │   │       ├── ProjectRepositoryImpl.kt
│   │   │   │   │       └── ImageRepositoryImpl.kt
│   │   │   │   ├── presentation/
│   │   │   │   │   ├── ui/
│   │   │   │   │   │   ├── home/
│   │   │   │   │   │   ├── capture/
│   │   │   │   │   │   ├── processing/
│   │   │   │   │   │   └── results/
│   │   │   │   │   └── viewmodel/
│   │   │   │   │       ├── HomeViewModel.kt
│   │   │   │   │       └── ProcessingViewModel.kt
│   │   │   │   └── native/
│   │   │   │       ├── NativeBridge.kt           # JNI Interface
│   │   │   │       ├── SfMEngine.kt
│   │   │   │       └── FeatureExtractor.kt
│   │   │   ├── jniLibs/
│   │   │   │   └── arm64-v8a/
│   │   │   │       ├── libopencv_java4.so
│   │   │   │       ├── libceres.so
│   │   │   │       ├── libsfm_core.so
│   │   │   │       ├── libgeotiff.so
│   │   │   │       └── libfaststitching.so
│   │   │   └── cpp/
│   │   │       ├── CMakeLists.txt
│   │   │       ├── native_bridge.cpp             # JNI Implementation
│   │   │       ├── sfm/
│   │   │       │   ├── feature_extraction.cpp
│   │   │       │   ├── feature_matching.cpp
│   │   │       │   ├── reconstruction.cpp
│   │   │       │   └── bundle_adjustment.cpp
│   │   │       ├── ortho/
│   │   │       │   ├── fast_orthophoto.cpp
│   │   │       │   └── geotiff_writer.cpp
│   │   │       └── utils/
│   │   │           ├── memory_pool.cpp
│   │   │           └── image_utils.cpp
│   │   └── test/
│   └── proguard-rules.pro
├── native/
│   ├── CMakeLists.txt                            # Main CMake
│   ├── toolchain/
│   │   └── android.toolchain.cmake
│   └── thirdparty/
│       ├── opencv/
│       ├── ceres-solver/
│       ├── eigen/
│       ├── gflags/
│       ├── glog/
│       ├── libtiff/
│       └── libgeotiff/
├── build.gradle.kts
├── settings.gradle.kts
└── gradle.properties
```

---

## 5.2 INTERFACE JNI (Java Native Interface)

### 5.2.1 Definição da Interface Kotlin

```kotlin
// app/src/main/java/com/faststitching/native/NativeBridge.kt
package com.faststitching.native

import android.graphics.Bitmap
import java.nio.ByteBuffer

/**
 * Bridge para código nativo C++ via JNI
 * Todas as operações pesadas de processamento são feitas em C++
 */
object NativeBridge {
    
    init {
        // Carregar bibliotecas nativas na ordem correta de dependência
        System.loadLibrary("opencv_java4")
        System.loadLibrary("glog")
        System.loadLibrary("gflags") 
        System.loadLibrary("ceres")
        System.loadLibrary("geotiff")
        System.loadLibrary("sfm_core")
        System.loadLibrary("faststitching")
    }
    
    // ========== EXTRAÇÃO DE FEATURES ==========
    
    /**
     * Extrair features de uma imagem
     * @param imagePath Caminho absoluto da imagem
     * @param detectorType Tipo de detector: "AKAZE", "ORB", "SIFT"
     * @param maxFeatures Número máximo de features
     * @return Handle para o objeto nativo de features
     */
    external fun extractFeatures(
        imagePath: String,
        detectorType: String,
        maxFeatures: Int
    ): Long  // Retorna ponteiro nativo como Long
    
    /**
     * Obter número de features extraídas
     */
    external fun getFeatureCount(featuresHandle: Long): Int
    
    /**
     * Liberar memória das features
     */
    external fun releaseFeatures(featuresHandle: Long)
    
    // ========== MATCHING ==========
    
    /**
     * Fazer matching entre duas imagens
     * @return Array de matches [queryIdx, trainIdx, distance]
     */
    external fun matchFeatures(
        features1Handle: Long,
        features2Handle: Long,
        matcherType: String,  // "FLANN", "BF"
        crossCheck: Boolean,
        ratioThreshold: Float
    ): FloatArray
    
    // ========== RECONSTRUÇÃO ==========
    
    /**
     * Iniciar sessão de reconstrução
     * @return Handle da sessão
     */
    external fun createReconstructionSession(
        outputDir: String,
        numThreads: Int,
        useFastMode: Boolean
    ): Long
    
    /**
     * Adicionar imagem à reconstrução
     */
    external fun addImageToReconstruction(
        sessionHandle: Long,
        imagePath: String,
        focalLengthMm: Float,
        sensorWidthMm: Float,
        latitude: Double,
        longitude: Double,
        altitude: Double,
        yaw: Float,
        pitch: Float,
        roll: Float
    ): Boolean
    
    /**
     * Executar reconstrução incremental
     * @param progressCallback Callback para progresso (0-100)
     */
    external fun runReconstruction(
        sessionHandle: Long,
        progressCallback: ProgressCallback
    ): Int  // Status code
    
    /**
     * Obter estatísticas da reconstrução
     */
    external fun getReconstructionStats(sessionHandle: Long): ReconstructionStats
    
    /**
     * Liberar sessão de reconstrução
     */
    external fun releaseReconstructionSession(sessionHandle: Long)
    
    // ========== GERAÇÃO DE ORTHOPHOTO ==========
    
    /**
     * Gerar orthophoto a partir da reconstrução
     * @param gsd Ground Sampling Distance em metros/pixel
     * @return Caminho do arquivo GeoTIFF gerado
     */
    external fun generateOrthophoto(
        sessionHandle: Long,
        outputPath: String,
        gsd: Float,
        epsgCode: Int,
        progressCallback: ProgressCallback
    ): String
    
    // ========== UTILITÁRIOS ==========
    
    /**
     * Obter informações da biblioteca
     */
    external fun getLibraryInfo(): String
    
    /**
     * Definir diretório temporário
     */
    external fun setTempDirectory(path: String)
    
    /**
     * Obter uso de memória atual
     */
    external fun getCurrentMemoryUsageMB(): Int
    
    /**
     * Limpar caches
     */
    external fun clearCaches()
    
    // ========== INTERFACES CALLBACK ==========
    
    interface ProgressCallback {
        fun onProgress(stage: String, progress: Float)
        fun onLog(message: String)
        fun isCancelled(): Boolean
    }
    
    data class ReconstructionStats(
        val numImages: Int,
        val numCameras: Int,
        val numPoints: Int,
        val reprojectionError: Float,
        val coveragePercent: Float
    )
}
```

### 5.2.2 Implementação JNI em C++

```cpp
// app/src/main/cpp/native_bridge.cpp

#include <jni.h>
#include <string>
#include <memory>
#include <android/log.h>

#include "sfm/reconstruction_session.h"
#include "sfm/feature_extraction.h"
#include "ortho/fast_orthophoto.h"

#define LOG_TAG "FastStitching"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Namespace para funções JNI
extern "C" {

// ==================== FEATURE EXTRACTION ====================

JNIEXPORT jlong JNICALL
Java_com_faststitching_native_NativeBridge_extractFeatures(
    JNIEnv* env,
    jobject /* this */,
    jstring imagePath,
    jstring detectorType,
    jint maxFeatures
) {
    const char* path = env->GetStringUTFChars(imagePath, nullptr);
    const char* detector = env->GetStringUTFChars(detectorType, nullptr);
    
    LOGI("Extracting features from: %s using %s", path, detector);
    
    try {
        // Criar extrator baseado no tipo
        auto extractor = faststitching::FeatureExtractor::create(
            detector,
            static_cast<int>(maxFeatures)
        );
        
        // Carregar imagem
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty()) {
            LOGE("Failed to load image: %s", path);
            env->ReleaseStringUTFChars(imagePath, path);
            env->ReleaseStringUTFChars(detectorType, detector);
            return 0;
        }
        
        // Extrair features
        auto* features = new faststitching::Features();
        extractor->extract(image, features->keypoints, features->descriptors);
        
        LOGI("Extracted %zu features", features->keypoints.size());
        
        env->ReleaseStringUTFChars(imagePath, path);
        env->ReleaseStringUTFChars(detectorType, detector);
        
        // Retornar ponteiro como jlong
        return reinterpret_cast<jlong>(features);
        
    } catch (const std::exception& e) {
        LOGE("Exception in extractFeatures: %s", e.what());
        env->ReleaseStringUTFChars(imagePath, path);
        env->ReleaseStringUTFChars(detectorType, detector);
        return 0;
    }
}

JNIEXPORT jint JNICALL
Java_com_faststitching_native_NativeBridge_getFeatureCount(
    JNIEnv* env,
    jobject /* this */,
    jlong featuresHandle
) {
    if (featuresHandle == 0) return 0;
    auto* features = reinterpret_cast<faststitching::Features*>(featuresHandle);
    return static_cast<jint>(features->keypoints.size());
}

JNIEXPORT void JNICALL
Java_com_faststitching_native_NativeBridge_releaseFeatures(
    JNIEnv* env,
    jobject /* this */,
    jlong featuresHandle
) {
    if (featuresHandle != 0) {
        auto* features = reinterpret_cast<faststitching::Features*>(featuresHandle);
        delete features;
        LOGI("Released features");
    }
}

// ==================== RECONSTRUCTION ====================

JNIEXPORT jlong JNICALL
Java_com_faststitching_native_NativeBridge_createReconstructionSession(
    JNIEnv* env,
    jobject /* this */,
    jstring outputDir,
    jint numThreads,
    jboolean useFastMode
) {
    const char* dir = env->GetStringUTFChars(outputDir, nullptr);
    
    try {
        faststitching::ReconstructionConfig config;
        config.output_dir = dir;
        config.num_threads = static_cast<int>(numThreads);
        config.fast_mode = static_cast<bool>(useFastMode);
        
        auto* session = new faststitching::ReconstructionSession(config);
        
        env->ReleaseStringUTFChars(outputDir, dir);
        return reinterpret_cast<jlong>(session);
        
    } catch (const std::exception& e) {
        LOGE("Failed to create reconstruction session: %s", e.what());
        env->ReleaseStringUTFChars(outputDir, dir);
        return 0;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_faststitching_native_NativeBridge_addImageToReconstruction(
    JNIEnv* env,
    jobject /* this */,
    jlong sessionHandle,
    jstring imagePath,
    jfloat focalLengthMm,
    jfloat sensorWidthMm,
    jdouble latitude,
    jdouble longitude,
    jdouble altitude,
    jfloat yaw,
    jfloat pitch,
    jfloat roll
) {
    if (sessionHandle == 0) return JNI_FALSE;
    
    const char* path = env->GetStringUTFChars(imagePath, nullptr);
    auto* session = reinterpret_cast<faststitching::ReconstructionSession*>(sessionHandle);
    
    try {
        faststitching::ImageMetadata metadata;
        metadata.filepath = path;
        metadata.focal_length_mm = focalLengthMm;
        metadata.sensor_width_mm = sensorWidthMm;
        metadata.gps.latitude = latitude;
        metadata.gps.longitude = longitude;
        metadata.gps.altitude = altitude;
        metadata.orientation.yaw = yaw;
        metadata.orientation.pitch = pitch;
        metadata.orientation.roll = roll;
        
        bool result = session->addImage(metadata);
        
        env->ReleaseStringUTFChars(imagePath, path);
        return result ? JNI_TRUE : JNI_FALSE;
        
    } catch (const std::exception& e) {
        LOGE("Failed to add image: %s", e.what());
        env->ReleaseStringUTFChars(imagePath, path);
        return JNI_FALSE;
    }
}

// Classe helper para callbacks
class JNIProgressCallback : public faststitching::ProgressCallback {
private:
    JNIEnv* env_;
    jobject callback_;
    jmethodID onProgressMethod_;
    jmethodID onLogMethod_;
    jmethodID isCancelledMethod_;
    
public:
    JNIProgressCallback(JNIEnv* env, jobject callback) 
        : env_(env), callback_(callback) {
        jclass cls = env->GetObjectClass(callback);
        onProgressMethod_ = env->GetMethodID(cls, "onProgress", "(Ljava/lang/String;F)V");
        onLogMethod_ = env->GetMethodID(cls, "onLog", "(Ljava/lang/String;)V");
        isCancelledMethod_ = env->GetMethodID(cls, "isCancelled", "()Z");
    }
    
    void onProgress(const std::string& stage, float progress) override {
        jstring jStage = env_->NewStringUTF(stage.c_str());
        env_->CallVoidMethod(callback_, onProgressMethod_, jStage, progress);
        env_->DeleteLocalRef(jStage);
    }
    
    void onLog(const std::string& message) override {
        jstring jMsg = env_->NewStringUTF(message.c_str());
        env_->CallVoidMethod(callback_, onLogMethod_, jMsg);
        env_->DeleteLocalRef(jMsg);
    }
    
    bool isCancelled() override {
        return env_->CallBooleanMethod(callback_, isCancelledMethod_) == JNI_TRUE;
    }
};

JNIEXPORT jint JNICALL
Java_com_faststitching_native_NativeBridge_runReconstruction(
    JNIEnv* env,
    jobject /* this */,
    jlong sessionHandle,
    jobject progressCallback
) {
    if (sessionHandle == 0) return -1;
    
    auto* session = reinterpret_cast<faststitching::ReconstructionSession*>(sessionHandle);
    
    try {
        JNIProgressCallback callback(env, progressCallback);
        return session->run(&callback);
        
    } catch (const std::exception& e) {
        LOGE("Reconstruction failed: %s", e.what());
        return -1;
    }
}

// ==================== ORTHOPHOTO ====================

JNIEXPORT jstring JNICALL
Java_com_faststitching_native_NativeBridge_generateOrthophoto(
    JNIEnv* env,
    jobject /* this */,
    jlong sessionHandle,
    jstring outputPath,
    jfloat gsd,
    jint epsgCode,
    jobject progressCallback
) {
    if (sessionHandle == 0) return nullptr;
    
    const char* path = env->GetStringUTFChars(outputPath, nullptr);
    auto* session = reinterpret_cast<faststitching::ReconstructionSession*>(sessionHandle);
    
    try {
        JNIProgressCallback callback(env, progressCallback);
        
        faststitching::OrthophotoConfig config;
        config.output_path = path;
        config.gsd = gsd;
        config.epsg_code = epsgCode;
        
        faststitching::FastOrthophoto orthoGen;
        std::string result = orthoGen.generate(
            session->getReconstruction(),
            config,
            &callback
        );
        
        env->ReleaseStringUTFChars(outputPath, path);
        return env->NewStringUTF(result.c_str());
        
    } catch (const std::exception& e) {
        LOGE("Orthophoto generation failed: %s", e.what());
        env->ReleaseStringUTFChars(outputPath, path);
        return nullptr;
    }
}

// ==================== UTILITIES ====================

JNIEXPORT jstring JNICALL
Java_com_faststitching_native_NativeBridge_getLibraryInfo(
    JNIEnv* env,
    jobject /* this */
) {
    std::string info = "FastStitching Native Library v1.0.0\n";
    info += "OpenCV: " + std::string(CV_VERSION) + "\n";
    info += "Eigen: " + std::to_string(EIGEN_WORLD_VERSION) + "." +
            std::to_string(EIGEN_MAJOR_VERSION) + "." +
            std::to_string(EIGEN_MINOR_VERSION) + "\n";
    
    return env->NewStringUTF(info.c_str());
}

JNIEXPORT jint JNICALL
Java_com_faststitching_native_NativeBridge_getCurrentMemoryUsageMB(
    JNIEnv* env,
    jobject /* this */
) {
    // Ler de /proc/self/statm
    FILE* file = fopen("/proc/self/statm", "r");
    if (file) {
        long size, resident;
        if (fscanf(file, "%ld %ld", &size, &resident) == 2) {
            fclose(file);
            return static_cast<jint>(resident * sysconf(_SC_PAGESIZE) / (1024 * 1024));
        }
        fclose(file);
    }
    return 0;
}

} // extern "C"
```

---

## 5.3 COMPILAÇÃO NATIVA COM NDK

### 5.3.1 CMakeLists.txt Principal

```cmake
# native/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(FastStitchingNative VERSION 1.0.0 LANGUAGES CXX)

# Configurações Android
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")

# Detectar arquitetura
message(STATUS "Building for ABI: ${ANDROID_ABI}")
message(STATUS "Android NDK: ${ANDROID_NDK}")

# ========== THIRD PARTY LIBRARIES ==========

# OpenCV - pré-compilado para Android
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opencv/sdk/native/jni")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen (header-only)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen")
include_directories(${EIGEN3_INCLUDE_DIR})

# gflags
set(gflags_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags/build")
find_package(gflags REQUIRED)

# glog
set(glog_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glog/build")
find_package(glog REQUIRED)

# Ceres Solver
set(Ceres_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver/build")
find_package(Ceres REQUIRED)

# libtiff
set(TIFF_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtiff/include")
set(TIFF_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtiff/lib/${ANDROID_ABI}/libtiff.a")

# libgeotiff
set(GEOTIFF_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgeotiff/include")
set(GEOTIFF_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgeotiff/lib/${ANDROID_ABI}/libgeotiff.a")

# PROJ
set(PROJ_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/proj/include")
set(PROJ_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/proj/lib/${ANDROID_ABI}/libproj.a")

# ========== SFM CORE LIBRARY ==========

set(SFM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/sfm/feature_extraction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/sfm/feature_matching.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/sfm/two_view_geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/sfm/reconstruction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/sfm/bundle_adjustment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/sfm/triangulation.cpp
)

add_library(sfm_core SHARED ${SFM_SOURCES})

target_include_directories(sfm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(sfm_core
    ${OpenCV_LIBS}
    glog::glog
    gflags
    Ceres::ceres
    log
)

# ========== ORTHOPHOTO LIBRARY ==========

set(ORTHO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/ortho/fast_orthophoto.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/ortho/geotiff_writer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/ortho/image_warping.cpp
)

add_library(ortho_core SHARED ${ORTHO_SOURCES})

target_include_directories(ortho_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp
    ${TIFF_INCLUDE_DIR}
    ${GEOTIFF_INCLUDE_DIR}
    ${PROJ_INCLUDE_DIR}
)

target_link_libraries(ortho_core
    sfm_core
    ${TIFF_LIBRARY}
    ${GEOTIFF_LIBRARY}
    ${PROJ_LIBRARY}
    z
    log
)

# ========== MAIN JNI LIBRARY ==========

set(JNI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/native_bridge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/utils/memory_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp/utils/image_utils.cpp
)

add_library(faststitching SHARED ${JNI_SOURCES})

target_include_directories(faststitching PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp
)

target_link_libraries(faststitching
    sfm_core
    ortho_core
    android
    log
    jnigraphics
)

# ========== INSTALAÇÃO ==========

install(TARGETS sfm_core ortho_core faststitching
    LIBRARY DESTINATION lib/${ANDROID_ABI}
)
```

### 5.3.2 Build.gradle.kts (App Module)

```kotlin
// app/build.gradle.kts
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
    id("dagger.hilt.android.plugin")
}

android {
    namespace = "com.faststitching"
    compileSdk = 34
    
    defaultConfig {
        applicationId = "com.faststitching"
        minSdk = 26  // Android 8.0+
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
        
        // Apenas arm64 para tablets modernos
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        
        externalNativeBuild {
            cmake {
                cppFlags += listOf(
                    "-std=c++17",
                    "-O3",
                    "-ffast-math",
                    "-fPIC"
                )
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_TOOLCHAIN=clang",
                    "-DCMAKE_BUILD_TYPE=Release"
                )
            }
        }
    }
    
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
            isDebuggable = true
        }
    }
    
    externalNativeBuild {
        cmake {
            path = file("../native/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    
    buildFeatures {
        compose = true
        viewBinding = true
    }
    
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.8"
    }
    
    kotlinOptions {
        jvmTarget = "17"
    }
    
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    // Kotlin & Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // Jetpack Compose
    val composeBom = platform("androidx.compose:compose-bom:2024.02.00")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.8.2")
    
    // ViewModel & LiveData
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")
    
    // Room Database
    val roomVersion = "2.6.1"
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")
    
    // WorkManager para processamento em background
    implementation("androidx.work:work-runtime-ktx:2.9.0")
    
    // Hilt DI
    implementation("com.google.dagger:hilt-android:2.50")
    ksp("com.google.dagger:hilt-compiler:2.50")
    implementation("androidx.hilt:hilt-work:1.1.0")
    ksp("androidx.hilt:hilt-compiler:1.1.0")
    
    // CameraX
    val cameraXVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraXVersion")
    implementation("androidx.camera:camera-camera2:$cameraXVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraXVersion")
    implementation("androidx.camera:camera-view:$cameraXVersion")
    
    // Mapas offline (OSMdroid)
    implementation("org.osmdroid:osmdroid-android:6.1.18")
    
    // Exif para metadados de imagem
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    
    // Coil para carregamento de imagens
    implementation("io.coil-kt:coil-compose:2.5.0")
    
    // DataStore para preferências
    implementation("androidx.datastore:datastore-preferences:1.0.0")
    
    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation(composeBom)
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
}
```

### 5.3.3 Script de Build para Dependências Nativas

```bash
#!/bin/bash
# native/build_dependencies.sh
# Script para compilar todas as dependências nativas para Android

set -e

NDK_PATH="${ANDROID_NDK_HOME:-$HOME/Android/Sdk/ndk/26.1.10909125}"
API_LEVEL=26
ABI="arm64-v8a"
BUILD_DIR="$(pwd)/build"
INSTALL_DIR="$(pwd)/thirdparty"

CMAKE_TOOLCHAIN="$NDK_PATH/build/cmake/android.toolchain.cmake"

echo "=== Building native dependencies for Android ==="
echo "NDK: $NDK_PATH"
echo "API Level: $API_LEVEL"
echo "ABI: $ABI"

# Função helper
build_cmake_project() {
    local name=$1
    local src_dir=$2
    local cmake_args=${3:-""}
    
    echo "Building $name..."
    
    mkdir -p "$BUILD_DIR/$name"
    cd "$BUILD_DIR/$name"
    
    cmake "$src_dir" \
        -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_PLATFORM="android-$API_LEVEL" \
        -DANDROID_STL=c++_shared \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/$name" \
        $cmake_args
    
    make -j$(nproc)
    make install
    
    cd -
}

# 1. Build Eigen (header-only, apenas copiar)
echo "=== Setting up Eigen ==="
mkdir -p "$INSTALL_DIR/eigen"
cp -r thirdparty_src/eigen/Eigen "$INSTALL_DIR/eigen/"

# 2. Build gflags
build_cmake_project "gflags" "$(pwd)/thirdparty_src/gflags" \
    "-DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF"

# 3. Build glog
build_cmake_project "glog" "$(pwd)/thirdparty_src/glog" \
    "-DBUILD_SHARED_LIBS=ON -DWITH_GFLAGS=ON -Dgflags_DIR=$INSTALL_DIR/gflags/lib/cmake/gflags"

# 4. Build Ceres Solver
build_cmake_project "ceres-solver" "$(pwd)/thirdparty_src/ceres-solver" \
    "-DBUILD_SHARED_LIBS=ON \
     -DBUILD_TESTING=OFF \
     -DBUILD_EXAMPLES=OFF \
     -DMINIGLOG=OFF \
     -DEigen3_DIR=$INSTALL_DIR/eigen \
     -Dglog_DIR=$INSTALL_DIR/glog/lib/cmake/glog \
     -Dgflags_DIR=$INSTALL_DIR/gflags/lib/cmake/gflags"

# 5. Build libtiff
build_cmake_project "libtiff" "$(pwd)/thirdparty_src/libtiff" \
    "-DBUILD_SHARED_LIBS=OFF"

# 6. Build PROJ
build_cmake_project "proj" "$(pwd)/thirdparty_src/proj" \
    "-DBUILD_SHARED_LIBS=OFF \
     -DBUILD_TESTING=OFF \
     -DTIFF_LIBRARY=$INSTALL_DIR/libtiff/lib/libtiff.a \
     -DTIFF_INCLUDE_DIR=$INSTALL_DIR/libtiff/include"

# 7. Build libgeotiff
build_cmake_project "libgeotiff" "$(pwd)/thirdparty_src/libgeotiff" \
    "-DBUILD_SHARED_LIBS=OFF \
     -DWITH_UTILITIES=OFF \
     -DTIFF_LIBRARY=$INSTALL_DIR/libtiff/lib/libtiff.a \
     -DTIFF_INCLUDE_DIR=$INSTALL_DIR/libtiff/include \
     -DPROJ_LIBRARY=$INSTALL_DIR/proj/lib/libproj.a \
     -DPROJ_INCLUDE_DIR=$INSTALL_DIR/proj/include"

# 8. OpenCV (usar versão pré-compilada oficial)
echo "=== Setting up OpenCV ==="
OPENCV_VERSION="4.5.5"
OPENCV_ANDROID_URL="https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-android-sdk.zip"

if [ ! -d "$INSTALL_DIR/opencv" ]; then
    echo "Downloading OpenCV Android SDK..."
    wget -q "$OPENCV_ANDROID_URL" -O opencv-android.zip
    unzip -q opencv-android.zip
    mv OpenCV-android-sdk "$INSTALL_DIR/opencv"
    rm opencv-android.zip
fi

echo "=== Build complete ==="
echo "Libraries installed in: $INSTALL_DIR"
```

---

## 5.4 GESTÃO DE MEMÓRIA NO ANDROID

### 5.4.1 Memory Pool C++

```cpp
// app/src/main/cpp/utils/memory_pool.h
#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <cstdint>

namespace faststitching {

/**
 * Pool de memória para reutilização de buffers
 * Evita fragmentação e alocações frequentes
 */
class MemoryPool {
public:
    static MemoryPool& getInstance() {
        static MemoryPool instance;
        return instance;
    }
    
    // Alocar buffer do pool
    uint8_t* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Procurar buffer existente
        for (auto& buffer : free_buffers_) {
            if (buffer.capacity >= size) {
                buffer.in_use = true;
                return buffer.data.get();
            }
        }
        
        // Alocar novo buffer
        Buffer new_buffer;
        new_buffer.capacity = size;
        new_buffer.data = std::make_unique<uint8_t[]>(size);
        new_buffer.in_use = true;
        
        uint8_t* ptr = new_buffer.data.get();
        allocated_buffers_.push_back(std::move(new_buffer));
        
        total_allocated_ += size;
        return ptr;
    }
    
    // Liberar buffer para reutilização
    void release(uint8_t* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& buffer : allocated_buffers_) {
            if (buffer.data.get() == ptr) {
                buffer.in_use = false;
                free_buffers_.push_back(std::move(buffer));
                return;
            }
        }
    }
    
    // Limpar todos os buffers não utilizados
    void purge() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t freed = 0;
        for (auto& buffer : free_buffers_) {
            freed += buffer.capacity;
        }
        
        free_buffers_.clear();
        total_allocated_ -= freed;
    }
    
    // Obter uso atual de memória
    size_t getUsedMemory() const {
        return total_allocated_;
    }
    
private:
    struct Buffer {
        std::unique_ptr<uint8_t[]> data;
        size_t capacity = 0;
        bool in_use = false;
    };
    
    std::vector<Buffer> allocated_buffers_;
    std::vector<Buffer> free_buffers_;
    size_t total_allocated_ = 0;
    std::mutex mutex_;
    
    MemoryPool() = default;
};

// RAII wrapper
class PooledBuffer {
public:
    explicit PooledBuffer(size_t size) 
        : ptr_(MemoryPool::getInstance().allocate(size))
        , size_(size) {}
    
    ~PooledBuffer() {
        if (ptr_) {
            MemoryPool::getInstance().release(ptr_);
        }
    }
    
    uint8_t* data() { return ptr_; }
    size_t size() const { return size_; }
    
    // Move-only
    PooledBuffer(PooledBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
    }
    
    PooledBuffer& operator=(PooledBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                MemoryPool::getInstance().release(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;
    
private:
    uint8_t* ptr_;
    size_t size_;
};

} // namespace faststitching
```

### 5.4.2 Monitoramento de Memória (Kotlin)

```kotlin
// app/src/main/java/com/faststitching/utils/MemoryMonitor.kt
package com.faststitching.utils

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class MemoryMonitor @Inject constructor(
    private val context: Context
) {
    private val _memoryState = MutableStateFlow(MemoryState())
    val memoryState: StateFlow<MemoryState> = _memoryState
    
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) 
        as ActivityManager
    
    data class MemoryState(
        val usedHeapMB: Int = 0,
        val maxHeapMB: Int = 0,
        val usedNativeMB: Int = 0,
        val availableSystemMB: Int = 0,
        val isLowMemory: Boolean = false,
        val percentUsed: Float = 0f
    )
    
    fun update() {
        val runtime = Runtime.getRuntime()
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        
        val usedHeap = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        val maxHeap = runtime.maxMemory() / (1024 * 1024)
        val usedNative = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)
        val availableSystem = memInfo.availMem / (1024 * 1024)
        
        _memoryState.value = MemoryState(
            usedHeapMB = usedHeap.toInt(),
            maxHeapMB = maxHeap.toInt(),
            usedNativeMB = usedNative.toInt(),
            availableSystemMB = availableSystem.toInt(),
            isLowMemory = memInfo.lowMemory,
            percentUsed = (usedHeap.toFloat() / maxHeap.toFloat()) * 100
        )
    }
    
    fun shouldReduceMemoryUsage(): Boolean {
        update()
        return memoryState.value.percentUsed > 80f || memoryState.value.isLowMemory
    }
    
    fun getRecommendedBatchSize(imageWidth: Int, imageHeight: Int): Int {
        update()
        val state = memoryState.value
        
        // Estimar memória por imagem (3 canais * 4 bytes float)
        val bytesPerImage = imageWidth.toLong() * imageHeight * 3 * 4
        val mbPerImage = bytesPerImage / (1024 * 1024)
        
        // Usar no máximo 50% da memória disponível
        val availableForProcessing = (state.maxHeapMB - state.usedHeapMB) * 0.5
        
        return maxOf(1, (availableForProcessing / mbPerImage).toInt())
    }
}
```

---

## 5.5 CONCLUSÕES DA PARTE 1 DO CAPÍTULO 5

### 5.5.1 Resumo da Arquitetura

| Camada | Tecnologia | Responsabilidade |
|--------|------------|------------------|
| UI | Jetpack Compose | Interface de usuário |
| ViewModel | Kotlin + StateFlow | Estado e lógica de apresentação |
| Domain | Kotlin | Use cases e regras de negócio |
| Data | Room + FileSystem | Persistência |
| Native | C++ via JNI | Processamento pesado (SfM, Ortho) |
| Libraries | OpenCV, Ceres, Eigen | Algoritmos otimizados |

### 5.5.2 Decisões Técnicas Chave

1. **Apenas arm64-v8a**: Tablets modernos usam exclusivamente esta arquitetura
2. **C++17**: Suporte completo no NDK 26+
3. **Static linking**: Dependências menores (.a) linkadas estaticamente
4. **Memory pool**: Reutilização de buffers para evitar fragmentação
5. **WorkManager**: Processamento em background com suporte a retry

### 5.5.3 Próximos Passos (Parte 2)

- Implementação detalhada do pipeline de processamento
- UI com Jetpack Compose
- Exportação GeoTIFF e KML
- Estratégias de otimização para tablets

