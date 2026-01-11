# CAPÍTULO 5: BLUEPRINT DE IMPLEMENTAÇÃO ANDROID

## Parte 2: Implementação e Pipeline de Processamento

---

## 5.6 PIPELINE DE PROCESSAMENTO ANDROID

### 5.6.1 Fluxo de Processamento Completo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FAST STITCHING PROCESSING PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────┘

[1] IMPORT              [2] VALIDATE           [3] EXTRACT
┌─────────────┐         ┌─────────────┐        ┌─────────────┐
│ Select      │         │ Check EXIF  │        │ Resize      │
│ Images      │ ──────▶ │ GPS Tags    │ ─────▶ │ Images      │
│ (Gallery/   │         │ Focal Length│        │ (Max 2048px)│
│  Camera)    │         │ Orientation │        │             │
└─────────────┘         └─────────────┘        └─────────────┘
                                                      │
                        ┌─────────────────────────────┘
                        ▼
[4] FEATURES            [5] MATCHING           [6] SFM
┌─────────────┐         ┌─────────────┐        ┌─────────────┐
│ AKAZE/ORB   │         │ FLANN       │        │ Incremental │
│ Detection   │ ──────▶ │ k-NN Match  │ ─────▶ │ Reconstruc- │
│ (per image) │         │ Ratio Test  │        │ tion        │
│ ~2000 feat  │         │ Geometric   │        │ (P3P+BA)    │
└─────────────┘         └─────────────┘        └─────────────┘
                                                      │
                        ┌─────────────────────────────┘
                        ▼
[7] GEOREF              [8] FAST ORTHO         [9] EXPORT
┌─────────────┐         ┌─────────────┐        ┌─────────────┐
│ GPS ──▶ UTM │         │ Direct      │        │ GeoTIFF     │
│ Transform   │ ──────▶ │ Projection  │ ─────▶ │ + Tiles     │
│ Helmert     │         │ (No MVS)    │        │ + KML       │
│ 7-params    │         │ Blending    │        │             │
└─────────────┘         └─────────────┘        └─────────────┘
```

### 5.6.2 UseCase de Processamento

```kotlin
// app/src/main/java/com/faststitching/domain/usecase/ProcessImagesUseCase.kt
package com.faststitching.domain.usecase

import com.faststitching.domain.model.*
import com.faststitching.domain.repository.ImageRepository
import com.faststitching.native.NativeBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import javax.inject.Inject

class ProcessImagesUseCase @Inject constructor(
    private val imageRepository: ImageRepository,
    private val nativeBridge: NativeBridge
) {
    sealed class ProcessingState {
        object Idle : ProcessingState()
        data class Validating(val progress: Float) : ProcessingState()
        data class ExtractingFeatures(val progress: Float, val currentImage: String) : ProcessingState()
        data class Matching(val progress: Float, val matchedPairs: Int) : ProcessingState()
        data class Reconstructing(val progress: Float, val cameras: Int, val points: Int) : ProcessingState()
        data class GeneratingOrthophoto(val progress: Float) : ProcessingState()
        data class Exporting(val progress: Float) : ProcessingState()
        data class Completed(val result: ProcessingResult) : ProcessingState()
        data class Error(val message: String, val recoverable: Boolean) : ProcessingState()
    }
    
    data class ProcessingConfig(
        val maxImageSize: Int = 2048,
        val featureType: FeatureType = FeatureType.AKAZE,
        val maxFeatures: Int = 2000,
        val gsd: Float = 0.05f,  // 5cm/pixel
        val fastMode: Boolean = true,
        val outputFormat: OutputFormat = OutputFormat.GEOTIFF
    )
    
    fun execute(
        projectId: Long,
        config: ProcessingConfig
    ): Flow<ProcessingState> = flow {
        emit(ProcessingState.Idle)
        
        try {
            // 1. Obter imagens do projeto
            val images = imageRepository.getProjectImages(projectId)
            if (images.isEmpty()) {
                emit(ProcessingState.Error("No images found", recoverable = false))
                return@flow
            }
            
            // 2. Validar imagens
            emit(ProcessingState.Validating(0f))
            val validatedImages = validateImages(images) { progress ->
                emit(ProcessingState.Validating(progress))
            }
            
            if (validatedImages.size < 2) {
                emit(ProcessingState.Error(
                    "Need at least 2 valid images with GPS",
                    recoverable = false
                ))
                return@flow
            }
            
            // 3. Criar sessão de reconstrução nativa
            val outputDir = imageRepository.getProjectOutputDir(projectId)
            val sessionHandle = withContext(Dispatchers.IO) {
                nativeBridge.createReconstructionSession(
                    outputDir = outputDir.absolutePath,
                    numThreads = Runtime.getRuntime().availableProcessors(),
                    useFastMode = config.fastMode
                )
            }
            
            if (sessionHandle == 0L) {
                emit(ProcessingState.Error("Failed to create native session", recoverable = true))
                return@flow
            }
            
            try {
                // 4. Processar cada imagem
                var featuresExtracted = 0
                for ((index, image) in validatedImages.withIndex()) {
                    emit(ProcessingState.ExtractingFeatures(
                        progress = index.toFloat() / validatedImages.size,
                        currentImage = image.filename
                    ))
                    
                    // Redimensionar se necessário
                    val processedPath = withContext(Dispatchers.IO) {
                        imageRepository.prepareImageForProcessing(
                            image,
                            config.maxImageSize
                        )
                    }
                    
                    // Adicionar à reconstrução
                    val success = withContext(Dispatchers.IO) {
                        nativeBridge.addImageToReconstruction(
                            sessionHandle = sessionHandle,
                            imagePath = processedPath.absolutePath,
                            focalLengthMm = image.focalLengthMm,
                            sensorWidthMm = image.sensorWidthMm,
                            latitude = image.latitude,
                            longitude = image.longitude,
                            altitude = image.altitude,
                            yaw = image.yaw,
                            pitch = image.pitch,
                            roll = image.roll
                        )
                    }
                    
                    if (success) featuresExtracted++
                }
                
                // 5. Executar reconstrução
                emit(ProcessingState.Reconstructing(0f, 0, 0))
                
                val progressCallback = object : NativeBridge.ProgressCallback {
                    override fun onProgress(stage: String, progress: Float) {
                        // Atualizar estado baseado no stage
                    }
                    override fun onLog(message: String) {
                        // Log para debugging
                    }
                    override fun isCancelled(): Boolean = false
                }
                
                val reconstructionResult = withContext(Dispatchers.Default) {
                    nativeBridge.runReconstruction(sessionHandle, progressCallback)
                }
                
                if (reconstructionResult != 0) {
                    emit(ProcessingState.Error(
                        "Reconstruction failed with code: $reconstructionResult",
                        recoverable = true
                    ))
                    return@flow
                }
                
                // 6. Gerar orthophoto
                emit(ProcessingState.GeneratingOrthophoto(0f))
                
                val orthoPath = withContext(Dispatchers.IO) {
                    nativeBridge.generateOrthophoto(
                        sessionHandle = sessionHandle,
                        outputPath = "$outputDir/orthophoto.tif",
                        gsd = config.gsd,
                        epsgCode = 32632,  // UTM zone calculada dinamicamente
                        progressCallback = progressCallback
                    )
                }
                
                // 7. Exportar resultados
                emit(ProcessingState.Exporting(0f))
                
                val result = ProcessingResult(
                    orthophotoPath = orthoPath,
                    numImages = validatedImages.size,
                    numCameras = nativeBridge.getReconstructionStats(sessionHandle).numCameras,
                    numPoints = nativeBridge.getReconstructionStats(sessionHandle).numPoints,
                    reprojectionError = nativeBridge.getReconstructionStats(sessionHandle).reprojectionError,
                    gsd = config.gsd,
                    processingTimeSeconds = 0  // TODO: calcular
                )
                
                emit(ProcessingState.Completed(result))
                
            } finally {
                // Sempre liberar sessão nativa
                withContext(Dispatchers.IO) {
                    nativeBridge.releaseReconstructionSession(sessionHandle)
                }
            }
            
        } catch (e: Exception) {
            emit(ProcessingState.Error(
                message = e.message ?: "Unknown error",
                recoverable = true
            ))
        }
    }
    
    private suspend fun validateImages(
        images: List<GeoImage>,
        onProgress: suspend (Float) -> Unit
    ): List<GeoImage> {
        val valid = mutableListOf<GeoImage>()
        
        for ((index, image) in images.withIndex()) {
            onProgress(index.toFloat() / images.size)
            
            // Verificar se tem GPS válido
            if (image.latitude != 0.0 && image.longitude != 0.0) {
                // Verificar se tem focal length
                if (image.focalLengthMm > 0) {
                    valid.add(image)
                }
            }
        }
        
        return valid
    }
}
```

### 5.6.3 WorkManager para Background Processing

```kotlin
// app/src/main/java/com/faststitching/data/worker/ProcessingWorker.kt
package com.faststitching.data.worker

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.*
import com.faststitching.domain.usecase.ProcessImagesUseCase
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import kotlinx.coroutines.flow.collect

@HiltWorker
class ProcessingWorker @AssistedInject constructor(
    @Assisted context: Context,
    @Assisted params: WorkerParameters,
    private val processImagesUseCase: ProcessImagesUseCase
) : CoroutineWorker(context, params) {
    
    companion object {
        const val KEY_PROJECT_ID = "project_id"
        const val KEY_MAX_IMAGE_SIZE = "max_image_size"
        const val KEY_FAST_MODE = "fast_mode"
        const val KEY_GSD = "gsd"
        
        const val KEY_PROGRESS = "progress"
        const val KEY_STAGE = "stage"
        const val KEY_RESULT_PATH = "result_path"
        
        fun createRequest(
            projectId: Long,
            config: ProcessImagesUseCase.ProcessingConfig
        ): OneTimeWorkRequest {
            val data = workDataOf(
                KEY_PROJECT_ID to projectId,
                KEY_MAX_IMAGE_SIZE to config.maxImageSize,
                KEY_FAST_MODE to config.fastMode,
                KEY_GSD to config.gsd
            )
            
            val constraints = Constraints.Builder()
                .setRequiresBatteryNotLow(true)
                .setRequiresStorageNotLow(true)
                .build()
            
            return OneTimeWorkRequestBuilder<ProcessingWorker>()
                .setInputData(data)
                .setConstraints(constraints)
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    WorkRequest.MIN_BACKOFF_MILLIS,
                    java.util.concurrent.TimeUnit.MILLISECONDS
                )
                .build()
        }
    }
    
    override suspend fun doWork(): Result {
        val projectId = inputData.getLong(KEY_PROJECT_ID, -1)
        if (projectId < 0) return Result.failure()
        
        val config = ProcessImagesUseCase.ProcessingConfig(
            maxImageSize = inputData.getInt(KEY_MAX_IMAGE_SIZE, 2048),
            fastMode = inputData.getBoolean(KEY_FAST_MODE, true),
            gsd = inputData.getFloat(KEY_GSD, 0.05f)
        )
        
        var resultPath: String? = null
        
        processImagesUseCase.execute(projectId, config).collect { state ->
            when (state) {
                is ProcessImagesUseCase.ProcessingState.Validating -> {
                    setProgress(workDataOf(
                        KEY_STAGE to "Validating",
                        KEY_PROGRESS to state.progress
                    ))
                }
                is ProcessImagesUseCase.ProcessingState.ExtractingFeatures -> {
                    setProgress(workDataOf(
                        KEY_STAGE to "Extracting Features",
                        KEY_PROGRESS to state.progress
                    ))
                }
                is ProcessImagesUseCase.ProcessingState.Reconstructing -> {
                    setProgress(workDataOf(
                        KEY_STAGE to "Reconstructing",
                        KEY_PROGRESS to state.progress
                    ))
                }
                is ProcessImagesUseCase.ProcessingState.GeneratingOrthophoto -> {
                    setProgress(workDataOf(
                        KEY_STAGE to "Generating Orthophoto",
                        KEY_PROGRESS to state.progress
                    ))
                }
                is ProcessImagesUseCase.ProcessingState.Completed -> {
                    resultPath = state.result.orthophotoPath
                }
                else -> {}
            }
        }
        
        return if (resultPath != null) {
            Result.success(workDataOf(KEY_RESULT_PATH to resultPath))
        } else {
            Result.retry()
        }
    }
}
```

---

## 5.7 INTERFACE DE USUÁRIO COM JETPACK COMPOSE

### 5.7.1 Tela Principal

```kotlin
// app/src/main/java/com/faststitching/presentation/ui/home/HomeScreen.kt
package com.faststitching.presentation.ui.home

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.faststitching.domain.model.Project

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    viewModel: HomeViewModel = hiltViewModel(),
    onNavigateToProject: (Long) -> Unit,
    onNavigateToCapture: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Fast Stitching") },
                actions = {
                    IconButton(onClick = { /* Settings */ }) {
                        Icon(Icons.Default.Settings, "Settings")
                    }
                }
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = onNavigateToCapture,
                icon = { Icon(Icons.Default.Add, "New Project") },
                text = { Text("New Project") }
            )
        }
    ) { paddingValues ->
        when (val state = uiState) {
            is HomeUiState.Loading -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            }
            is HomeUiState.Empty -> {
                EmptyProjectsView(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues)
                )
            }
            is HomeUiState.Success -> {
                ProjectsList(
                    projects = state.projects,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues),
                    onProjectClick = onNavigateToProject,
                    onDeleteProject = viewModel::deleteProject
                )
            }
        }
    }
}

@Composable
private fun ProjectsList(
    projects: List<Project>,
    modifier: Modifier = Modifier,
    onProjectClick: (Long) -> Unit,
    onDeleteProject: (Long) -> Unit
) {
    LazyColumn(
        modifier = modifier,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        items(projects, key = { it.id }) { project ->
            ProjectCard(
                project = project,
                onClick = { onProjectClick(project.id) },
                onDelete = { onDeleteProject(project.id) }
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ProjectCard(
    project: Project,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Thumbnail
            Surface(
                modifier = Modifier.size(64.dp),
                shape = MaterialTheme.shapes.small,
                color = MaterialTheme.colorScheme.surfaceVariant
            ) {
                if (project.thumbnailPath != null) {
                    // AsyncImage com Coil
                }
            }
            
            Spacer(modifier = Modifier.width(16.dp))
            
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = project.name,
                    style = MaterialTheme.typography.titleMedium
                )
                Text(
                    text = "${project.imageCount} images",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = project.statusText,
                    style = MaterialTheme.typography.bodySmall,
                    color = when (project.status) {
                        Project.Status.COMPLETED -> MaterialTheme.colorScheme.primary
                        Project.Status.PROCESSING -> MaterialTheme.colorScheme.tertiary
                        Project.Status.ERROR -> MaterialTheme.colorScheme.error
                        else -> MaterialTheme.colorScheme.onSurfaceVariant
                    }
                )
            }
            
            IconButton(onClick = onDelete) {
                Icon(
                    Icons.Default.Delete,
                    contentDescription = "Delete",
                    tint = MaterialTheme.colorScheme.error
                )
            }
        }
    }
}

@Composable
private fun EmptyProjectsView(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            Icons.Default.PhotoLibrary,
            contentDescription = null,
            modifier = Modifier.size(96.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "No projects yet",
            style = MaterialTheme.typography.titleLarge
        )
        Text(
            "Create a new project to get started",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}
```

### 5.7.2 Tela de Processamento

```kotlin
// app/src/main/java/com/faststitching/presentation/ui/processing/ProcessingScreen.kt
package com.faststitching.presentation.ui.processing

import androidx.compose.animation.core.*
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun ProcessingScreen(
    projectId: Long,
    viewModel: ProcessingViewModel = hiltViewModel(),
    onNavigateBack: () -> Unit,
    onNavigateToResult: (String) -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    
    LaunchedEffect(projectId) {
        viewModel.startProcessing(projectId)
    }
    
    // Observar conclusão
    LaunchedEffect(uiState) {
        if (uiState is ProcessingUiState.Completed) {
            val path = (uiState as ProcessingUiState.Completed).resultPath
            onNavigateToResult(path)
        }
    }
    
    Scaffold { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            when (val state = uiState) {
                is ProcessingUiState.Processing -> {
                    ProcessingContent(
                        stage = state.stage,
                        progress = state.progress,
                        details = state.details
                    )
                }
                is ProcessingUiState.Error -> {
                    ErrorContent(
                        message = state.message,
                        onRetry = { viewModel.startProcessing(projectId) },
                        onCancel = onNavigateBack
                    )
                }
                else -> {
                    CircularProgressIndicator()
                }
            }
        }
    }
}

@Composable
private fun ProcessingContent(
    stage: String,
    progress: Float,
    details: String
) {
    // Animação de rotação
    val infiniteTransition = rememberInfiniteTransition(label = "rotation")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "rotation"
    )
    
    Icon(
        Icons.Default.Sync,
        contentDescription = null,
        modifier = Modifier
            .size(96.dp)
            .rotate(rotation),
        tint = MaterialTheme.colorScheme.primary
    )
    
    Spacer(modifier = Modifier.height(32.dp))
    
    Text(
        text = stage,
        style = MaterialTheme.typography.headlineSmall
    )
    
    Spacer(modifier = Modifier.height(16.dp))
    
    LinearProgressIndicator(
        progress = { progress },
        modifier = Modifier
            .fillMaxWidth()
            .height(8.dp),
    )
    
    Spacer(modifier = Modifier.height(8.dp))
    
    Text(
        text = "${(progress * 100).toInt()}%",
        style = MaterialTheme.typography.titleMedium
    )
    
    Spacer(modifier = Modifier.height(16.dp))
    
    Text(
        text = details,
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}

@Composable
private fun ErrorContent(
    message: String,
    onRetry: () -> Unit,
    onCancel: () -> Unit
) {
    Icon(
        Icons.Default.Error,
        contentDescription = null,
        modifier = Modifier.size(96.dp),
        tint = MaterialTheme.colorScheme.error
    )
    
    Spacer(modifier = Modifier.height(16.dp))
    
    Text(
        text = "Processing Failed",
        style = MaterialTheme.typography.headlineSmall
    )
    
    Spacer(modifier = Modifier.height(8.dp))
    
    Text(
        text = message,
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
    
    Spacer(modifier = Modifier.height(24.dp))
    
    Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        OutlinedButton(onClick = onCancel) {
            Text("Cancel")
        }
        Button(onClick = onRetry) {
            Text("Retry")
        }
    }
}
```

---

## 5.8 EXPORTAÇÃO DE RESULTADOS

### 5.8.1 GeoTIFF Writer (C++)

```cpp
// app/src/main/cpp/ortho/geotiff_writer.cpp
#include "geotiff_writer.h"
#include <tiffio.h>
#include <geotiff.h>
#include <geo_normalize.h>
#include <xtiffio.h>

namespace faststitching {

bool GeoTiffWriter::write(
    const cv::Mat& image,
    const GeoTransform& transform,
    int epsg_code,
    const std::string& output_path
) {
    // Abrir arquivo TIFF
    TIFF* tif = XTIFFOpen(output_path.c_str(), "w");
    if (!tif) {
        return false;
    }
    
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    
    // Configurar tags TIFF básicas
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 
                 channels == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);
    
    // Compressão
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    TIFFSetField(tif, TIFFTAG_PREDICTOR, 2);  // Horizontal differencing
    
    // Tiled para melhor acesso randômico
    TIFFSetField(tif, TIFFTAG_TILEWIDTH, 256);
    TIFFSetField(tif, TIFFTAG_TILELENGTH, 256);
    
    // Configurar GeoTIFF
    GTIF* gtif = GTIFNew(tif);
    if (!gtif) {
        XTIFFClose(tif);
        return false;
    }
    
    // Model Type: Projected
    GTIFKeySet(gtif, GTModelTypeGeoKey, TYPE_SHORT, 1, ModelTypeProjected);
    
    // Raster Type: Pixel is Area
    GTIFKeySet(gtif, GTRasterTypeGeoKey, TYPE_SHORT, 1, RasterPixelIsArea);
    
    // Projected CRS (ex: UTM)
    GTIFKeySet(gtif, ProjectedCSTypeGeoKey, TYPE_SHORT, 1, epsg_code);
    
    // GeoTransform: [originX, pixelSizeX, rotationX, originY, rotationY, pixelSizeY]
    double tiepoint[6] = {0, 0, 0, transform.origin_x, transform.origin_y, 0};
    double pixelscale[3] = {transform.pixel_size_x, transform.pixel_size_y, 0};
    
    TIFFSetField(tif, TIFFTAG_GEOTIEPOINTS, 6, tiepoint);
    TIFFSetField(tif, TIFFTAG_GEOPIXELSCALE, 3, pixelscale);
    
    // Escrever keys GeoTIFF
    GTIFWriteKeys(gtif);
    
    // Escrever dados da imagem (tiled)
    int tile_width = 256;
    int tile_height = 256;
    int tile_size = tile_width * tile_height * channels;
    std::vector<uint8_t> tile_buffer(tile_size);
    
    for (int y = 0; y < height; y += tile_height) {
        for (int x = 0; x < width; x += tile_width) {
            // Copiar dados do tile
            int actual_width = std::min(tile_width, width - x);
            int actual_height = std::min(tile_height, height - y);
            
            std::fill(tile_buffer.begin(), tile_buffer.end(), 0);
            
            for (int ty = 0; ty < actual_height; ty++) {
                const uint8_t* src = image.ptr<uint8_t>(y + ty) + x * channels;
                uint8_t* dst = tile_buffer.data() + ty * tile_width * channels;
                std::memcpy(dst, src, actual_width * channels);
            }
            
            TIFFWriteTile(tif, tile_buffer.data(), x, y, 0, 0);
        }
    }
    
    // Cleanup
    GTIFFree(gtif);
    XTIFFClose(tif);
    
    return true;
}

} // namespace faststitching
```

### 5.8.2 KML Exporter (Kotlin)

```kotlin
// app/src/main/java/com/faststitching/data/export/KmlExporter.kt
package com.faststitching.data.export

import com.faststitching.domain.model.ProcessingResult
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class KmlExporter {
    
    fun export(
        result: ProcessingResult,
        outputFile: File,
        projectName: String
    ): Boolean {
        val kml = buildKml(result, projectName)
        
        return try {
            outputFile.writeText(kml)
            true
        } catch (e: Exception) {
            false
        }
    }
    
    private fun buildKml(result: ProcessingResult, projectName: String): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        val timestamp = dateFormat.format(Date())
        
        return """
            <?xml version="1.0" encoding="UTF-8"?>
            <kml xmlns="http://www.opengis.net/kml/2.2">
                <Document>
                    <name>$projectName</name>
                    <description>
                        Orthophoto generated by Fast Stitching
                        Images: ${result.numImages}
                        GSD: ${String.format("%.2f", result.gsd * 100)} cm/pixel
                        Generated: $timestamp
                    </description>
                    
                    <Style id="orthophoto">
                        <IconStyle>
                            <scale>1.0</scale>
                        </IconStyle>
                    </Style>
                    
                    <GroundOverlay>
                        <name>Orthophoto</name>
                        <Icon>
                            <href>${File(result.orthophotoPath).name}</href>
                        </Icon>
                        <LatLonBox>
                            <north>${result.bounds.north}</north>
                            <south>${result.bounds.south}</south>
                            <east>${result.bounds.east}</east>
                            <west>${result.bounds.west}</west>
                        </LatLonBox>
                    </GroundOverlay>
                    
                    ${generateCameraPlacemarks(result)}
                    
                </Document>
            </kml>
        """.trimIndent()
    }
    
    private fun generateCameraPlacemarks(result: ProcessingResult): String {
        return result.cameras.mapIndexed { index, camera ->
            """
            <Placemark>
                <name>Camera ${index + 1}</name>
                <description>
                    File: ${camera.filename}
                    Altitude: ${String.format("%.1f", camera.altitude)} m
                </description>
                <Point>
                    <coordinates>${camera.longitude},${camera.latitude},${camera.altitude}</coordinates>
                </Point>
            </Placemark>
            """
        }.joinToString("\n")
    }
}
```

---

## 5.9 OTIMIZAÇÕES PARA TABLET

### 5.9.1 Detecção de Capacidades do Dispositivo

```kotlin
// app/src/main/java/com/faststitching/utils/DeviceCapabilities.kt
package com.faststitching.utils

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class DeviceCapabilities @Inject constructor(
    private val context: Context
) {
    data class Capabilities(
        val isTablet: Boolean,
        val totalRamMB: Int,
        val availableRamMB: Int,
        val cpuCores: Int,
        val is64Bit: Boolean,
        val recommendedMaxImageSize: Int,
        val recommendedBatchSize: Int,
        val recommendedThreadCount: Int
    )
    
    fun analyze(): Capabilities {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) 
            as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        
        val totalRam = (memInfo.totalMem / (1024 * 1024)).toInt()
        val availableRam = (memInfo.availMem / (1024 * 1024)).toInt()
        val cpuCores = Runtime.getRuntime().availableProcessors()
        val is64Bit = Build.SUPPORTED_64_BIT_ABIS.isNotEmpty()
        
        // Determinar se é tablet (heurística simples)
        val isTablet = context.resources.configuration.smallestScreenWidthDp >= 600
        
        // Calcular recomendações baseadas em recursos
        val recommendedMaxImageSize = when {
            totalRam >= 8192 -> 4096  // 8GB+ RAM
            totalRam >= 4096 -> 2048  // 4GB+ RAM
            totalRam >= 2048 -> 1024  // 2GB+ RAM
            else -> 800               // Low-end
        }
        
        val recommendedBatchSize = when {
            totalRam >= 8192 -> 20
            totalRam >= 4096 -> 10
            totalRam >= 2048 -> 5
            else -> 2
        }
        
        val recommendedThreadCount = when {
            cpuCores >= 8 -> 6
            cpuCores >= 4 -> 4
            else -> 2
        }
        
        return Capabilities(
            isTablet = isTablet,
            totalRamMB = totalRam,
            availableRamMB = availableRam,
            cpuCores = cpuCores,
            is64Bit = is64Bit,
            recommendedMaxImageSize = recommendedMaxImageSize,
            recommendedBatchSize = recommendedBatchSize,
            recommendedThreadCount = recommendedThreadCount
        )
    }
}
```

### 5.9.2 Configurações Automáticas

```kotlin
// app/src/main/java/com/faststitching/domain/usecase/ConfigureProcessingUseCase.kt
package com.faststitching.domain.usecase

import com.faststitching.utils.DeviceCapabilities
import javax.inject.Inject

class ConfigureProcessingUseCase @Inject constructor(
    private val deviceCapabilities: DeviceCapabilities
) {
    fun getOptimalConfig(imageCount: Int): ProcessImagesUseCase.ProcessingConfig {
        val caps = deviceCapabilities.analyze()
        
        // Ajustar baseado no número de imagens
        val maxImageSize = when {
            imageCount <= 20 -> caps.recommendedMaxImageSize
            imageCount <= 50 -> minOf(caps.recommendedMaxImageSize, 2048)
            imageCount <= 100 -> minOf(caps.recommendedMaxImageSize, 1024)
            else -> 800
        }
        
        // Usar AKAZE para dispositivos mais fracos, SIFT para mais fortes
        val featureType = when {
            caps.totalRamMB >= 4096 && caps.cpuCores >= 4 -> FeatureType.SIFT
            caps.totalRamMB >= 2048 -> FeatureType.AKAZE
            else -> FeatureType.ORB
        }
        
        // Max features
        val maxFeatures = when {
            caps.totalRamMB >= 4096 -> 4000
            caps.totalRamMB >= 2048 -> 2000
            else -> 1000
        }
        
        return ProcessImagesUseCase.ProcessingConfig(
            maxImageSize = maxImageSize,
            featureType = featureType,
            maxFeatures = maxFeatures,
            gsd = 0.05f,  // 5cm default
            fastMode = true
        )
    }
}
```

---

## 5.10 DIAGRAMA DE SEQUÊNCIA COMPLETO

```
┌──────┐   ┌──────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────┐
│  UI  │   │ ViewModel│   │  UseCase   │   │ NativeBridge │   │  C++ Libs  │
└──┬───┘   └────┬─────┘   └─────┬──────┘   └──────┬───────┘   └─────┬──────┘
   │            │               │                  │                 │
   │ Start      │               │                  │                 │
   │──────────▶│               │                  │                 │
   │            │ execute()     │                  │                 │
   │            │──────────────▶│                  │                 │
   │            │               │ createSession()  │                 │
   │            │               │─────────────────▶│                 │
   │            │               │                  │ new Session()   │
   │            │               │                  │────────────────▶│
   │            │               │                  │◀────────────────│
   │            │               │◀─────────────────│                 │
   │            │               │                  │                 │
   │  [loop]    │               │ addImage()       │                 │
   │            │               │─────────────────▶│                 │
   │            │               │                  │ extractFeatures │
   │            │               │                  │────────────────▶│
   │            │               │                  │◀────────────────│
   │ progress   │◀──────────────│                  │                 │
   │◀───────────│               │                  │                 │
   │            │               │                  │                 │
   │            │               │ runReconstruct() │                 │
   │            │               │─────────────────▶│                 │
   │            │               │                  │ matching        │
   │            │               │                  │────────────────▶│
   │            │               │                  │◀────────────────│
   │            │               │                  │ incremental SfM │
   │            │               │                  │────────────────▶│
   │            │               │                  │◀────────────────│
   │            │               │                  │ bundleAdjust    │
   │            │               │                  │────────────────▶│
   │            │               │                  │◀────────────────│
   │ progress   │◀──────────────│                  │                 │
   │◀───────────│               │                  │                 │
   │            │               │                  │                 │
   │            │               │ generateOrtho()  │                 │
   │            │               │─────────────────▶│                 │
   │            │               │                  │ fastOrthophoto  │
   │            │               │                  │────────────────▶│
   │            │               │                  │◀────────────────│
   │            │               │                  │ writeGeoTIFF    │
   │            │               │                  │────────────────▶│
   │            │               │                  │◀────────────────│
   │            │               │◀─────────────────│                 │
   │ completed  │◀──────────────│                  │                 │
   │◀───────────│               │                  │                 │
   │            │               │                  │                 │
```

---

## 5.11 CONCLUSÕES E PRÓXIMOS PASSOS

### 5.11.1 Resumo da Implementação

| Componente | Tecnologia | Status |
|------------|------------|--------|
| UI | Jetpack Compose | Especificado |
| ViewModel | Kotlin StateFlow | Especificado |
| Background | WorkManager | Especificado |
| Database | Room | Especificado |
| Native Core | C++17 / NDK 26 | Especificado |
| SfM Engine | OpenCV + Ceres | A implementar |
| Orthophoto | Custom + GeoTIFF | A implementar |
| Export | GeoTIFF + KML | Especificado |

### 5.11.2 Roadmap de Desenvolvimento

```
Fase 1: Fundação (4 semanas)
├── Setup projeto Android
├── Build sistema nativo (NDK)
├── Compilar dependências (OpenCV, Ceres, etc.)
└── JNI básico funcionando

Fase 2: Core SfM (6 semanas)
├── Feature extraction (AKAZE/ORB)
├── Feature matching (FLANN)
├── Two-view geometry (Essential Matrix)
├── Incremental reconstruction
└── Bundle adjustment (Ceres)

Fase 3: Fast Orthophoto (4 semanas)
├── Projeção direta (sem MVS)
├── Image warping
├── Blending (feather)
└── GeoTIFF export

Fase 4: UI & Polish (4 semanas)
├── Jetpack Compose UI completa
├── CameraX integration
├── Map viewer (OSMdroid)
├── Export (KML, tiles)
└── Performance tuning

Fase 5: Testing & Release (2 semanas)
├── Unit tests
├── Integration tests
├── Performance benchmarks
└── Play Store release
```

### 5.11.3 Métricas de Sucesso

| Métrica | Target |
|---------|--------|
| Tempo 50 imagens | < 5 minutos |
| RAM máxima | < 2 GB |
| Precisão GSD | ± 10% do configurado |
| Crash rate | < 1% |
| APK size | < 100 MB |

### 5.11.4 Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Memória insuficiente | Alto | Image downscaling, batch processing |
| Bundle adjustment lento | Médio | Simplificar BA, usar subset de pontos |
| GeoTIFF muito grande | Médio | Tiling, compressão, COG |
| Precisão insuficiente | Médio | Permitir mais iterações de BA |
| Compatibilidade NDK | Baixo | Testar em múltiplos dispositivos |

---

## FIM DO RELATÓRIO TÉCNICO

Este documento completo fornece a análise exaustiva do código-fonte do WebODM/ODM e o blueprint detalhado para implementação de um aplicativo Android "Fast Stitching" 100% offline para tablets.

