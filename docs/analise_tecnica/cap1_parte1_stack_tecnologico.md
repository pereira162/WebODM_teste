# CAPÍTULO 1: STACK TECNOLÓGICO E PORTABILIDADE

## Parte 1: Arquitetura Geral e Linguagens

---

## 1.1 VISÃO GERAL DA ARQUITETURA

O ecossistema OpenDroneMap/WebODM é composto por uma arquitetura de **três camadas** que separa claramente as responsabilidades de cada componente:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CAMADA 1: FRONTEND                        │
│  WebODM (Django + React)                                         │
│  ├── Interface Web para usuários                                 │
│  ├── API REST para integração                                    │
│  └── Gerenciamento de projetos/tarefas                          │
├─────────────────────────────────────────────────────────────────┤
│                     CAMADA 2: ORQUESTRAÇÃO                       │
│  NodeODM (Node.js)                                               │
│  ├── API REST para processamento                                 │
│  ├── Fila de tarefas                                             │
│  └── Comunicação com engine ODM                                  │
├─────────────────────────────────────────────────────────────────┤
│                    CAMADA 3: ENGINE DE PROCESSAMENTO             │
│  ODM Core (Python + C++ SuperBuild)                              │
│  ├── OpenSfM (Structure from Motion)                             │
│  ├── OpenMVS (Multi-View Stereo)                                 │
│  ├── PDAL (Point Data Abstraction Library)                       │
│  └── GDAL/OGR (Geospatial Data Abstraction)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1.1 Fluxo de Dados entre Camadas

```
[Usuário] → [WebODM:8000] → [NodeODM:3000] → [ODM Engine] → [Assets]
     ↑                                                          │
     └──────────────── Resultados ←─────────────────────────────┘
```

---

## 1.2 LINGUAGENS DE PROGRAMAÇÃO

### 1.2.1 WebODM - Stack Principal

| Componente | Linguagem | Versão | Propósito |
|------------|-----------|--------|-----------|
| Backend | Python | 3.9 | Framework Django, APIs, processamento |
| Frontend | JavaScript/ES6 | Node 20 | React UI, visualização |
| Bundler | JavaScript | Webpack 5.89 | Empacotamento de assets |
| Database | SQL | PostgreSQL | Persistência de dados |
| Cache | - | Redis | Sessões, filas Celery |

### 1.2.2 ODM Core - Engine de Processamento

| Componente | Linguagem | Propósito |
|------------|-----------|-----------|
| Pipeline Principal | Python 3.x | Orquestração de stages |
| OpenSfM | Python + C++ | Structure from Motion |
| OpenMVS | C++17 | Dense reconstruction |
| Ceres Solver | C++14 | Bundle adjustment |
| PDAL | C++14 | Point cloud processing |
| OpenCV | C++14 | Computer vision |

---

## 1.3 ESTRUTURA DO REPOSITÓRIO WEBODM

```
WebODM_teste/
├── app/                          # Aplicação Django principal
│   ├── api/                      # REST API endpoints
│   │   ├── tasks.py             # API de tarefas (upload, status)
│   │   ├── projects.py          # API de projetos
│   │   ├── tiler.py             # Tile server para mapas
│   │   └── formulas.py          # Índices vegetativos (NDVI, etc)
│   ├── models/                   # Modelos Django ORM
│   │   ├── task.py              # Modelo Task (1089 linhas)
│   │   ├── project.py           # Modelo Project
│   │   └── plugin.py            # Sistema de plugins
│   ├── classes/                  # Classes auxiliares
│   │   ├── gcp.py               # Parser de Ground Control Points
│   │   └── console.py           # Output de console
│   ├── static/                   # Assets estáticos
│   │   └── app/js/              # Código React
│   ├── plugins/                  # Sistema de extensões
│   └── views/                    # Views Django
├── nodeodm/                      # Integração com NodeODM
│   ├── external/NodeODM/        # Submódulo Git (vazio no fork)
│   └── models.py                # ProcessingNode model
├── worker/                       # Celery workers
│   ├── celery.py                # Configuração Celery
│   └── tasks.py                 # Background tasks
├── webodm/                       # Configurações Django
│   ├── settings.py              # Settings principais
│   └── urls.py                  # URL routing
├── coreplugins/                  # Plugins integrados
│   ├── measure/                 # Medições no mapa
│   ├── contours/                # Geração de contornos
│   └── lightning/               # Processamento em nuvem
├── docs/                         # Documentação
├── nginx/                        # Configuração reverse proxy
├── db/                           # Scripts PostgreSQL
├── Dockerfile                    # Build container
├── docker-compose.yml            # Orquestração de serviços
├── requirements.txt              # Dependências Python
├── package.json                  # Dependências Node.js
└── webpack.config.js             # Configuração bundler
```

---

## 1.4 DEPENDÊNCIAS PYTHON (requirements.txt)

### 1.4.1 Framework Web

```python
Django==2.2.27                    # Framework web principal
djangorestframework==3.13.1       # API REST
djangorestframework-jwt==1.9.0    # Autenticação JWT
django-cors-headers==3.0.2        # CORS para APIs
django-guardian==1.4.9            # Permissões por objeto
django-filter==2.4.0              # Filtros de queryset
drf-nested-routers==0.11.1        # Rotas aninhadas
drf-yasg==1.20.0                  # Documentação Swagger
```

### 1.4.2 Processamento Assíncrono

```python
celery==4.4.0                     # Task queue
redis==3.2.0                      # Message broker
kombu==4.6.7                      # Messaging library
billiard==3.6.3.0                 # Multiprocessing
amqp==2.5.2                       # AMQP protocol
```

### 1.4.3 Processamento Geoespacial

```python
rasterio==1.3.10                  # Raster I/O (GDAL binding)
rio_tiler-2.1.2                   # Tile generation
Shapely==1.8.0                    # Geometrias vetoriais
numpy==1.26.2                     # Arrays numéricos
scipy==1.11.3                     # Computação científica
numexpr                           # Expressões numéricas otimizadas
```

### 1.4.4 Processamento de Imagens

```python
Pillow==8.3.2                     # Manipulação de imagens
piexif==1.1.3                     # EXIF metadata
pilkit==2.0                       # Image processing toolkit
django-imagekit==4.0.1            # Thumbnails Django
```

### 1.4.5 Integração ODM

```python
pyodm==1.5.11                     # Cliente Python para NodeODM API
```

---

## 1.5 DEPENDÊNCIAS NODE.JS (package.json)

### 1.5.1 Core React

```json
{
  "react": "^16.4.0",
  "react-dom": "^16.4.0",
  "react-router": "^4.1.1",
  "react-router-dom": "^4.1.1"
}
```

### 1.5.2 Mapeamento e Visualização

```json
{
  "leaflet": "1.3.1",              // Mapas interativos
  "leaflet-fullscreen": "^1.0.2", // Modo fullscreen
  "proj4": "^2.4.3",              // Projeções cartográficas
  "d3": "^3.5.5",                 // Visualização de dados
  "gl-matrix": "^2.3.2"           // Operações matriciais 3D
}
```

### 1.5.3 Processamento 3D

```json
{
  "@gltf-transform/extensions": "^4.2.1",  // glTF manipulation
  "@gltf-transform/functions": "^4.2.1",   // glTF optimization
  "draco3dgltf": "^1.5.7"                  // Compressão Draco
}
```

### 1.5.4 Build Tools

```json
{
  "webpack": "5.89.0",
  "webpack-bundle-tracker": "0.4.3",
  "@babel/core": "^7.0.0-beta.54",
  "@babel/preset-react": "^7.0.0-beta.54",
  "sass": "^1.22.7",
  "sass-loader": "13.3.2"
}
```

---

## 1.6 CONFIGURAÇÃO DOCKER

### 1.6.1 Dockerfile Principal

```dockerfile
FROM ubuntu:22.04 AS common

# Variáveis de Build
ARG NODE_MAJOR=20
ARG PYTHON_VERSION=3.9
ARG RELEASE_CODENAME=jammy

# Variáveis de Runtime
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=$WORKDIR
ENV PROJ_LIB=/usr/share/proj

# Dependências de Sistema
apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv python3.9-dev \
    libpq-dev build-essential git \
    libproj-dev gdal-bin pdal \
    libgdal-dev nginx certbot \
    postgresql-client gettext tzdata
```

### 1.6.2 Serviços Docker Compose

```yaml
# docker-compose.yml
services:
  webapp:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - db
      - broker
    volumes:
      - ./app/media:/webodm/app/media
    
  db:
    image: opendronemap/webodm_db
    # PostgreSQL com PostGIS
    
  broker:
    image: redis:7.0.10
    # Message broker para Celery
    
  worker:
    # Celery worker para tarefas background
    command: celery -A worker worker
    
  node-odm:
    image: opendronemap/nodeodm
    ports:
      - "3000:3000"
    # Engine de processamento
```

---

## 1.7 ANÁLISE DE PORTABILIDADE PARA ANDROID

### 1.7.1 Componentes por Viabilidade de Porte

| Componente | Dificuldade | Estratégia |
|------------|-------------|------------|
| **Pillow/PIL** | ✅ Fácil | Substituto: Android Bitmap APIs |
| **NumPy** | ✅ Fácil | NDK build ou Chaquopy |
| **OpenCV** | ✅ Médio | OpenCV Android SDK oficial |
| **GDAL** | ⚠️ Difícil | Cross-compile com NDK |
| **OpenSfM** | ⚠️ Difícil | Requer Ceres, OpenCV |
| **OpenMVS** | 🔴 Muito Difícil | CUDA dependente |
| **PDAL** | 🔴 Muito Difícil | Muitas dependências |
| **Django** | ❌ Impossível | Não aplicável mobile |
| **PostgreSQL** | ❌ Impossível | SQLite como substituto |

### 1.7.2 Dependências Nativas Críticas

Para um app Android "Fast Stitching", as seguintes dependências nativas são essenciais:

```
ESSENCIAIS PARA FAST STITCHING:
├── OpenCV 4.5.0 (Feature Detection, Matching)
│   └── Android SDK disponível oficialmente
├── Ceres Solver 2.0.0 (Bundle Adjustment)
│   └── Requer cross-compile NDK
├── Eigen 3.4 (Linear Algebra)
│   └── Header-only, fácil de portar
├── GFlags 2.2.2 (Configuration)
│   └── Cross-compile simples
└── OpenSfM (SfM Pipeline)
    └── Python + C++ híbrido
    └── Pode ser portado como biblioteca C++ pura

OPCIONAIS (para qualidade superior):
├── OpenMVS (Dense Reconstruction)
│   └── Pode ser omitido para "fast" mode
└── PDAL (Point Cloud Processing)
    └── Pode usar alternativas mais leves
```

### 1.7.3 Matriz de Compatibilidade Android NDK

| Biblioteca | ARM64-v8a | armeabi-v7a | x86_64 | Notas |
|------------|-----------|-------------|--------|-------|
| OpenCV | ✅ | ✅ | ✅ | SDK oficial |
| Eigen | ✅ | ✅ | ✅ | Header-only |
| Ceres | ✅ | ⚠️ | ✅ | SSE→NEON |
| GFlags | ✅ | ✅ | ✅ | Simples |
| OpenMVS | ⚠️ | ❌ | ⚠️ | SSE issue |

