# Trabajo de Fin de Grado - Ciencia de Datos e Inteligencia Artificial

## Productor Musical Automático con IA: Transformación de Grabaciones Acústicas en Versiones Producidas con Acompañamiento

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Objetivos](#2-objetivos)
3. [Estado del Arte](#3-estado-del-arte)
4. [Metodología](#4-metodología)
5. [Datasets](#5-datasets)
6. [Arquitectura](#6-arquitectura)
7. [Plan de Trabajo](#7-plan-de-trabajo)
8. [Evaluación](#8-evaluación)

---

## 1. Introducción

### 1.1 Contexto y Motivación

La industria musical ha experimentado una revolución tecnológica significativa en las últimas décadas, especialmente con la integración de la inteligencia artificial y el aprendizaje automático. La producción musical tradicional requiere conocimientos técnicos especializados, equipamiento costoso y un tiempo considerable para transformar una grabación acústica simple en una producción musical completa con múltiples instrumentos y efectos.

### 1.2 Planteamiento del Problema

Actualmente existe una brecha significativa entre la creación musical básica (grabaciones acústicas simples) y la producción musical profesional. Muchos músicos y compositores tienen ideas creativas pero carecen de:
- Conocimientos técnicos en producción musical
- Acceso a instrumentos digitales y software especializado
- Tiempo para aprender técnicas de mezcla y masterización
- Recursos económicos para contratar productores profesionales

### 1.3 Propuesta de Solución

Este trabajo propone el desarrollo de un sistema de inteligencia artificial capaz de automatizar el proceso de producción musical, transformando grabaciones acústicas básicas (voz, guitarra acústica, piano, etc.) en versiones completamente producidas con:
- Acompañamiento instrumental automático
- Arreglos musicales inteligentes
- Efectos de audio apropiados
- Mezcla y masterización automatizada

### 1.4 Alcance del Proyecto

El proyecto se centra en géneros musicales populares (pop, rock, folk) y se limitará inicialmente a:
- Grabaciones mono o estéreo de instrumentos acústicos
- Duración máxima de 5 minutos por canción
- Formatos de audio estándar (WAV, MP3)
- Generación de hasta 4 pistas adicionales de acompañamiento

---

## 2. Objetivos

### 2.1 Objetivo General

Desarrollar un sistema de inteligencia artificial capaz de transformar automáticamente grabaciones acústicas simples en producciones musicales completas con acompañamiento instrumental, manteniendo la coherencia musical y el estilo original de la grabación.

### 2.2 Objetivos Específicos

#### 2.2.1 Objetivos Técnicos
- **OT1**: Implementar un sistema de análisis musical automático que extraiga características armónicas, melódicas y rítmicas de grabaciones acústicas
- **OT2**: Desarrollar algoritmos de separación de fuentes para aislar la melodía principal del audio original
- **OT3**: Crear un generador de acompañamiento basado en redes neuronales que produzca pistas instrumentales coherentes
- **OT4**: Implementar un sistema de mezcla automática que equilibre los niveles y aplique efectos apropiados
- **OT5**: Optimizar el sistema para procesamiento en tiempo real o casi-real

#### 2.2.2 Objetivos de Investigación
- **OI1**: Evaluar diferentes arquitecturas de redes neuronales para la generación de acompañamiento musical
- **OI2**: Analizar la efectividad de técnicas de transfer learning en el dominio musical
- **OI3**: Investigar métodos de evaluación objetiva y subjetiva para producciones musicales automatizadas
- **OI4**: Estudiar la preservación de la intención artística original en el proceso automatizado

#### 2.2.3 Objetivos de Usabilidad
- **OU1**: Diseñar una interfaz de usuario intuitiva para músicos sin conocimientos técnicos avanzados
- **OU2**: Proporcionar controles de personalización para ajustar el estilo y la intensidad del acompañamiento
- **OU3**: Garantizar tiempos de procesamiento aceptables (< 2 minutos para una canción de 3 minutos)

---

## 3. Estado del Arte

### 3.1 Análisis Musical Automatizado

#### 3.1.1 Extracción de Características Musicales
- **Chromagrams y Análisis Armónico**: Técnicas establecidas para la extracción de información tonal
- **Beat Tracking y Análisis Rítmico**: Algoritmos para detección de tempo y estructura métrica
- **Segmentación Musical**: Identificación automática de secciones musicales (verso, estribillo, puente)

#### 3.1.2 Herramientas y Frameworks Existentes
- **Librosa**: Biblioteca de Python para análisis de audio y música
- **Essentia**: Framework de análisis de audio desarrollado por Music Technology Group
- **Madmom**: Biblioteca especializada en análisis de audio musical en tiempo real

### 3.2 Generación Musical con IA

#### 3.2.1 Modelos Generativos Tradicionales
- **Hidden Markov Models (HMM)**: Primeros intentos de modelado musical estadístico
- **N-gramas musicales**: Aproximaciones basadas en secuencias para composición automática

#### 3.2.2 Deep Learning en Generación Musical
- **RNN y LSTM**: 
  - Music RNN (Google): Generación de melodías y acompañamientos
  - PerformanceRNN: Generación de interpretaciones musicales expresivas
  
- **Transformer y Attention Models**:
  - Music Transformer: Aplicación de la arquitectura transformer a la música
  - MuseBERT: Modelos tipo BERT para comprensión musical
  
- **Generative Adversarial Networks (GANs)**:
  - MuseGAN: Generación polifónica multi-pista
  - SeqGAN aplicado a secuencias musicales

- **Variational Autoencoders (VAEs)**:
  - MusicVAE: Interpolación y variación de melodías
  - Aplicaciones en arreglos y armonización

#### 3.2.3 Modelos Especializados en Acompañamiento
- **Coconet (Google)**: Armonización automática de melodías
- **BachBot**: Sistema especializado en contrapunto estilo Bach
- **JazzGAN**: Generación de acompañamientos de jazz

### 3.3 Separación de Fuentes Musicales

#### 3.3.1 Técnicas Clásicas
- **Independent Component Analysis (ICA)**
- **Non-negative Matrix Factorization (NMF)**
- **Spectral Masking Techniques**

#### 3.3.2 Enfoques con Deep Learning
- **U-Net para separación de audio**: Arquitecturas encoder-decoder
- **Wave-U-Net**: Separación directa en el dominio temporal
- **Spleeter (Deezer)**: Sistema de separación de fuentes de código abierto
- **DEMUCS (Facebook)**: Separación end-to-end con convoluciones temporales

### 3.4 Producción Musical Automatizada

#### 3.4.1 Sistemas Comerciales
- **LANDR**: Masterización automática basada en AI
- **eMastered**: Plataforma de masterización online
- **AIVA**: Compositor artificial para música orquestal

#### 3.4.2 Investigación Académica
- **Sistemas de mezcla automática**: Equilibrio de niveles y panoramización
- **Aplicación automática de efectos**: Reverb, compresión, EQ adaptativos
- **Optimización perceptual**: Métricas de calidad de audio

### 3.5 Lagunas en la Investigación Actual

#### 3.5.1 Limitaciones Identificadas
- Falta de sistemas integrados que combinen análisis, generación y producción
- Limitada evaluación de la coherencia estilística en producciones automatizadas
- Escasez de datasets de alta calidad con pistas separadas y metadatos musicales
- Dificultad para preservar la intención artística original

#### 3.5.2 Oportunidades de Mejora
- Integración de múltiples modalidades (audio, MIDI, partitura)
- Aplicación de técnicas de few-shot learning para estilos específicos
- Desarrollo de métricas de evaluación más robustas
- Personalización basada en preferencias del usuario

---

## 4. Metodología

### 4.1 Metodología de Desarrollo

#### 4.1.1 Enfoque de Desarrollo
**Metodología Ágil con Sprints de Investigación**
- Desarrollo iterativo e incremental
- Sprints de 2 semanas con objetivos específicos
- Integración continua de componentes
- Validación constante con usuarios objetivo

#### 4.1.2 Fases del Proyecto

**Fase 1: Investigación y Análisis (4 semanas)**
- Revisión exhaustiva de literatura
- Análisis de datasets disponibles
- Definición de arquitectura preliminar
- Prototipado inicial de componentes clave

**Fase 2: Desarrollo del Core (8 semanas)**
- Implementación del módulo de análisis musical
- Desarrollo del sistema de separación de fuentes
- Creación del generador de acompañamiento
- Integración de componentes básicos

**Fase 3: Optimización y Mejora (4 semanas)**
- Refinamiento de algoritmos
- Optimización de rendimiento
- Implementación de la interfaz de usuario
- Testing exhaustivo

**Fase 4: Evaluación y Documentación (4 semanas)**
- Evaluación objetiva y subjetiva
- Análisis de resultados
- Documentación final
- Preparación de la presentación

### 4.2 Metodología de Investigación

#### 4.2.1 Paradigma de Investigación
**Investigación Mixta (Cuantitativa y Cualitativa)**
- Análisis cuantitativo de métricas técnicas de audio
- Evaluación cualitativa de la calidad musical y artística
- Estudios de caso con músicos reales
- Comparación con sistemas existentes

#### 4.2.2 Diseño Experimental

**Experimentos Controlados**
- Variables independientes: tipo de entrada, género musical, configuración del modelo
- Variables dependientes: calidad de la producción, tiempo de procesamiento, satisfacción del usuario
- Grupos de control: producciones manuales vs automatizadas
- Randomización de muestras musicales para evitar sesgos

### 4.3 Pipeline de Procesamiento

#### 4.3.1 Preprocesamiento de Audio
```
Audio de Entrada → Normalización → Análisis de Características → Separación de Fuentes
```

**Componentes específicos:**
- Normalización de nivel y frecuencia de muestreo
- Extracción de chromagrams, spectrograms, MFCC
- Detección de tempo, tonalidad y estructura musical
- Separación melodía/acompañamiento si existe

#### 4.3.2 Generación de Acompañamiento
```
Características Extraídas → Modelo Generativo → Post-procesamiento → Síntesis de Audio
```

**Proceso detallado:**
- Generación de progresiones armónicas coherentes
- Creación de patrones rítmicos apropiados
- Síntesis de instrumentos virtuales
- Aplicación de humanización y variación

#### 4.3.3 Mezcla y Masterización
```
Pistas Separadas → Balance de Niveles → Aplicación de Efectos → Master Final
```

**Elementos incluidos:**
- Ecualizador automático por pista
- Compresión dinámica inteligente
- Reverb y efectos ambientales
- Limitador y maximizador final

### 4.4 Herramientas y Tecnologías

#### 4.4.1 Frameworks de Machine Learning
- **TensorFlow 2.x**: Framework principal para redes neuronales
- **PyTorch**: Para experimentación rápida de arquitecturas
- **Scikit-learn**: Algoritmos de ML tradicionales y métricas
- **Keras**: API de alto nivel para prototipado rápido

#### 4.4.2 Bibliotecas de Audio
- **Librosa**: Análisis y procesamiento de audio musical
- **Soundfile**: I/O de archivos de audio
- **PyDub**: Manipulación básica de audio
- **Pretty_MIDI**: Manejo de archivos MIDI

#### 4.4.3 Herramientas de Desarrollo
- **Python 3.8+**: Lenguaje principal de desarrollo
- **Jupyter Notebooks**: Experimentación y análisis
- **Git/GitHub**: Control de versiones
- **Docker**: Containerización para reproducibilidad

#### 4.4.4 Infraestructura y Recursos
- **GPU NVIDIA RTX 3080+**: Para entrenamiento de modelos
- **Google Colab Pro**: Recursos computacionales adicionales
- **AWS/GCP**: Almacenamiento de datasets y modelos
- **Weights & Biases**: Tracking de experimentos

---

## 5. Datasets

### 5.1 Datasets Primarios

#### 5.1.1 MUSDB18-HQ
**Descripción**: Dataset de separación de fuentes con pistas multipista de alta calidad
- **Tamaño**: 150 canciones completas (train: 100, test: 50)
- **Formato**: WAV 44.1kHz/16-bit, stems separados
- **Géneros**: Pop, rock, electrónica, folk
- **Uso en el proyecto**: Entrenamiento del módulo de separación de fuentes

#### 5.1.2 Lakh MIDI Dataset (LMD)
**Descripción**: Colección masiva de archivos MIDI para análisis musical
- **Tamaño**: ~170,000 archivos MIDI únicos
- **Metadatos**: Información de artista, álbum, género
- **Uso en el proyecto**: Entrenamiento del generador de acompañamiento

#### 5.1.3 MagnaTagATune
**Descripción**: Dataset con anotaciones de género y estado de ánimo
- **Tamaño**: ~25,000 clips de audio de 30 segundos
- **Anotaciones**: Tags de género, instrumentos, emociones
- **Uso en el proyecto**: Validación y clasificación de estilos

#### 5.1.4 OpenMIC-2018
**Descripción**: Dataset para reconocimiento multi-etiqueta de instrumentos
- **Tamaño**: ~20,000 clips de audio con anotaciones de instrumentos
- **Uso en el proyecto**: Identificación de instrumentos en grabaciones originales

### 5.2 Datasets Secundarios

#### 5.2.1 Free Music Archive (FMA)
**Descripción**: Colección de música libre con metadatos ricos
- **Subconjuntos**: FMA-small (8GB), FMA-medium (25GB)
- **Uso**: Análisis de géneros y validación de diversidad musical

#### 5.2.2 Million Song Dataset (MSD)
**Descripción**: Colección de características de audio de un millón de canciones
- **Características**: Audio features pre-extraídas
- **Uso**: Análisis estadístico y benchmarking

#### 5.2.3 Groove MIDI Dataset
**Descripción**: Dataset de patrones de batería MIDI con audio sincronizado
- **Uso**: Generación de patrones rítmicos realistas

### 5.3 Creación de Datasets Propios

#### 5.3.1 Dataset de Grabaciones Acústicas
**Objetivo**: Crear un conjunto de grabaciones simples para testing
- **Contenido**: 50 grabaciones acústicas originales
- **Variedad**: Guitarra, piano, voz, combinaciones simples
- **Calidad**: Grabaciones amateur típicas de usuarios objetivo

#### 5.3.2 Dataset de Validación Humana
**Objetivo**: Evaluaciones subjetivas de calidad musical
- **Estructura**: Pares (original, producción automatizada)
- **Evaluadores**: 20 músicos y productores
- **Métricas**: Coherencia, calidad, preferencia

### 5.4 Preprocesamiento y Aumento de Datos

#### 5.4.1 Técnicas de Data Augmentation
- **Pitch shifting**: ±2 semitonos para variación tonal
- **Time stretching**: ±10% para variación temporal
- **Noise injection**: Simulación de condiciones reales de grabación
- **Equalización**: Simulación de diferentes equipos de grabación

#### 5.4.2 Normalización y Estandarización
- **Normalización de nivel**: Peak normalization a -3dB
- **Frecuencia de muestreo**: Resampling uniforme a 44.1kHz
- **Duración**: Segmentación en clips de longitud fija
- **Formato**: Conversión uniforme a mono/estéreo según necesidad

### 5.5 Gestión y Almacenamiento

#### 5.5.1 Infraestructura de Datos
- **Almacenamiento local**: SSD de 2TB para desarrollo
- **Cloud storage**: AWS S3 para backup y colaboración
- **Versionado**: DVC (Data Version Control) para tracking de datasets

#### 5.5.2 Consideraciones Éticas y Legales
- **Licencias**: Verificación de permisos de uso para todos los datasets
- **Privacidad**: Anonimización de grabaciones propias
- **Derechos de autor**: Uso exclusivo de contenido libre o con licencia apropiada

---

## 6. Arquitectura

### 6.1 Arquitectura General del Sistema

#### 6.1.1 Diseño Modular
```
[Audio Input] → [Análisis Musical] → [Separación de Fuentes] → [Generación de Acompañamiento] → [Mezcla Automática] → [Audio Output]
```

**Principios de Diseño:**
- **Modularidad**: Componentes independientes e intercambiables
- **Escalabilidad**: Capacidad de procesamiento paralelo
- **Extensibilidad**: Fácil adición de nuevos módulos
- **Reproducibilidad**: Configuración determinística

#### 6.1.2 Flujo de Datos Principal
```
Audio Input → Audio Preprocessor → Feature Extractor → Music Analyzer → Source Separator → Accompaniment Generator → Audio Mixer → Post-processor → Audio Output
```

### 6.2 Módulo de Análisis Musical

#### 6.2.1 Extractor de Características
**Arquitectura**: CNN + LSTM híbrida

```python
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)
```

**Características Extraídas:**
- Chromagram (12 dimensiones)
- Spectral centroid, rolloff, zero-crossing rate
- MFCCs (13 coeficientes)
- Tempo y beat tracking
- Tonalidad y modo musical

#### 6.2.2 Analizador de Estructura Musical
**Metodología**: Segmentación basada en similaridad y detección de patrones repetitivos

**Componentes:**
- Detección de secciones (intro, verse, chorus, bridge, outro)
- Identificación de progresiones armónicas
- Análisis de forma musical (AABA, verse-chorus, etc.)

### 6.3 Módulo de Separación de Fuentes

#### 6.3.1 Arquitectura U-Net Modificada
**Base**: U-Net con skip connections adaptada para audio

```python
class AudioUNet(nn.Module):
    def __init__(self, num_sources=2):
        super().__init__()
        self.encoder = EncoderBlock()
        self.decoder = DecoderBlock()
        self.skip_connections = SkipConnections()
        self.output_layer = nn.Conv2d(64, num_sources, 1)
```

**Especificaciones:**
- **Input**: Spectrogram complejo (magnitud + fase)
- **Output**: Máscaras para separación de fuentes
- **Arquitectura**: 5 niveles de encoder/decoder
- **Loss Function**: L1 Loss + Perceptual Loss

#### 6.3.2 Técnica de Separación
- **Dominio**: Tiempo-frecuencia (STFT)
- **Ventana**: Hann window, 2048 samples, 75% overlap
- **Método**: Soft masking con suavizado temporal
- **Post-procesamiento**: Filtrado de artefactos y reconstrucción de fase

### 6.4 Módulo de Generación de Acompañamiento

#### 6.4.1 Arquitectura Transformer Musical
**Base**: Transformer modificado para secuencias musicales

```python
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.output_layer = nn.Linear(d_model, vocab_size)
```

**Características Específicas:**
- **Vocabulario**: 128 notas MIDI + metadatos (velocidad, duración)
- **Contexto**: Ventana deslizante de 512 tokens
- **Conditioning**: Información harmónica y rítmica de la melodía original
- **Multi-track**: Generación simultánea de hasta 4 pistas de acompañamiento

#### 6.4.2 Sistema de Conditioning
**Método**: Cross-attention con características musicales

**Inputs de Conditioning:**
- Progresión armónica detectada
- Patrón rítmico principal
- Género musical inferido
- Estructura de la canción
- Velocidad (tempo) y compás

#### 6.4.3 Generación por Instrumentos
**Instrumentos Target:**
- **Bass**: Líneas de bajo siguiendo la progresión armónica
- **Drums**: Patrones rítmicos coherentes con el estilo
- **Piano/Keyboard**: Acompañamiento armónico
- **Strings/Pads**: Texturas ambientales

### 6.5 Módulo de Mezcla Automática

#### 6.5.1 Sistema de Balance Automático
**Algoritmo**: Análisis espectral + reglas perceptuales

**Componentes:**
- Detección automática de conflictos frecuenciales
- Balance de niveles basado en loudness estándar
- Panoramización inteligente por instrumento
- Compresión multibanda adaptativa

#### 6.5.2 Aplicación de Efectos
**Framework**: Chain de efectos parametrizables

```python
class EffectChain:
    def __init__(self):
        self.effects = [
            EQEffect(),
            CompressorEffect(),
            ReverbEffect(),
            LimiterEffect()
        ]
    
    def apply(self, audio, track_type):
        params = self.get_optimal_params(track_type)
        return self.process_chain(audio, params)
```

**Efectos Implementados:**
- **EQ**: Equalización adaptativa por instrumento
- **Compresión**: Dinámica controlada automáticamente
- **Reverb**: Espacialización coherente
- **Stereo Imaging**: Amplitud estéreo inteligente

### 6.6 Arquitectura de Software

#### 6.6.1 Patrón de Diseño
**Patrón**: Pipeline + Observer + Factory

```python
class AudioProcessingPipeline:
    def __init__(self):
        self.stages = []
        self.observers = []
    
    def add_stage(self, stage):
        self.stages.append(stage)
    
    def process(self, audio_input):
        result = audio_input
        for stage in self.stages:
            result = stage.process(result)
            self.notify_observers(stage, result)
        return result
```

#### 6.6.2 Gestión de Configuración
**Sistema**: YAML-based configuration con validación

**Estructura de Configuración:**
```yaml
model_config:
  feature_extractor:
    sample_rate: 44100
    n_fft: 2048
    hop_length: 512
  
  generator:
    model_type: "transformer"
    hidden_size: 512
    num_layers: 6
    num_heads: 8
  
  mixing:
    enable_auto_eq: true
    enable_compression: true
    target_lufs: -16.0
```

### 6.7 Infraestructura y Deployment

#### 6.7.1 Containerización
**Docker**: Entorno reproducible con dependencias fijas

```dockerfile
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

#### 6.7.2 API REST
**Framework**: FastAPI con endpoints asíncronos

```python
@app.post("/process-audio")
async def process_audio(file: UploadFile):
    # Procesamiento asíncrono del audio
    result = await audio_processor.process(file)
    return {"status": "success", "output_url": result}
```

**Endpoints Principales:**
- `POST /process-audio`: Procesamiento principal
- `GET /status/{job_id}`: Estado del procesamiento
- `GET /download/{job_id}`: Descarga de resultado
- `POST /configure`: Actualización de configuración

---

## 7. Plan de Trabajo

### 7.1 Cronograma General

**Duración Total**: 20 semanas (5 meses)
**Modalidad**: Tiempo completo con dedicación de 40 horas/semana

#### 7.1.1 Vista General por Fases
```
Fase 1: Investigación y Análisis (4 semanas)
├── Semana 1-2: Revisión de Literatura
├── Semana 3: Análisis de Datasets  
└── Semana 4: Prototipo Inicial

Fase 2: Desarrollo del Core (8 semanas)
├── Semana 5-6: Módulo de Análisis Musical
├── Semana 7-8: Separación de Fuentes
├── Semana 9-11: Generador de Acompañamiento
└── Semana 12: Sistema de Mezcla

Fase 3: Optimización y Mejora (4 semanas)
├── Semana 13-14: Refinamiento de Algoritmos
├── Semana 15: Interfaz de Usuario
└── Semana 16: Testing del Sistema

Fase 4: Evaluación y Documentación (4 semanas)
├── Semana 17-18: Experimentos de Evaluación
├── Semana 19: Análisis de Resultados
└── Semana 20: Documentación Final
```

### 7.2 Fase 1: Investigación y Análisis (4 semanas)

#### 7.2.1 Semana 1-2: Revisión de Literatura
**Objetivos:**
- Análisis exhaustivo del estado del arte
- Identificación de técnicas más prometedoras
- Definición de gaps en la investigación actual

**Entregables:**
- Documento de revisión bibliográfica (20 páginas)
- Base de datos de papers relevantes (Zotero)
- Análisis comparativo de enfoques existentes

**Actividades Específicas:**
- Búsqueda sistemática en IEEE Xplore, ACM Digital Library, arXiv
- Análisis de 50+ papers relacionados con generación musical y AI
- Contacto con autores de trabajos relevantes
- Identificación de datasets y herramientas disponibles

#### 7.2.2 Semana 3: Análisis de Datasets
**Objetivos:**
- Evaluación de calidad y utilidad de datasets disponibles
- Planificación de estrategia de datos
- Configuración de infraestructura de datos

**Entregables:**
- Informe de evaluación de datasets
- Pipeline de preprocesamiento inicial
- Configuración de almacenamiento de datos

**Actividades Específicas:**
- Descarga y exploración de MUSDB18, LMD, MagnaTagATune
- Análisis estadístico de distribuciones de géneros y características
- Implementación de scripts de preprocesamiento
- Setup de infraestructura en AWS/GCP

#### 7.2.3 Semana 4: Prototipo Inicial
**Objetivos:**
- Validación de conceptos clave
- Implementación de pipeline básico
- Identificación de desafíos técnicos

**Entregables:**
- Prototipo funcional básico
- Resultados preliminares de pruebas
- Plan técnico detallado

**Actividades Específicas:**
- Implementación de extractor de características básico
- Desarrollo de generador simple basado en reglas
- Testing inicial con grabaciones de prueba
- Documentación de arquitectura preliminar

### 7.3 Fase 2: Desarrollo del Core (8 semanas)

#### 7.3.1 Semanas 5-6: Módulo de Análisis Musical
**Objetivos:**
- Implementación completa del analizador musical
- Optimización de extracción de características
- Validación de precisión en detección de elementos musicales

**Entregables:**
- Módulo de análisis musical completo
- Suite de tests unitarios
- Benchmarks de rendimiento

**Tareas Detalladas:**
- Implementación de extractor de chromagrams optimizado
- Desarrollo de detector de tempo robusto
- Creación de analizador de estructura musical
- Integración con pipeline principal

**Métricas de Éxito:**
- Precisión > 90% en detección de tonalidad
- Error < 2 BPM en detección de tempo
- Tiempo de procesamiento < 10 segundos/minuto de audio

#### 7.3.2 Semanas 7-8: Separación de Fuentes
**Objetivos:**
- Implementación de sistema de separación estado-del-arte
- Entrenamiento de modelos especializados
- Optimización para casos de uso específicos

**Entregables:**
- Modelo de separación entrenado
- Scripts de entrenamiento y evaluación
- Análisis de rendimiento comparativo

**Tareas Detalladas:**
- Implementación de arquitectura U-Net para audio
- Entrenamiento con MUSDB18-HQ dataset
- Fine-tuning para instrumentos acústicos específicos
- Implementación de post-procesamiento de máscaras

**Métricas de Éxito:**
- SDR > 8 dB en separación melodía/acompañamiento
- Tiempo de inferencia < 0.1x tiempo real
- Calidad perceptual evaluada por panel de expertos

#### 7.3.3 Semanas 9-11: Generador de Acompañamiento
**Objetivos:**
- Desarrollo del componente central del sistema
- Entrenamiento de modelo transformer musical
- Implementación de sistema de conditioning

**Entregables:**
- Modelo generativo entrenado
- Sistema de conditioning implementado
- Generador multi-instrumento funcional

**Tareas Detalladas:**
- Implementación de arquitectura Transformer adaptada
- Preprocesamiento de MIDI datasets para entrenamiento
- Desarrollo de sistema de tokenización musical
- Implementación de conditioning con características extraídas

**Hitos Intermedios:**
- Semana 9: Arquitectura base implementada
- Semana 10: Modelo entrenado para un instrumento
- Semana 11: Sistema multi-instrumento completo

#### 7.3.4 Semana 12: Sistema de Mezcla
**Objetivos:**
- Implementación de mezcla automática
- Desarrollo de algoritmos de balance
- Integración de efectos de audio

**Entregables:**
- Módulo de mezcla automática
- Biblioteca de efectos de audio
- Sistema de masterización básico

**Tareas Detalladas:**
- Implementación de algoritmos de balance automático
- Desarrollo de sistema de EQ adaptativo
- Integración de efectos de reverb y compresión
- Calibración con referencias comerciales

### 7.4 Fase 3: Optimización y Mejora (4 semanas)

#### 7.4.1 Semanas 13-14: Refinamiento de Algoritmos
**Objetivos:**
- Optimización de rendimiento
- Mejora de calidad de salida
- Resolución de problemas identificados

**Actividades:**
- Profiling y optimización de código
- Fine-tuning de hiperparámetros
- Implementación de técnicas de aceleración
- Validación con casos de prueba complejos

#### 7.4.2 Semana 15: Interfaz de Usuario
**Objetivos:**
- Desarrollo de interfaz intuitiva
- Implementación de controles de personalización
- Testing de usabilidad

**Entregables:**
- Aplicación web funcional
- API REST documentada
- Guía de usuario

#### 7.4.3 Semana 16: Testing del Sistema
**Objetivos:**
- Testing integral del sistema
- Validación de casos extremos
- Optimización de estabilidad

**Actividades:**
- Testing automatizado completo
- Pruebas de stress y rendimiento
- Validación con usuarios beta
- Documentación de bugs y fixes

### 7.5 Fase 4: Evaluación y Documentación (4 semanas)

#### 7.5.1 Semanas 17-18: Experimentos de Evaluación
**Objetivos:**
- Evaluación objetiva comprehensiva
- Estudios de usuario controlados
- Benchmarking con sistemas existentes

**Diseño Experimental:**
- 50 grabaciones de prueba variadas
- Panel de 20 evaluadores expertos
- Métricas objetivas y subjetivas
- Comparación con 3 sistemas baseline

#### 7.5.2 Semana 19: Análisis de Resultados
**Objetivos:**
- Análisis estadístico de resultados
- Identificación de fortalezas y limitaciones
- Formulación de conclusiones

**Entregables:**
- Informe de resultados experimentales
- Análisis estadístico completo
- Recomendaciones para trabajo futuro

#### 7.5.3 Semana 20: Documentación Final
**Objetivos:**
- Finalización de memoria de TFG
- Preparación de presentación
- Entrega de código y datasets

**Entregables:**
- Memoria de TFG completa (80-100 páginas)
- Presentación para defensa (30 slides)
- Repositorio de código documentado
- Datasets y modelos entrenados

### 7.6 Gestión de Riesgos

#### 7.6.1 Riesgos Técnicos
**Riesgo**: Dificultad en converger el entrenamiento de modelos generativos
- **Probabilidad**: Media
- **Impacto**: Alto
- **Mitigación**: Implementar múltiples arquitecturas alternativas, buscar modelos pre-entrenados

**Riesgo**: Calidad insuficiente en separación de fuentes
- **Probabilidad**: Media
- **Impacto**: Medio
- **Mitigación**: Usar técnicas de ensemble, implementar post-procesamiento avanzado

#### 7.6.2 Riesgos de Recursos
**Riesgo**: Insuficiente capacidad computacional para entrenamiento
- **Probabilidad**: Baja
- **Impacto**: Medio
- **Mitigación**: Acceso a recursos cloud, colaboración con laboratorios universitarios

**Riesgo**: Problemas de acceso a datasets
- **Probabilidad**: Baja
- **Impacto**: Medio
- **Mitigación**: Múltiples fuentes de datos, creación de datasets propios

#### 7.6.3 Plan de Contingencia
**Escenario**: Retraso significativo en desarrollo core
- **Acción**: Priorizar componentes esenciales, simplificar arquitectura
- **Timeline alternativo**: Extensión de 2 semanas con reducción de scope

### 7.7 Hitos y Entregables Clave

#### 7.7.1 Hitos Principales
| Hito | Fecha | Criterio de Éxito |
|------|--------|------------------|
| H1: Prototipo funcional | Semana 4 | Pipeline básico operativo |
| H2: Módulo análisis completo | Semana 6 | Precisión > 90% en métricas clave |
| H3: Separación implementada | Semana 8 | SDR > 8 dB en dataset test |
| H4: Generador operativo | Semana 11 | Genera acompañamiento coherente |
| H5: Sistema integrado | Semana 12 | Pipeline end-to-end funcional |
| H6: Sistema optimizado | Semana 16 | Rendimiento acceptable para usuarios |
| H7: Evaluación completa | Semana 18 | Resultados estadísticamente significativos |

#### 7.7.2 Entregables por Fase
**Fase 1**: Análisis y diseño (30% del trabajo)
**Fase 2**: Implementación core (50% del trabajo)
**Fase 3**: Refinamiento (15% del trabajo)
**Fase 4**: Evaluación y documentación (5% del trabajo)

---

## 8. Evaluación

### 8.1 Marco de Evaluación

#### 8.1.1 Enfoque Multi-dimensional
La evaluación del sistema se realizará desde múltiples perspectivas complementarias:

**Evaluación Técnica (40%)**
- Métricas objetivas de calidad de audio
- Rendimiento computacional
- Robustez y estabilidad del sistema

**Evaluación Musical (35%)**
- Coherencia musical y armónica
- Calidad artística del acompañamiento
- Preservación del estilo original

**Evaluación de Usuario (25%)**
- Usabilidad de la interfaz
- Satisfacción de usuarios finales
- Utilidad práctica del sistema

#### 8.1.2 Metodología de Evaluación Mixta
**Cuantitativa**: Métricas objetivas y análisis estadístico
**Cualitativa**: Evaluaciones subjetivas por expertos y usuarios
**Comparativa**: Benchmarking con sistemas existentes

### 8.2 Métricas Objetivas

#### 8.2.1 Métricas de Calidad de Audio

**Signal-to-Distortion Ratio (SDR)**
- Evaluación de separación de fuentes
- Target: SDR > 8 dB para melodía principal
- Comparación con baseline Spleeter

**Espectral Angular Distance (SAD)**
- Medición de similitud espectral
- Aplicación a acompañamientos generados
- Comparación con referencias humanas

**Perceptual Evaluation of Audio Quality (PEAQ)**
- Estándar ITU-R BS.1387-1
- Evaluación objetiva de calidad perceptual
- Escala: -4 (muy molesto) a 0 (imperceptible)

**LUFS (Loudness Units relative to Full Scale)**
- Medición de loudness estándar
- Target: -16 LUFS ±2 para compatibilidad streaming
- Evaluación de consistencia en masterización

#### 8.2.2 Métricas Musicales Específicas

**Harmonic Coherence Score**
```python
def harmonic_coherence_score(original_chroma, generated_chroma):
    """
    Evalúa coherencia armónica entre original y acompañamiento
    """
    correlation = np.corrcoef(original_chroma.flatten(), 
                             generated_chroma.flatten())[0,1]
    return max(0, correlation)
```

**Rhythmic Alignment Score**
- Medición de coherencia rítmica
- Análisis de cross-correlation entre patrones de beat
- Penalización por desviaciones de tempo

**Genre Consistency Score**
- Clasificador de género entrenado independientemente
- Evaluación de preservación de género original
- Métrica: Probabilidad del género correcto

#### 8.2.3 Métricas de Rendimiento

**Latencia de Procesamiento**
- Tiempo real de procesamiento vs duración de audio
- Target: < 2x tiempo real para audio de 3 minutos
- Medición en diferentes configuraciones de hardware

**Uso de Recursos**
- Consumo de memoria RAM durante procesamiento
- Utilización de GPU (si disponible)
- Throughput en procesamiento batch

**Estabilidad del Sistema**
- Tasa de éxito en procesamiento (> 95%)
- Handling de casos edge y archivos corruptos
- Consistencia en resultados múltiples

### 8.3 Evaluación Subjetiva

#### 8.3.1 Panel de Expertos Musicales

**Composición del Panel**
- 10 productores musicales profesionales
- 5 músicos con experiencia en grabación
- 5 ingenieros de audio certificados

**Criterios de Evaluación (Escala 1-7)**

**Calidad Musical (40%)**
- Coherencia armónica con el original
- Naturalidad del acompañamiento
- Creatividad y musicalidad

**Calidad Técnica (35%)**
- Claridad y separación de instrumentos
- Balance de mezcla
- Ausencia de artefactos audibles

**Utilidad Práctica (25%)**
- Valor como herramienta de producción
- Adecuación para diferentes géneros
- Potencial de uso profesional

#### 8.3.2 Estudio de Usuario con Músicos

**Participantes**
- 30 músicos amateur y semi-profesionales
- Diversidad de estilos e instrumentos
- Experiencia variada en producción musical

**Protocolo de Evaluación**
1. **Fase de familiarización** (10 min): Tutorial del sistema
2. **Fase de uso libre** (30 min): Procesamiento de propias grabaciones
3. **Evaluación estructurada** (20 min): Cuestionario y entrevista

**Métricas de Usabilidad**
- System Usability Scale (SUS) > 70
- Task completion rate > 90%
- Time to first successful result < 5 minutos

#### 8.3.3 Test A/B con Audiencia General

**Diseño del Experimento**
- 100 participantes de población general
- Listening test ciego con pares (original vs producido)
- Randomización de orden de presentación

**Variables Medidas**
- Preferencia general (escala 1-7)
- Percepción de calidad (escala 1-7)
- Intención de uso (sí/no)

### 8.4 Benchmarking Comparativo

#### 8.4.1 Sistemas de Referencia

**LANDR (Masterización Automática)**
- Comparación en calidad de masterización
- Evaluación de balance tonal
- Análisis de loudness y dinámica

**Humanos de Control**
- 3 productores creando acompañamientos manuales
- Mismo conjunto de grabaciones de prueba
- Tiempo limitado: 2 horas por canción

**Sistema Baseline Propio**
- Versión simplificada sin AI (reglas heurísticas)
- Acompañamiento basado en progresiones estándar
- Control de complejidad del problema

#### 8.4.2 Dataset de Evaluación

**MusicEval Dataset (Creación Propia)**
- 50 grabaciones acústicas originales
- Diversidad de géneros: Pop, Rock, Folk, Blues, Jazz
- Variedad de instrumentos: Guitarra, piano, voz, combinaciones
- Metadatos: BPM, tonalidad, género, duración

**Criterios de Selección**
- Calidad de grabación amateur típica
- Estructura musical clara
- Diversidad demográfica de intérpretes
- Licencias abiertas para uso en investigación

### 8.5 Análisis Estadístico

#### 8.5.1 Diseño Experimental

**Variables Independientes**
- Tipo de entrada (instrumento, género, calidad)
- Configuración del sistema (modo conservativo/creativo)
- Experiencia del evaluador

**Variables Dependientes**
- Scores de calidad musical
- Métricas técnicas objetivas
- Preferencias de usuario

**Control de Variables Confusas**
- Randomización del orden de evaluación
- Balanceo de géneros y estilos
- Cegamiento de evaluadores cuando posible

#### 8.5.2 Análisis Estadístico Planificado

**Análisis Descriptivo**
- Medias, medianas y distribuciones de scores
- Análisis de correlación entre métricas
- Identificación de outliers y casos extremos

**Análisis Inferencial**
- ANOVA para diferencias entre grupos
- Tests t para comparaciones pareadas
- Regresión múltiple para factores predictivos

**Tamaño de Efecto**
- Cohen's d para diferencias entre sistemas
- R² para varianza explicada
- Intervalos de confianza al 95%

### 8.6 Validación y Reproducibilidad

#### 8.6.1 Validación Cruzada

**K-Fold Cross-Validation (k=5)**
- División del dataset en 5 folds
- Entrenamiento en 4 folds, evaluación en 1
- Promedio de métricas across folds

**Leave-One-Genre-Out**
- Entrenamiento excluyendo un género
- Evaluación de generalización cross-genre
- Identificación de sesgos de estilo

#### 8.6.2 Reproducibilidad

**Semillas Aleatorias Fijas**
- Seeds determinísticas para todos los experimentos
- Documentación completa de configuraciones
- Versionado de datasets y modelos

**Entorno Containerizado**
- Docker image con dependencias exactas
- Scripts automatizados de evaluación
- Resultados completamente reproducibles

### 8.7 Interpretación y Limitaciones

#### 8.7.1 Criterios de Éxito

**Umbral Mínimo de Aceptabilidad**
- Score musical promedio > 4.0/7
- Métricas técnicas > baseline en 80% de casos
- SUS score > 70 en evaluación de usabilidad

**Target de Excelencia**
- Score musical promedio > 5.5/7
- Preferencia sobre baseline en > 70% de casos
- Tiempo de procesamiento < 1.5x tiempo real

#### 8.7.2 Limitaciones Reconocidas

**Limitaciones del Alcance**
- Géneros limitados a pop/rock/folk mainstream
- Duración máxima de 5 minutos por canción
- Calidad dependiente de la grabación original

**Limitaciones Metodológicas**
- Subjetividad inherente en evaluación musical
- Sesgo potencial hacia estilos occidentales
- Limitaciones del panel de evaluadores

**Limitaciones Técnicas**
- Dependencia de datasets de entrenamiento
- Recursos computacionales requeridos
- Posibles artefactos en casos extremos

#### 8.7.3 Plan de Mitigación de Limitaciones

**Diversificación de Evaluadores**
- Inclusión de múltiples perspectivas culturales
- Balanceo de experiencia técnica/musical
- Evaluación en diferentes contextos de uso

**Análisis de Sensibilidad**
- Evaluación con diferentes configuraciones
- Análisis de robustez ante variaciones
- Identificación de casos de fallo típicos

### 8.8 Cronograma de Evaluación

#### 8.8.1 Timeline de Evaluación

**Semana 17: Preparación**
- Finalización de dataset de evaluación
- Reclutamiento de panel de expertos
- Setup de infraestructura de testing

**Semana 18: Ejecución**
- Evaluación objetiva automatizada
- Sesiones con panel de expertos
- Estudios de usuario con músicos

**Semana 19: Análisis**
- Análisis estadístico de resultados
- Interpretación y síntesis de findings
- Preparación de reporte de evaluación

#### 8.8.2 Entregables de Evaluación

**Reporte Técnico de Evaluación**
- Metodología completa utilizada
- Resultados detallados por métrica
- Análisis comparativo con baselines

**Dataset de Benchmarking**
- MusicEval dataset documentado
- Scripts de evaluación reproducibles
- Baseline scores para comparación futura

**Conclusiones y Recomendaciones**
- Interpretación de resultados
- Identificación de fortalezas/debilidades
- Direcciones para trabajo futuro

---

## Conclusiones y Trabajo Futuro

### 9.1 Contribuciones Esperadas

Este Trabajo de Fin de Grado tiene como objetivo realizar contribuciones significativas en múltiples áreas:

**Contribuciones Técnicas:**
- Sistema integral de producción musical automatizada
- Arquitectura novedosa que combina análisis, separación, generación y mezcla
- Metodologías de evaluación específicas para sistemas de producción musical

**Contribuciones Científicas:**
- Análisis comparativo de técnicas de generación musical
- Evaluación de la preservación de intención artística en procesos automatizados
- Dataset de referencia para benchmarking de sistemas similares

**Contribuciones Prácticas:**
- Herramienta accesible para músicos sin conocimientos técnicos avanzados
- Democratización de técnicas de producción musical
- Reducción de barreras de entrada en la creación musical

### 9.2 Impacto Esperado

**En la Investigación:**
- Avance en la comprensión de sistemas de IA musical integral
- Establecimiento de métricas y metodologías de evaluación estándar
- Base para futuros trabajos en producción musical automatizada

**En la Industria:**
- Potencial para nuevas herramientas comerciales
- Inspiración para mejoras en sistemas existentes
- Validación de enfoques de IA en aplicaciones creativas

**En la Educación:**
- Recurso didáctico para aprendizaje de producción musical
- Plataforma para experimentación musical
- Herramienta de apoyo en educación musical

### 9.3 Extensiones Futuras

**Mejoras Técnicas:**
- Expansión a más géneros musicales
- Soporte para formatos de audio avanzados
- Integración con DAWs profesionales

**Funcionalidades Adicionales:**
- Generación de letras automática
- Vocalización sintética
- Adaptación a preferencias de usuario

**Aplicaciones Especializadas:**
- Versión para música clásica y orquestal
- Adaptación para música étnica y mundial
- Sistema especializado en géneros electrónicos

---

*Este esquema representa una base sólida para el desarrollo de un Trabajo de Fin de Grado ambicioso y técnicamente desafiante que combina aspectos fundamentales de la inteligencia artificial, el procesamiento de audio y la creatividad musical.*