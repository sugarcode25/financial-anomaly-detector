# Sistema de Detección de Anomalías Basado en SOM para Mercados de Renta Variable

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Cómputo_Científico-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Análisis_de_Datos-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualización-11557c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![MiniSom](https://img.shields.io/badge/MiniSom-Motor_SOM-red)
![yfinance](https://img.shields.io/badge/yfinance-Datos_de_Mercado-green)
![Licencia](https://img.shields.io/badge/Licencia-Propietaria-critical)

> **Sistema de detección de anomalías no supervisado para mercados financieros mediante Mapas Auto-Organizados (SOM), validado en 10 tickers cross-asset que abarcan 7 perfiles de activo distintos.**

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Motivación y Contexto](#motivación-y-contexto)
3. [Arquitectura](#arquitectura)
4. [Ingeniería de Features](#ingeniería-de-features)
5. [Metodología de Detección](#metodología-de-detección)
6. [Marco de Validación](#marco-de-validación)
7. [Resultados Cross-Asset](#resultados-cross-asset)
8. [Taxonomía de Señales y Accionabilidad](#taxonomía-de-señales-y-accionabilidad)
9. [Análisis de Clustering de Anomalías](#análisis-de-clustering-de-anomalías)
10. [Hallazgos Clave](#hallazgos-clave)
11. [Limitaciones Conocidas y Transparencia](#limitaciones-conocidas-y-transparencia)
12. [Stack Tecnológico y Dependencias](#stack-tecnológico-y-dependencias)
13. [Aviso Legal](#aviso-legal)
14. [Autor](#autor)

---

## Descripción General

Este proyecto implementa un **sistema de detección de anomalías no supervisado** para mercados de renta variable mediante **Mapas Auto-Organizados (Self-Organizing Maps, SOM)**. Originalmente diseñado para un entorno simulado de derivados, el sistema fue adaptado iterativamente y validado contra datos reales de mercado en múltiples clases de activos.

El modelo identifica días de mercado estadísticamente anómalos — momentos en los que la combinación de acción del precio, comportamiento del volumen, régimen de volatilidad y divergencias de momentum se desvía significativamente de los patrones históricos aprendidos. Estas anomalías se clasifican por tipo (pisos direccionales, techos o cambios de régimen neutrales), se someten a backtesting de retornos futuros con pruebas de significancia, y se segmentan por régimen de mercado.

**Escala del proyecto:** ~1,900 líneas de Python distribuidas en múltiples bloques modulares, desarrolladas a lo largo de varios ciclos de iteración mayor que incluyen rondas de testing cross-asset sobre 10 tickers.

---

## Motivación y Contexto

La detección de anomalías en mercados de renta variable enfrenta un desafío fundamental: los mercados son no estacionarios, dependientes del régimen, y están fuertemente influenciados tanto por eventos macro-sistémicos como por dinámicas específicas de cada activo. Los modelos paramétricos tradicionales (z-scores, reglas de umbrales fijos) tienen dificultades con esto porque asumen distribuciones estáticas.

Los Mapas Auto-Organizados ofrecen una alternativa: aprenden la **estructura topológica** de datos financieros de alta dimensionalidad sin supuestos paramétricos, mapeando estados de mercado multivariados en una representación 2D comprimida. Cuando un nuevo punto de datos se mapea pobremente sobre la topología aprendida (alto error de cuantización), señala un estado de mercado genuinamente inusual — sin requerir que el analista predefina qué significa "inusual".

Este proyecto fue construido para responder una pregunta práctica: **¿Puede un sistema SOM no supervisado, originalmente prototipado con datos sintéticos de opciones, adaptarse en un detector de anomalías robusto y cross-asset para renta variable real?**

Tras múltiples versiones de desarrollo iterativo, la respuesta es un sí condicionado — con límites claramente documentados sobre dónde funciona el modelo, dónde falla, y por qué.

---

## Arquitectura

El sistema emplea una **arquitectura multicapa** que combina enfoques de detección complementarios:

```
┌──────────────────────────────────────────────────────────────┐
│                   PIPELINE DE DATOS                          │
│  Datos de Mercado → Ingeniería de Features → Normalización   │
│                 (ventana expansiva, sin look-ahead)           │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │  Múltiples Métodos de  │
               │  Detección Independien-│
               │  tes (topológicos +    │
               │  estadísticos)         │
               └───────────┬────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │  Votación por    │
                 │  Consenso        │
                 │  (acuerdo multi- │
                 │  método)         │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │  Clasificación   │
                 │  & Backtesting   │
                 └──────────────────┘
```

**Principios de diseño clave:**

- **Doble perspectiva temporal:** La arquitectura utiliza tanto una topología global entrenada sobre el historial completo como un componente adaptativo que se re-entrena periódicamente sobre datos recientes. Esto captura patrones estructurales de largo plazo y cambios de régimen de corto plazo simultáneamente.
- **Mecanismo de consenso:** Una anomalía se señaliza solo cuando múltiples métodos de detección independientes coinciden, lo que reduce significativamente los falsos positivos. El número específico de métodos, su configuración y el umbral de votación son propietarios.
- **Normalización expansiva sin look-ahead:** Las features se normalizan mediante una ventana expansiva — el escalador solo ve datos disponibles hasta cada punto en el tiempo, asegurando que ninguna información futura se filtre hacia observaciones pasadas.
- **Estabilización mediante warmup:** Datos históricos previos a la ventana de análisis estabilizan el cómputo de features y la normalización desde el primer día de la ventana de detección. El mecanismo mediante el cual los datos de warmup informan la normalización — sin contaminar la distribución de la ventana de detección — fue el componente de diseño más iterado del proyecto, requiriendo múltiples enfoques antes de llegar a la solución final.
- **Retrocompatibilidad:** Activos sin historial previo suficiente (por ejemplo, tickers post-IPO) caen automáticamente en una ruta de normalización simplificada, manteniendo funcionalidad con advertencias documentadas.

> ⚠️ **Nota:** Las dimensiones de grilla, hiperparámetros de entrenamiento, reglas de consenso, calendarios de re-entrenamiento, lógica de warmup y estrategias de normalización son propietarios y no se divulgan.

---

## Ingeniería de Features

El modelo computa un **conjunto propietario de features financieras** a partir de datos OHLCV crudos, diseñadas para capturar dimensiones ortogonales del comportamiento del mercado:

| Categoría | Qué Captura |
|---|---|
| **Dinámica de precios** | Posicionamiento tendencial, momentum y tendencias de reversión a la media |
| **Régimen de volatilidad** | Ciclos de expansión/compresión de volatilidad y comportamiento intradía |
| **Comportamiento del volumen** | Cambios de liquidez, participación del mercado y momentum de volumen |
| **Divergencia de momentum** | Señales de agotamiento y dinámicas de distribución/acumulación |

Las features abarcan múltiples longitudes de ventanas rolling para capturar tanto shocks de corto plazo como transiciones de régimen de mediano plazo. Tras su cómputo, las features se redondean para mitigar el ruido de punto flotante introducido por la variabilidad entre sesiones de la fuente de datos.

Se aplica **reducción de dimensionalidad** post-normalización para comprimir el espacio de features preservando la estructura de varianza dominante. El método específico, el número de componentes retenidos y el alcance del ajuste/transformación están calibrados para evitar interacciones con el pipeline de warmup.

> ⚠️ **Nota:** El número de features, sus fórmulas específicas, longitudes de ventana, configuración de reducción de dimensionalidad y parámetros de normalización no se divulgan. Las categorías anteriores describen las dimensiones conceptuales capturadas, no la implementación.

---

## Metodología de Detección

El pipeline de detección opera en múltiples capas, cada una abordando un modo de fallo distinto de la detección de anomalías en datos financieros no estacionarios:

### Error de Cuantización como Señal de Anomalía
Cada día de negociación se mapea sobre la topología SOM entrenada. El **error de cuantización** (QE) — la distancia entre el vector de features del dato y su Unidad de Mejor Coincidencia (BMU, por sus siglas en inglés) — mide qué tan bien el estado de mercado actual encaja con los patrones históricos aprendidos. Un QE alto indica un estado de mercado que el SOM nunca aprendió a representar adecuadamente: genuinamente inusual, no simplemente volátil.

La distinción es importante: un día de alta volatilidad durante un régimen de volatilidad conocido se mapea bien en el SOM (QE bajo); el *mismo* nivel de volatilidad apareciendo durante un régimen previamente calmo se mapea pobremente (QE alto). El SOM captura **contexto**, no solo magnitud.

### Umbralización Multicapa
Los valores crudos de QE se convierten en señales de anomalía a través de múltiples métodos de umbralización independientes. Cada método captura una definición estadística diferente de "extremo", operando en distintas escalas temporales y con distintos supuestos distribucionales. Este enfoque por capas evita la fragilidad de cualquier umbral único — un requisito crítico cuando la distribución subyacente cambia entre mercados calmos, mercados tendenciales y regímenes de crisis.

### Filtrado por Consenso
Un punto de datos se clasifica como anómalo **solo cuando una combinación configurable de métodos de detección independientes coincide**. Esta es la decisión de diseño más importante del pipeline: transforma múltiples detectores ruidosos e individualmente imperfectos en un ensamble robusto. En la práctica, el sistema concentra las detecciones en estados de mercado genuinamente extremos, filtrando el ruido que cualquier método individual produciría.

### Clasificación de Anomalías
Las anomalías detectadas se clasifican según sus **drivers de features dominantes**, priorizados por magnitud de desviación respecto a los umbrales normales. Esto produce una taxonomía legible e interpretable de tipos de anomalía. Versiones anteriores producían un número excesivo de combinaciones de tipos; la lógica de clasificación se simplificó progresivamente para mejorar la interpretabilidad sin sacrificar calidad de detección.

> ⚠️ **Nota:** El número de métodos de detección, su implementación específica, las reglas de consenso, la lógica de clasificación y el número de drivers por anomalía son propietarios.

---

## Marco de Validación

El sistema incluye un pipeline de validación riguroso y multidimensional:

### Matching de Eventos Macro
Un catálogo curado de **eventos macro sistémicos** (que abarca crisis geopolíticas, decisiones de política monetaria, sell-offs de mercado, temporadas de resultados y eventos específicos por activo) se utiliza para medir la capacidad del modelo de detectar shocks de mercado conocidos. El catálogo se extendió progresivamente a lo largo de las versiones y cubre el período de análisis completo.

### Detección de Shocks Estadísticos
Shocks estadísticos identificados independientemente (retornos que exceden umbrales rolling adaptativos) se contrastan con las anomalías detectadas para medir la **tasa de captura de shocks** — el porcentaje de shocks estadísticos genuinos que el modelo señalizó exitosamente como anomalías.

### Backtesting con Retornos Futuros
Las anomalías detectadas se someten a backtesting en múltiples ventanas de retorno futuro para evaluar retornos promedio de días anómalos vs. días normales, precisión direccional y significancia estadística mediante **intervalos de confianza bootstrap** (IC 95%).

### Segmentación por Régimen
Todas las métricas de backtesting se computan separadamente para el período COVID (régimen de volatilidad extrema) y el período post-COVID (condiciones de mercado normalizadas). Esta segmentación previene que los retornos extremos de COVID inflen el rendimiento aparente del modelo — un sesgo explícitamente documentado y corregido a lo largo de las versiones. Sin esta segmentación, los win rates se inflaban entre 9 y 18 puntos porcentuales según el ticker.

### Win Rate por Tipo de Señal
Las anomalías se segmentan en señales **Bottom**, **Top** y **Neutral**, con win rates, expectancy, métricas de riesgo e intervalos de confianza computados por tipo, por ticker y por régimen.

---

## Resultados Cross-Asset

El modelo fue validado en **10 tickers que abarcan 7 perfiles de activo distintos**: índice amplio de mercado, ETF tecnológico, semiconductores, farmacéuticas, proxy de criptomonedas, mercados emergentes, oro/commodities, fintech (post-IPO), ride-hailing del sudeste asiático (post-IPO) y eVTOL (post-IPO).

**Observaciones clave:**

- El modelo es **agnóstico al ticker** por diseño — el mismo código y arquitectura se ejecuta en los 10 activos sin ajuste manual ni calibración por activo.
- La **tasa de captura de shocks** varía significativamente entre perfiles de activo, correlacionándose inversamente con la volatilidad base del activo. Los activos de mayor volatilidad producen movimientos grandes con mayor frecuencia, haciendo que cada shock individual sea menos distinguible del comportamiento normal.
- La **detección de eventos macro** varía según la exposición sectorial del activo a eventos globales, no por calidad del modelo — un activo con sensibilidad limitada a la geopolítica naturalmente coincide con menos eventos geopolíticos.
- Los activos post-IPO sin historial previo suficiente caen automáticamente en la ruta de normalización simplificada y aun así producen resultados significativos, aunque las primeras anomalías en estos activos deben interpretarse con cautela.
- El número de anomalías detectadas por ticker oscila aproximadamente entre 40 y 75 a lo largo de la ventana de análisis de ~1,500 días de negociación, dependiendo del perfil de volatilidad inherente del activo.

> ⚠️ **Nota:** Los conteos específicos de anomalías, porcentajes de captura de shocks y tasas de coincidencia macro por ticker están documentados internamente pero no se divulgan aquí para prevenir su uso como objetivos de calibración.

---

## Taxonomía de Señales y Accionabilidad

Las anomalías detectadas se clasifican en tres tipos de señal accionables según su perfil de features:

| Tipo de Señal | Interpretación | Mejor Caso de Uso |
|---|---|---|
| **Bottom** | Condiciones negativas extremas (perfil de crash) | Monitoreo de potenciales reversiones en activos establecidos |
| **Top** | Agotamiento/divergencia en niveles elevados | Evaluación de reducción de riesgo o timing de salida |
| **Neutral** | Incertidumbre de régimen sin dirección clara | Alerta de volatilidad — cobertura o reducción de exposición |

### Patrones Agregados (cross-asset)

Las **señales Bottom** funcionan bien en índices y acciones de gran capitalización (win rates consistentemente por encima del umbral de coin flip con significancia estadística), pero son **anti-predictivas en activos especulativos ultra-volátiles** — un hallazgo estructural confirmado en todos los tickers evaluados. La brecha entre estos dos grupos no es un problema de calibración; refleja la diferencia fundamental entre el comportamiento de reversión a la media y el comportamiento de momentum durante situaciones de estrés agudo.

Las **señales Top** muestran el patrón inverso: efectivas en activos especulativos y de alta volatilidad donde el agotamiento conduce a reversiones abruptas, pero poco fiables en activos con tendencia alcista secular donde las caídas son consistentemente absorbidas. Las señales Top además se degradan en ventanas de retorno futuro más largas para activos tendenciales.

Las **señales Neutral** son **universalmente fuertes en todos los tickers evaluados** — el único tipo de señal que mantiene significancia estadística independientemente del perfil de activo, régimen o nivel de volatilidad. Esto confirma el valor primario del modelo como **detector de incertidumbre de régimen**: excele en identificar *cuándo* el mercado se encuentra en territorio inexplorado, no *hacia dónde* se dirigirá.

### Backtesting Post-Anomalía — Significancia Estadística

Una nota de transparencia importante: **el backtesting de retornos futuros post-COVID alcanza significancia estadística solo en una minoría de los tickers evaluados** (IC bootstrap al 95% excluyendo cero). El modelo es informativo como indicador de riesgo en todos los activos, pero no debe interpretarse como un predictor universal de retornos. Donde se alcanza significancia, incluye tanto perfiles positivos (continuación de momentum) como negativos (reversión contrarian), dependiendo de la clase de activo.

---

## Análisis de Clustering de Anomalías

Más allá de la detección individual de anomalías, el sistema analiza el **clustering temporal** — si las anomalías se concentran en ráfagas o ocurren de forma aislada, y si las anomalías agrupadas en clusters presentan perfiles de retorno futuro diferentes a las aisladas.

Las anomalías dentro de una proximidad temporal configurable se agrupan en clusters. Cada cluster se caracteriza por su densidad, duración, perfil direccional y tipos de anomalía dominantes. Los clusters se comparan luego con las anomalías aisladas a lo largo de las ventanas de retorno futuro.

**Patrones observados:**

Las anomalías en cluster tienden a ocurrir durante estrés agudo del mercado — ráfagas densas de múltiples anomalías en ventanas cortas — mientras que las anomalías aisladas son más frecuentes durante eventos de resultados corporativos o eventos idiosincráticos. En ventanas de retorno futuro más largas, las anomalías en cluster mostraron retornos absolutos significativamente mayores que las aisladas en varios tickers, aunque esta relación **no es estable cross-asset**. En al menos un activo, la correlación entre densidad de cluster y magnitud de retorno futuro fue **negativa**, contradiciendo directamente el patrón observado en otros.

Este análisis proporciona contexto narrativo para entender *cuándo* el modelo dispara en ráfagas vs. de forma aislada, pero no genera reglas de trading universales. Los patrones de clustering son dependientes del activo y del régimen.

---

## Hallazgos Clave

1. **El modelo es un indicador de riesgo/incertidumbre, no un predictor direccional.** Las señales Neutral (cambios de régimen) son el producto más fiable en todos los tickers evaluados. La significancia estadística de retornos futuros post-COVID se alcanza solo en una minoría de activos — el modelo informa la evaluación de riesgo, no la predicción de retornos.

2. **Las señales direccionales son bimodales por tipo de activo.** Las señales Bottom funcionan para índices y large caps; las señales Top funcionan para activos especulativos y de alta volatilidad. Esto no es un fallo de calibración — refleja la diferencia estructural entre cómo se manifiestan los crashes agudos (extremos simultáneos en múltiples features) y los techos graduales (impulsados por divergencia) en diferentes perfiles de activo.

3. **La segmentación COVID es indispensable.** Sin separar los datos de la era COVID, los win rates se inflan entre 9 y 18 puntos porcentuales según el ticker. Todas las métricas reportadas utilizan análisis segmentado. Los benchmarks anteriores están formalmente documentados como inflados debido a un sesgo de inmadurez de features que solo se corrigió en la versión estable final.

4. **El oro se comporta como señal contrarian.** Las anomalías en oro marcan techos locales que corrigen — la única clase de activo donde el backtesting post-anomalía es significativamente negativo. Esto introduce una categoría de uso donde la detección de anomalías funciona como señal de **reducción de posición** en lugar de señal de entrada.

5. **La estrategia de normalización es la decisión de diseño con mayor impacto** — más que la topología del SOM o la selección de features. Se requirieron múltiples iteraciones para encontrar un enfoque que estabilice las features desde el primer día sin introducir look-ahead bias ni distorsionar la distribución de la ventana de detección. Esta lección es el hallazgo más transferible del proyecto para cualquiera que construya sistemas de ML con ventanas expansivas sobre datos financieros.

6. **Los techos graduales permanecen estructuralmente indetectables.** La distribución lenta sin picos de volatilidad no dispara anomalías en el SOM. Los pisos generan múltiples features en extremos simultáneos; los techos requieren features específicas de divergencia que se activan de forma más sutil. Esta es una limitación fundamental de la detección basada en topología.

7. **Las interacciones entre limitaciones no son obvias y se componen.** Rastrear las limitaciones como un sistema interconectado — no como una checklist independiente — fue esencial para avanzar a lo largo de las versiones.

---

## Limitaciones Conocidas y Transparencia

Este proyecto mantiene un documento formal de limitaciones con **20 ítems rastreados** (8 resueltos, 5 estructurales sin solución, 2 mitigados, 5 documentados). En lugar de listarlos independientemente, lo que sigue describe cómo se conectan las limitaciones más importantes.

La restricción más fundamental es arquitectónica: **los Mapas Auto-Organizados detectan anomalías a través de distancia topológica**, lo que significa que inherentemente requieren que el punto de datos esté lejos de todos los patrones aprendidos. Los techos graduales de mercado — donde la distribución ocurre lentamente sin producir valores extremos de features en ningún día individual — nunca generan alto error de cuantización, y por lo tanto nunca disparan detección. Esto no es un problema de ajuste; es un techo estructural de los métodos basados en topología. Por contraste, los pisos de mercado tienden a producir múltiples features en extremos simultáneos (picos de volatilidad, explosiones de volumen, expansión de rango), haciéndolos naturalmente visibles para el SOM.

Esta asimetría se propaga hacia la **bimodalidad direccional** observada cross-asset. Las señales Bottom son efectivas en activos donde el estrés agudo es seguido por reversión a la media (índices, large caps), pero se vuelven **anti-predictivas en activos especulativos ultra-volátiles** donde un perfil de features similar a un crash es simplemente un martes más. En esos activos, lo que parece un piso para el SOM es en realidad un patrón de continuación. Inversamente, las señales Top funcionan en activos especulativos — donde el agotamiento conduce a reversiones abruptas — pero fallan en activos con tendencia alcista secular donde las caídas son consistentemente absorbidas. Esta bimodalidad está confirmada en los 10 tickers y es estructural, no dependiente de calibración. A su vez, interactúa con la **sensibilidad de shocks**: la capacidad del modelo para capturar shocks estadísticos se degrada a medida que la volatilidad base del activo aumenta, porque los movimientos extremos se vuelven menos distinguibles del comportamiento normal cuando la volatilidad ya es elevada.

En el plano técnico, la limitación resuelta de mayor impacto fue la **inmadurez de features** en los primeros días de la ventana de análisis. Antes de implementar el mecanismo de warmup, el escalador expansivo arrancaba en frío — produciendo normalizaciones distorsionadas que contaminaban las primeras anomalías con artefactos. Esto interactuaba con la **inflación de la era COVID**: los retornos extremos de marzo de 2020 generaron win rates aparentemente espectaculares que eran parcialmente atribuibles a features inmaduras, no a edge genuino del modelo. Corregir esto requirió segmentación por régimen (separar métricas COVID de post-COVID) y un rediseño de la normalización que estabiliza desde el día uno usando datos de pre-historia — sin permitir que esa pre-historia distorsione los supuestos distribucionales de la ventana de detección. La solución requirió tres iteraciones a lo largo de las versiones para quedar correcta, y cada enfoque intermedio introdujo sus propios artefactos.

Una cascada relacionada involucró **parámetros de warmup propagándose hacia la topología del modelo**: cuando el período de pre-historia cambiaba el tamaño efectivo del dataset, alteraba parámetros computados dinámicamente (dimensiones de grilla, componentes de reducción de dimensionalidad), produciendo cambios no obvios en el comportamiento de detección. La solución fue **hardcodear** todos los parámetros sensibles a la topología, desacoplándolos del tamaño de datos. Esta es una lección generalizable: en pipelines de ML con múltiples etapas, cualquier parámetro computado dinámicamente a partir de los datos puede convertirse en un acoplamiento oculto entre etapas.

Entre las limitaciones más blandas, se reconoce la **multicolinealidad de features** — varias features están correlacionadas y codifican conceptos superpuestos. La reducción de dimensionalidad absorbe la mayor parte de esta redundancia, pero el conjunto de features probablemente podría comprimirse sin pérdida de rendimiento. El **análisis de clustering** produce narrativas interesantes por activo pero ninguna regla universal — las correlaciones densidad-magnitud varían en signo entre tickers, haciendo que las reglas de trading basadas en clusters sean poco confiables como generalizaciones cross-asset.

Finalmente, la "limitación" más importante es también el hallazgo central del proyecto: **Neutral es la única señal universalmente fiable**. Todos los demás tipos de señal tienen efectividad dependiente del activo. Esto significa que el producto principal del modelo es la detección de incertidumbre de régimen — identificar *cuándo* el mercado está en territorio genuinamente inusual — no la predicción direccional. Aceptar esto redefinió el alcance del proyecto y, paradójicamente, hizo sus outputs más útiles.

> Documento completo de limitaciones (20 ítems con mapas de interacción) disponible bajo solicitud.

---

## Stack Tecnológico y Dependencias

| Componente | Tecnología | Propósito |
|---|---|---|
| **Lenguaje** | Python 3.10+ | Desarrollo central |
| **Adquisición de datos** | `yfinance` | Descarga de datos OHLCV de mercado |
| **Cómputo numérico** | `NumPy`, `SciPy` | Operaciones con arrays, tests estadísticos, bootstrap CI |
| **Manipulación de datos** | `Pandas` | Manejo de series temporales, cómputo de features |
| **Machine learning** | `scikit-learn` | Reducción de dimensionalidad, preprocesamiento |
| **Motor SOM** | `MiniSom` | Entrenamiento e inferencia de Mapas Auto-Organizados |
| **Visualización** | `Matplotlib` | Gráficos diagnósticos multi-panel |
| **Entorno** | Google Colab | Ejecución y prototipado |

### Reproducibilidad

- Se establece una semilla aleatoria global para la inicialización del SOM y el muestreo bootstrap.
- Los valores de features se redondean post-cómputo para reducir la varianza entre sesiones.
- Todos los hiperparámetros están centralizados en un encabezado de configuración.

> ⚠️ **Nota:** El código fuente no está incluido en este repositorio. Este README documenta la metodología, resultados y proceso de desarrollo profesional del proyecto.

---

## Aviso Legal

Este proyecto fue desarrollado como un **ejercicio de investigación y desarrollo profesional** en finanzas cuantitativas y aprendizaje automático aplicado. No pretende ser asesoría financiera, un sistema de trading ni una recomendación de inversión.

- El modelo es un **indicador de incertidumbre de régimen de mercado** — identifica *cuándo* el mercado se encuentra en un estado inusual, no *hacia dónde* irá. Esta distinción es fundamental.
- La significancia estadística de retornos futuros post-COVID se alcanza solo en una minoría de los tickers evaluados. El modelo no debe interpretarse como un predictor universal de retornos.
- El rendimiento pasado en detección de anomalías no garantiza resultados futuros. Los mercados son no estacionarios; las relaciones estadísticas observadas pueden degradarse o revertirse.
- Todos los resultados están basados en backtesting histórico con limitaciones conocidas (20 formalmente rastreadas, documentadas arriba).
- No se realizó trading real ni paper trading con este sistema.
- El caso de uso validado más fuerte del modelo es como **capa de monitoreo de riesgo** — señalizar momentos que ameritan la atención de un analista humano — no como generador autónomo de señales.

---

## Autor

Desarrollado como parte de un portafolio profesional que demuestra competencia en:
- **Ingeniería de datos financieros** — pipelines de datos de mercado con normalización expansiva y estabilización por warmup
- **Machine learning no supervisado** — Mapas Auto-Organizados aplicados a datos financieros no estacionarios
- **Metodología de validación rigurosa** — intervalos de confianza bootstrap, segmentación por régimen, testing cross-asset
- **Disciplina de desarrollo iterativo** — rastreo formal de limitaciones, corrección de sesgos y análisis de interacciones entre versiones
- **Arquitectura de software en Python** — ~1,900 líneas organizadas en bloques modulares y reproducibles

---

*© 2025. Todos los derechos reservados. El código fuente, parámetros del modelo y lógica de detección son propietarios y no están disponibles para distribución o reproducción.*
