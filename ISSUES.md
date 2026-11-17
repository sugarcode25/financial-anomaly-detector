# Errores conocidos — financial-anomaly-detector (v9.3)

**Estado:** repositorio **archivado** (noviembre de 2026). La versión corregida está en [financial-anomaly-detector-v2](https://github.com/sugarcode25/financial-anomaly-detector-v2). El README de esta versión se conserva sin cambios como registro histórico.

**Alcance:** los hallazgos se verificaron sobre el código v9.3 (congelado el 8-abr-2026) y sus corridas cross-asset (10 tickers, marzo de 2026). **Las cifras y conclusiones del README de este repositorio no deben usarse.**

Severidad: **A** = error que cambia la conclusión · **B** = error metodológico · **C** = presentación.

## Errores de severidad A

| # | Problema | Evidencia | Estado |
|---|---|---|---|
| 1 | **Fuga de información (look-ahead).** El README afirma que el modelo no mira al futuro. Solo la normalización cumple eso: varios componentes del modelo y sus umbrales se ajustan con toda la muestra, incluidos días posteriores a los que califican. | Prueba de truncación en SPY con la misma descarga (variables idénticas): al agregar los datos de 2024–2026, las anomalías de 2020 a jun-2023 pasan de 25 a 40 (cambian 19). La versión corregida, en la misma prueba, cambia 0. | ✅ Corregido en la v2 |
| 2 | **"Neutral es la señal universalmente fiable" es un efecto de la definición.** Un Neutral "gana" si el precio se mueve más de 1% en cualquier dirección, y se compara contra 50%. Casi cualquier día cumple esa condición. | Con datos sintéticos sin ninguna información, Neutral obtiene 79–87% marcado como significativo. Los días normales "ganan" 52–81% con la misma definición. | ✅ Retirado en la v2 (se mide contra una línea base) |
| 3 | **No hay muestra de prueba.** Los parámetros se ajustaron durante 9 versiones mirando resultados del mismo periodo (2020–2026) que luego se presenta como validación. | Historial de versiones en el código y en el documento de limitaciones. | ✅ Corregido en la v2 (protocolo prerregistrado; prueba desde el 9-abr-2026, una sola vez) |
| 4 | **Nada es verificable en el repositorio.** No hay código, figuras ni tablas por ticker. | El repositorio solo contiene el README. | 🟡 Atendido en parte: la v2 publica figuras, tablas y la celda de origen de cada número; el código sigue siendo privado |
| 5 | **Datos de SOFI, GRAB y JOBY anteriores a su listado son de la SPAC previa.** Ese tramo concentra buena parte de sus anomalías "COVID". | Antes de la fecha de listado: JOBY 189 sesiones con precio mediano US$10.29 y volumen 3× menor que después; GRAB 253 sesiones, US$11.84 y volumen 39× menor; SOFI 102 sesiones, volumen 4× menor. En v9.3, 39 de 65 anomalías de GRAB caen en ese tramo. | ✅ Corregido en la v2: cada activo empieza en su fecha de listado, verificada en SEC (SOFI 1-jun-2021, JOBY 11-ago-2021, GRAB 2-dic-2021) |

## Errores de severidad B

| # | Problema | Estado |
|---|---|---|
| 6 | No hay línea base: no se compara contra una regla simple (por ejemplo, marcar los días de mayor volatilidad reciente), así que "indicador de incertidumbre" no está demostrado. | ✅ Corregido en la v2 (regla con la misma frecuencia de alertas) |
| 7 | No hay corrección por comparaciones múltiples (cientos de pruebas entre tickers, ventanas y segmentos). | ✅ Corregido en la v2 (7 pruebas, Benjamini-Hochberg) |
| 8 | El bootstrap trata como independientes anomalías agrupadas en un mismo episodio. Ejemplo: 12 de las 30 anomalías post-COVID de SPY son un solo episodio (abril de 2025). | ✅ Corregido en la v2 (episodios y errores Driscoll-Kraay) |
| 9 | Las métricas de señales comparan contra 50% en vez de contra los días normales, eligen la "mejor ventana" por resultado y marcan `*` sin prueba estadística. | ✅ Corregido en la v2 (regresión contra el modelo HAR, ventanas fijadas de antemano) |
| 10 | La validación con eventos macro no tiene línea base por azar, y el catálogo incluye temporadas de resultados y eventos sin fuente. | ✅ Retirado en la v2 |

## Presentación (C)

Licencia y año inconsistentes (© 2025, sin archivo `LICENSE`), parámetros visibles en los títulos de las figuras y afirmaciones de método sin respaldo en el repositorio. ✅ Corregido en la v2 (`LICENSE` en dos capas, figuras sin parámetros del modelo).

## Resultado de la v2

La v2 corrige la fuga (0 cambios en la prueba de truncación) y somete el modelo a una prueba fuera de muestra prerregistrada, corrida una sola vez (9-abr a 16-nov-2026, 10 activos). **Ninguna de las 7 pruebas fue significativa: el resultado es no concluyente.** Detalle en el [README de la v2](https://github.com/sugarcode25/financial-anomaly-detector-v2).
