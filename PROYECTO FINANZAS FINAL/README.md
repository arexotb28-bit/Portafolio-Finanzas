# PortfolioLab

PortfolioLab es una aplicación local desarrollada con Streamlit para analizar
cobertura, riesgo y rentabilidad, optimización de portafolios, CAPM y
regresiones OLS. No requiere una base de datos externa ni credenciales: todos
los datos se leen desde `base_diaria.xlsx`.

## 1. Descargar el proyecto completo

Conserva estos archivos y carpetas juntos, sin cambiar sus nombres:

```text
PROYECTO FINANZAS FINAL/
├── base_diaria.xlsx
├── herramienta_finanzas_completa.py
├── .python-version
├── .env.example
├── requirements.txt
├── README.md
├── outputs/
└── tests/
    └── test_financial_engine.py
```

La aplicación utiliza rutas relativas. Por eso la carpeta puede copiarse a
cualquier equipo y ubicación; no contiene rutas personales de la autora.

## 2. Requisitos del equipo

- Windows, macOS o Linux.
- Python 3.12 de 64 bits (recomendado y declarado en `.python-version`).
- Aproximadamente 1 GB libre para Python, dependencias y el entorno virtual.
- Un navegador web moderno.
- Conexión a internet solo durante la instalación de dependencias.

Comprueba Python desde una terminal:

```bash
python --version
```

En Windows también puedes probar `py --version`; en macOS o Linux,
`python3 --version`. Si Python no está instalado, descárgalo desde
[python.org](https://www.python.org/downloads/) y activa la opción para agregarlo
al `PATH` cuando el instalador la ofrezca.

## 3. Crear un entorno virtual

Abre una terminal dentro de la carpeta del proyecto. También puedes usar `cd`
seguido de la ruta donde descomprimiste o copiaste la carpeta.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación, ejecuta una vez en esa misma ventana:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS o Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Cuando el entorno está activo, la terminal suele mostrar `(.venv)` al inicio.

## 4. Verificar la instalación

Ejecuta las pruebas antes de abrir la aplicación:

```bash
python -m unittest discover -s tests -v
```

El resultado correcto termina con:

```text
Ran 4 tests
OK
```

Las pruebas verifican que el Excel pueda leerse, que los cambios del archivo se
detecten, que estén disponibles los diagnósticos OLS y que los datos limpios se
puedan exportar a una carpeta relativa.

## 5. Ejecutar PortfolioLab

Con el entorno virtual activo y desde la raíz del proyecto, ejecuta:

```bash
python -m streamlit run herramienta_finanzas_completa.py
```

Streamlit mostrará una dirección local, normalmente:

```text
http://localhost:8501
```

Si el navegador no se abre automáticamente, copia esa dirección en el
navegador. Para detener la aplicación, vuelve a la terminal y presiona
`Ctrl+C`.

## 6. Formato requerido del Excel

El archivo debe llamarse `base_diaria.xlsx` y permanecer en la raíz del
proyecto. Debe contener exactamente estas hojas:

| Hoja | Contenido esperado |
|---|---|
| `ACTIVOS` | Ticker, RIC, grupo y tipo de cada activo. |
| `BASE FINAL` | Fechas y precios históricos de los activos. |
| `BENCHMARK` | Fechas y precios históricos del benchmark SPX. |
| `T-BILL` | Fechas y tasa libre de riesgo anual expresada en porcentaje. |

La aplicación espera la estructura original de encabezados de Refinitiv. Para
actualizar datos, modifica valores o agrega observaciones sin borrar las hojas,
los encabezados ni las columnas existentes.

## 7. Actualización automática del Excel

Mientras PortfolioLab está abierto, guarda los cambios en
`base_diaria.xlsx`. La aplicación comprueba periódicamente su tamaño y fecha de
modificación. Antes de recargar exige dos comprobaciones idénticas y valida que
el archivo sea un Excel completo, evitando leerlo mientras todavía se guarda.

Cuando detecta una versión nueva:

- invalida el caché anterior;
- recarga activos, precios, benchmark y tasa libre de riesgo;
- actualiza el rango de fechas;
- vuelve a calcular los módulos con los filtros seleccionados.

El botón **Actualizar análisis** permite forzar una revisión manual. También se
puede cargar temporalmente otro `.xlsx` desde la barra lateral; esa acción no
reemplaza el archivo local.

## 8. Exportar datos limpios

El botón **Guardar datos limpios** crea, dentro de `outputs/`, estos archivos:

- `precios_limpios.xlsx`;
- `benchmark_limpio.xlsx`;
- `tbill_limpio.xlsx`;
- `indicadores.xlsx`.

La exportación es opcional y usa rutas relativas al proyecto. No altera el
Excel original ni las fórmulas financieras.

## 9. Solución de problemas

### Si existen errores de paquetes

Activa el entorno virtual y reinstala el conjunto compatible declarado:

```bash
pip uninstall numpy pandas scipy -y
pip install -r requirements.txt
```

Si el error persiste, elimina solamente `.venv`, confirma que estás usando
Python 3.12 y crea el entorno nuevamente. Esto evita errores como
`Importing numpy C extensions failed` causados por mezclar binarios de versiones
distintas de Python o NumPy.

### `python`, `python3` o `py` no se reconoce

Python no está instalado o no está agregado al `PATH`. Instálalo nuevamente y
activa la opción correspondiente, o abre una terminal nueva después de la
instalación.

### No se puede activar `.venv` en PowerShell

Usa el comando `Set-ExecutionPolicy` de la sección de Windows PowerShell. Su
alcance es únicamente la ventana actual.

### `No module named streamlit`

Activa el entorno virtual y vuelve a ejecutar:

```bash
python -m pip install -r requirements.txt
```

### La aplicación indica que faltan hojas

Confirma que el archivo se llame `base_diaria.xlsx`, esté junto al script y
contenga `ACTIVOS`, `BASE FINAL`, `BENCHMARK` y `T-BILL` con esos nombres.

### El Excel está abierto y aparece un error de lectura

Termina de guardarlo y espera unos segundos. Si continúa, cierra Excel y pulsa
**Actualizar análisis**.

### El puerto 8501 está ocupado

Ejecuta la aplicación en otro puerto:

```bash
python -m streamlit run herramienta_finanzas_completa.py --server.port 8502
```

### La optimización tarda al iniciar

Es normal con muchos activos y miles de portafolios simulados. Reduce
temporalmente la cantidad de activos o el control **Portafolios simulados**. Los
resultados se guardan en caché y no se recalculan al cambiar de módulo mientras
los datos y parámetros permanezcan iguales.

## 10. Repetir la instalación desde cero

Para probar una instalación limpia, elimina únicamente la carpeta `.venv`,
vuelve a crearla con los comandos de la sección 3, instala `requirements.txt`,
ejecuta las pruebas y luego inicia Streamlit. No es necesario modificar el
código ni usar rutas absolutas.
