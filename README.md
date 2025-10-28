# CorkyScab :gb: EN – English
Package to segment corky scab lesions on potato tuber images using OpenCV + K-Means.
Includes:
- `CorkyScabSegmenter` class
- `analyze_corky_scab(image_path)` -> (percentage, mask)
- `analyze_corky_scab_to_df(image_path)` -> pandas.DataFrame with image name & lesion percentage
- `save_figures(fig, out_dir, base_name)` -> save PNG and PDF
- `save_results_df(df, out_path)` -> save CSV

# Installation
You can install CorkyScab from a `.zip` file:

```bash
pip install /path/to/corkyscab.zip
```
# Usage examples
 `corkyscab_demo.ipynb` file.

--- 

# LLS SevEst :es: ESP – Español

Paquete para estimar la severidad de lesiones por corchosis en imágenes de papas mediante segmentación KMeans.

## Instalación
Puedes instalar lls_sev_est desde un archivo `.zip`: 
```bash
pip install /path/to/corkyscab.zip
```

# Ejemplos de uso
Archivo  `corkyscab_demo.ipynb` .
