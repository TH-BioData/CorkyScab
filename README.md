
# corky_scab

Tools to segment corky scab lesions on potato tuber images using OpenCV + K-Means.
Includes:
- `CorkyScabSegmenter` class
- `analyze_corky_scab(image_path)` -> (percentage, mask)
- `analyze_corky_scab_to_df(image_path)` -> pandas.DataFrame with image name & lesion percentage
- `save_figures(fig, out_dir, base_name)` -> save PNG and PDF
- `save_results_df(df, out_path)` -> save CSV

> Designed for Google Colab. Make sure OpenCV and scikit-learn are installed.
