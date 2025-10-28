
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage import filters, exposure, morphology, segmentation
import warnings
warnings.filterwarnings('ignore')

class CorkyScabSegmenter:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model_name = "Optimized K-Means with Aggressive Smoothing"
        self.features_used = "LAB-L, LAB-A, Texture-Gradient, Local-Contrast"
        self.processing_size = 600   # Max width for processing
        self.view_size = 800         # Max width for visualization

    def segment_potato(self, image_bgr):
        """Segments the potato in the image using the provided code"""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        potato_mask = cv2.bitwise_not(mask_blue)
        potato_mask = cv2.morphologyEx(potato_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(potato_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(potato_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)
        return final_mask

    def load_image(self, image_path):
        """Loads and preprocesses the image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        h, w = image.shape[:2]
        if w > self.processing_size:
            scale = self.processing_size / w
            new_w = self.processing_size
            new_h = int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h))
            print(f"🔧 Image resized for processing: {new_h}x{new_w}")
        else:
            resized_image = image
        potato_mask = self.segment_potato(resized_image)
        segmented_image = cv2.bitwise_and(resized_image, resized_image, mask=potato_mask)
        image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        return image_rgb, potato_mask, resized_image

    def apply_aggressive_smoothing(self, image):
        smoothed = cv2.GaussianBlur(image, (15, 15), 0)
        smoothed = cv2.bilateralFilter(smoothed, 15, 100, 100)
        smoothed = cv2.medianBlur(smoothed, 7)
        return smoothed

    def preprocess_image(self, image, potato_mask):
        masked_image = cv2.bitwise_and(image, image, mask=potato_mask)
        smoothed = self.apply_aggressive_smoothing(masked_image)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        contrast_improved = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return contrast_improved, potato_mask

    def extract_task_specific_features(self, image, potato_mask):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = image.shape[:2]
        potato_idx = np.where(potato_mask.flatten() > 0)[0]
        n_potato_pixels = len(potato_idx)
        feats = np.zeros((n_potato_pixels, 4))
        lab_flat = lab.reshape(-1, 3)
        gray_smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(gray_smoothed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(gray_smoothed, -1, kernel)
        local_contrast = np.abs(gray_smoothed - local_mean)
        feats[:, 0] = lab_flat[potato_idx, 0]
        feats[:, 1] = lab_flat[potato_idx, 1]
        feats[:, 2] = grad_mag.flatten()[potato_idx]
        feats[:, 3] = local_contrast.flatten()[potato_idx]
        return feats, (h, w), potato_idx

    def kmeans_segmentation_optimized(self, image, potato_mask, n_clusters=2):
        processed_image, processed_mask = self.preprocess_image(image, potato_mask)
        feats, dims, potato_idx = self.extract_task_specific_features(processed_image, processed_mask)
        if len(feats) == 0:
            print("⚠️ No pixels found inside the segmented potato.")
            return np.zeros_like(potato_mask), None, processed_image
        feats_norm = self.scaler.fit_transform(feats)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feats_norm)
        h_proc, w_proc = processed_image.shape[:2]
        seg_map_proc = np.zeros(h_proc * w_proc, dtype=np.int32)
        seg_map_proc[potato_idx] = labels + 1
        seg_map_proc = seg_map_proc.reshape(h_proc, w_proc)
        return seg_map_proc, kmeans, processed_image

    def aggressively_clean_false_positives(self, lesion_mask, potato_mask, min_area=500):
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        clean = cv2.morphologyEx(lesion_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_open)
        n_comp, labels_img, stats, centroids = cv2.connectedComponentsWithStats(clean, 8)
        final_mask = np.zeros_like(clean)
        for i in range(1, n_comp):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                final_mask[labels_img == i] = 1
        final_mask = cv2.bitwise_and(final_mask, final_mask, mask=potato_mask)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
        return final_mask

    def identify_clusters_simple(self, image, seg_map):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0]
        clusters_info = []
        for cluster_id in [1, 2]:
            mask = (seg_map == cluster_id)
            if np.any(mask):
                mean_L = float(np.mean(L[mask]))
                area = int(np.sum(mask))
                clusters_info.append({'id': cluster_id, 'mean_luminance': mean_L, 'area': area, 'type': 'unknown'})
        if len(clusters_info) == 2:
            clusters_info.sort(key=lambda x: x['mean_luminance'])
            clusters_info[0]['type'] = 'lesion'
            clusters_info[1]['type'] = 'healthy'
        elif len(clusters_info) == 1:
            clusters_info[0]['type'] = 'lesion' if clusters_info[0]['mean_luminance'] < 100 else 'healthy'
        return clusters_info

    def compute_corky_scab_percentage(self, image_path, return_fig=False):
        try:
            image, potato_mask, original_image = self.load_image(image_path)
            print(f"✅ Image loaded and potato segmented: {image.shape}")
            print(f"✅ Potato area: {np.sum(potato_mask > 0)} pixels")
            seg_map, model, processed_image = self.kmeans_segmentation_optimized(image, potato_mask, n_clusters=2)
            print("✅ K-Means segmentation completed")
            clusters_info = self.identify_clusters_simple(image, seg_map)
            print("✅ Cluster identification completed")
            lesion_mask = None
            for cl in clusters_info:
                if cl['type'] == 'lesion':
                    lesion_mask = (seg_map == cl['id'])
                    break
            if lesion_mask is None:
                print("⚠️ Lesion cluster not identified. Using the darkest cluster as fallback.")
                if clusters_info:
                    clusters_info.sort(key=lambda x: x['mean_luminance'])
                    lesion_mask = (seg_map == clusters_info[0]['id'])
                else:
                    lesion_mask = np.zeros_like(seg_map, dtype=bool)
            print("🔧 Applying aggressive false-positive cleanup...")
            lesion_mask_clean = self.aggressively_clean_false_positives(lesion_mask, potato_mask, min_area=300)
            print("✅ Aggressive post-processing completed")
            total_potato_pixels = int(np.sum(potato_mask > 0))
            lesion_pixels = int(np.sum(lesion_mask_clean))
            lesion_percentage = (lesion_pixels / total_potato_pixels) * 100 if total_potato_pixels > 0 else 0.0
            fig = self.visualize_results_optimized(
                image, seg_map, lesion_mask_clean, potato_mask, clusters_info, lesion_percentage, original_image
            )
            if return_fig:
                return lesion_percentage, lesion_mask_clean, fig
            return lesion_percentage, lesion_mask_clean, None
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def visualize_results_optimized(self, image, seg_map, lesion_mask,
                                    potato_mask, clusters_info, lesion_percentage, original_image):
        h, w = image.shape[:2]
        if w > self.view_size:
            scale = self.view_size / w
            new_w = self.view_size
            new_h = int(h * scale)
            image_vis = cv2.resize(image, (new_w, new_h))
            seg_vis = cv2.resize(seg_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            lesion_vis = cv2.resize(lesion_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            potato_mask_vis = cv2.resize(potato_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            image_vis = image
            seg_vis = seg_map
            lesion_vis = lesion_mask
            potato_mask_vis = potato_mask
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f'POTATO CORKY SCAB ANALYSIS - AGGRESSIVE SMOOTHING\n'
            f'Model: {self.model_name} | 2 Clusters (Lesion/Healthy skin)',
            fontsize=14, fontweight='bold', y=0.95
        )
        h_orig, w_orig = original_image.shape[:2]
        if w_orig > self.view_size:
            scale_orig = self.view_size / w_orig
            new_w_orig = self.view_size
            new_h_orig = int(h_orig * scale_orig)
            orig_vis = cv2.resize(original_image, (new_w_orig, new_h_orig))
            orig_vis = cv2.cvtColor(orig_vis, cv2.COLOR_BGR2RGB)
        else:
            scale_orig = 1.0
            orig_vis = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(orig_vis)
        contours, _ = cv2.findContours(potato_mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if w_orig > self.view_size:
                contour = contour * scale_orig
            axes[0, 0].plot(contour[:, 0, 0], contour[:, 0, 1], 'y-', linewidth=2)
        axes[0, 0].set_title('ORIGINAL IMAGE + POTATO CONTOUR', fontweight='bold')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(potato_mask_vis, cmap='gray')
        axes[0, 1].set_title('POTATO MASK', fontweight='bold')
        axes[0, 1].axis('off')
        seg_color = np.zeros_like(image_vis)
        for cl in clusters_info:
            cl_mask = (seg_vis == cl['id'])
            if cl['type'] == 'lesion':
                seg_color[cl_mask] = [255, 0, 0]
            elif cl['type'] == 'healthy':
                seg_color[cl_mask] = [0, 255, 0]
        axes[0, 2].imshow(seg_color)
        axes[0, 2].set_title('SEGMENTATION (Red: Lesion, Green: Healthy)', fontweight='bold')
        axes[0, 2].axis('off')
        axes[1, 0].imshow(lesion_vis, cmap='Reds')
        axes[1, 0].set_title(
            f'CLEANED LESION MASK\n{lesion_percentage:.2f}% of potato area',
            fontweight='bold', color='red'
        )
        axes[1, 0].axis('off')
        overlay = image_vis.copy()
        overlay[lesion_vis.astype(bool)] = [255, 0, 0]
        for contour in contours:
            cv2.drawContours(overlay, [contour], -1, (255, 255, 0), 2)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('LESIONS (RED) + POTATO CONTOUR (YELLOW)', fontweight='bold')
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
        info_text = "DETAILED INFO:\n\n"
        info_text += f"LESION PERCENTAGE: {lesion_percentage:.2f}%\n\n"
        info_text += f"POTATO AREA: {np.sum(potato_mask > 0):,} pixels\n\n"
        info_text += "AGGRESSIVE CONFIGURATION:\n"
        info_text += "• Gaussian + Bilateral + Median smoothing\n"
        info_text += "• Wide morphological operations\n"
        info_text += "• Removal of small components\n\n"
        info_text += "IDENTIFIED CLUSTERS:\n"
        for cl in clusters_info:
            info_text += f"\n● {cl['type'].upper()}:\n"
            info_text += f" - Mean luminance: {cl['mean_luminance']:.1f}\n"
            info_text += f" - Area: {cl['area']:,} pixels\n"
        axes[1, 2].text(
            0.05, 0.95, info_text, transform=axes[1, 2].transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray")
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.2, wspace=0.2)
        # Return the figure so users can optionally save it
        return fig

# ---- Convenience functions ----

def analyze_corky_scab(image_path, return_fig=False):
    """Run the full pipeline and return (percentage, mask, fig|None)."""
    print("🔍 Starting corky scab analysis with aggressive smoothing...")
    segmenter = CorkyScabSegmenter()
    percentage, mask, fig = segmenter.compute_corky_scab_percentage(image_path, return_fig=return_fig)
    if percentage is not None:
        print("✅ ANALYSIS COMPLETED")
        print(f"📊 Corky scab percentage: {percentage:.2f}%")
    else:
        print("❌ Analysis error")
    return percentage, mask, fig

def analyze_corky_scab_to_df(image_path):
    """Returns a pandas DataFrame with columns: image_name, lesion_percentage."""
    percentage, mask, _ = analyze_corky_scab(image_path, return_fig=False)
    image_name = os.path.basename(image_path)
    df = pd.DataFrame([{'image_name': image_name, 'lesion_percentage': percentage if percentage is not None else np.nan}])
    return df

def save_figures(fig, out_dir, base_name="corky_scab"):
    """Save figure to PNG and PDF in the given directory."""
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{base_name}.png")
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ Figures saved to:\n - {png_path}\n - {pdf_path}")
    return png_path, pdf_path

def save_results_df(df, out_path):
    """Save the provided DataFrame to CSV."""
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"🧾 Results saved to: {out_path}")
    return out_path

def detect_corky_scab(image_name: str):
    if __name__ == "__main__":
        image_path = image_name
        if os.path.exists(image_path):
            analyze_corky_scab(image_path, return_fig=False)
        else:
            print(f"⚠️ File not found: {image_path}")
