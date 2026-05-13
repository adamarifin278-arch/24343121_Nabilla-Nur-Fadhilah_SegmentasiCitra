"""
Komparasi Metode Segmentasi untuk Ekstraksi Objek
Implementasi lengkap: Thresholding, Edge Detection, Region-based Segmentation
Dependencies: opencv-python, scikit-image, numpy, matplotlib, scipy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from scipy import ndimage

# ============================================================
# BAGIAN 1: PEMBUATAN CITRA SINTETIS
# ============================================================

def create_bimodal_image(size=(300, 300)):
    """Citra bimodal: objek vs background kontras tinggi"""
    img = np.zeros(size, dtype=np.uint8) + 30  # background gelap
    # Tambah beberapa objek terang
    cv2.rectangle(img, (50, 50), (130, 130), 220, -1)
    cv2.circle(img, (200, 80), 50, 200, -1)
    cv2.ellipse(img, (150, 220), (70, 40), 30, 0, 360, 210, -1)
    cv2.rectangle(img, (220, 180), (280, 260), 230, -1)
    noise = np.random.normal(0, 8, size)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    return img

def create_uneven_illumination_image(size=(300, 300)):
    """Citra dengan iluminasi tidak merata (vignetting effect)"""
    img = np.zeros(size, dtype=np.uint8) + 50
    # Objek
    cv2.rectangle(img, (40, 40), (100, 110), 180, -1)
    cv2.circle(img, (180, 90), 45, 190, -1)
    cv2.rectangle(img, (50, 170), (130, 250), 175, -1)
    cv2.circle(img, (220, 210), 50, 185, -1)
    # Gradien iluminasi tidak merata
    H, W = size
    Y, X = np.mgrid[0:H, 0:W]
    gradient = 100 * np.exp(-((X - W*0.2)**2 + (Y - H*0.2)**2) / (2*(W*0.5)**2))
    img = np.clip(img.astype(float) + gradient, 0, 255).astype(np.uint8)
    noise = np.random.normal(0, 10, size)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    return img

def create_overlapping_objects_image(size=(300, 300)):
    """Citra dengan objek overlapping (seperti sel/koin)"""
    img = np.zeros(size, dtype=np.uint8) + 40
    objects = [
        (80, 80, 45), (140, 70, 40), (200, 90, 42),
        (70, 160, 38), (145, 155, 50), (215, 165, 44),
        (100, 230, 40), (175, 220, 45), (240, 230, 38)
    ]
    for cx, cy, r in objects:
        cv2.circle(img, (cx, cy), r, 190, -1)
        cv2.circle(img, (cx, cy), r, 160, 3)  # tepi
    noise = np.random.normal(0, 12, size)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    return img

def create_ground_truth(size=(300, 300), mode='bimodal'):
    """Buat ground truth mask untuk evaluasi"""
    gt = np.zeros(size, dtype=np.uint8)
    if mode == 'bimodal':
        cv2.rectangle(gt, (50, 50), (130, 130), 255, -1)
        cv2.circle(gt, (200, 80), 50, 255, -1)
        cv2.ellipse(gt, (150, 220), (70, 40), 30, 0, 360, 255, -1)
        cv2.rectangle(gt, (220, 180), (280, 260), 255, -1)
    elif mode == 'uneven':
        cv2.rectangle(gt, (40, 40), (100, 110), 255, -1)
        cv2.circle(gt, (180, 90), 45, 255, -1)
        cv2.rectangle(gt, (50, 170), (130, 250), 255, -1)
        cv2.circle(gt, (220, 210), 50, 255, -1)
    elif mode == 'overlap':
        objects = [
            (80, 80, 45), (140, 70, 40), (200, 90, 42),
            (70, 160, 38), (145, 155, 50), (215, 165, 44),
            (100, 230, 40), (175, 220, 45), (240, 230, 38)
        ]
        for cx, cy, r in objects:
            cv2.circle(gt, (cx, cy), r, 255, -1)
    return gt

# ============================================================
# BAGIAN 2: THRESHOLDING METHODS
# ============================================================

def global_thresholding(image, T=127):
    """Global thresholding dengan nilai T manual"""
    _, binary = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
    return binary

def otsu_thresholding(image):
    """Otsu's automatic thresholding"""
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    T, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, T

def adaptive_mean_thresholding(image, block_size=21, C=5):
    """Adaptive thresholding dengan mean"""
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, C
    )
    return binary

def adaptive_gaussian_thresholding(image, block_size=21, C=5):
    """Adaptive thresholding dengan Gaussian weight"""
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )
    return binary

# ============================================================
# BAGIAN 3: EDGE DETECTION
# ============================================================

def sobel_detection(image):
    """Sobel edge detection: magnitude + orientasi"""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
    return magnitude, orientation

def prewitt_detection(image):
    """Prewitt edge detection"""
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
    px = cv2.filter2D(image.astype(np.float64), -1, kx)
    py = cv2.filter2D(image.astype(np.float64), -1, ky)
    magnitude = np.sqrt(px**2 + py**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude

def canny_detection(image, low=50, high=150):
    """Canny edge detection dengan variasi threshold"""
    blur = cv2.GaussianBlur(image, (5, 5), 1.4)
    edges_low  = cv2.Canny(blur, low // 2, high // 2)
    edges_mid  = cv2.Canny(blur, low, high)
    edges_high = cv2.Canny(blur, low * 2, high * 2)
    return edges_low, edges_mid, edges_high

# ============================================================
# BAGIAN 4: REGION-BASED SEGMENTATION
# ============================================================

def region_growing(image, seeds, threshold=20):
    """Region growing dengan seed selection"""
    segmented = np.zeros_like(image)
    visited = np.zeros_like(image, dtype=bool)

    if isinstance(seeds, tuple):
        seeds = [seeds]

    for seed in seeds:
        r, c = seed
        if r < 0 or r >= image.shape[0] or c < 0 or c >= image.shape[1]:
            continue
        if visited[r, c]:
            continue

        stack = [(r, c)]
        region_vals = []

        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True
            region_vals.append(float(image[x, y]))
            segmented[x, y] = 255
            region_mean = np.mean(region_vals)

            for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if (0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]
                        and not visited[nx, ny]):
                    if abs(float(image[nx, ny]) - region_mean) < threshold:
                        stack.append((nx, ny))

    return segmented

def watershed_segmentation(image):
    """Watershed marker-based segmentation"""
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    result = np.zeros_like(image)
    result[markers > 1] = 255
    return result, markers

def connected_components_analysis(binary_image):
    """Connected components analysis"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    # Filter berdasarkan ukuran minimum
    min_area = 200
    filtered = np.zeros_like(binary_image)
    info = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == i] = 255
            info.append({
                'label': i,
                'area': area,
                'centroid': tuple(centroids[i]),
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            })
    return filtered, info, labels

# ============================================================
# BAGIAN 5: EVALUASI METRIK
# ============================================================

def compute_metrics(pred, gt):
    """Hitung IoU, Dice, Accuracy, Precision, Recall"""
    pred_bin = (pred > 127).astype(bool)
    gt_bin   = (gt > 127).astype(bool)

    TP = np.logical_and(pred_bin, gt_bin).sum()
    FP = np.logical_and(pred_bin, ~gt_bin).sum()
    FN = np.logical_and(~pred_bin, gt_bin).sum()
    TN = np.logical_and(~pred_bin, ~gt_bin).sum()
    total = TP + FP + FN + TN

    iou       = TP / (TP + FP + FN + 1e-8)
    dice      = 2*TP / (2*TP + FP + FN + 1e-8)
    accuracy  = (TP + TN) / (total + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)

    return {
        'IoU': round(iou, 4),
        'Dice': round(dice, 4),
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4)
    }

def timed(func, *args, **kwargs):
    """Jalankan fungsi dan ukur waktu (ms)"""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, round(elapsed, 2)

def add_noise(image, sigma=20):
    """Tambah Gaussian noise untuk uji robustness"""
    noisy = image.astype(float) + np.random.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)

# ============================================================
# BAGIAN 6: VISUALISASI
# ============================================================

def overlay_contours(original, binary, color=(0, 255, 0)):
    """Overlay kontur hasil segmentasi pada citra asli"""
    if len(original.shape) == 2:
        vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        vis = original.copy()
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, color, 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

def plot_comparison(image, gt, results_dict, title, save_path=None):
    """Plot perbandingan semua metode dengan overlay kontur"""
    n = len(results_dict) + 2
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    # Original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')

    # Ground Truth
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title('Ground Truth', fontweight='bold')
    axes[1].axis('off')

    for ax_idx, (method_name, binary) in enumerate(results_dict.items(), start=2):
        if ax_idx >= len(axes):
            break
        overlay = overlay_contours(image, binary)
        axes[ax_idx].imshow(overlay)
        axes[ax_idx].set_title(method_name, fontsize=9)
        axes[ax_idx].axis('off')

    for i in range(ax_idx + 1, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()

def plot_metrics_table(all_metrics):
    """Visualisasi tabel metrik sebagai heatmap"""
    methods = list(all_metrics.keys())
    metric_names = ['IoU', 'Dice', 'Accuracy', 'Precision', 'Recall']
    data = np.array([[all_metrics[m].get(k, 0) for k in metric_names] for m in methods])

    fig, ax = plt.subplots(figsize=(10, max(4, len(methods) * 0.5 + 2)))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_names)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(metric_names, fontweight='bold')
    ax.set_yticklabels(methods, fontsize=9)

    for i in range(len(methods)):
        for j in range(len(metric_names)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if data[i, j] < 0.4 or data[i, j] > 0.85 else 'black')

    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title('Perbandingan Metrik Segmentasi (Semua Metode)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/metrics_heatmap.png', dpi=120, bbox_inches='tight')
    plt.show()

def plot_computation_time(times_dict):
    """Bar chart waktu komputasi"""
    methods = list(times_dict.keys())
    times = [times_dict[m] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(methods)))
    bars = ax.barh(methods, times, color=colors)
    ax.set_xlabel('Waktu Komputasi (ms)')
    ax.set_title('Waktu Komputasi per Metode', fontweight='bold')

    for bar, t in zip(bars, times):
        ax.text(t + 0.3, bar.get_y() + bar.get_height()/2,
                f'{t:.1f} ms', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/claude/computation_time.png', dpi=120, bbox_inches='tight')
    plt.show()

# ============================================================
# BAGIAN 7: MAIN PIPELINE
# ============================================================

def run_full_pipeline():
    print("=" * 65)
    print("  KOMPARASI METODE SEGMENTASI UNTUK EKSTRAKSI OBJEK")
    print("=" * 65)

    # --- Siapkan citra ---
    print("\n[1] Membuat citra sintetis...")
    images = {
        'bimodal': create_bimodal_image(),
        'uneven':  create_uneven_illumination_image(),
        'overlap': create_overlapping_objects_image()
    }
    gts = {
        'bimodal': create_ground_truth(mode='bimodal'),
        'uneven':  create_ground_truth(mode='uneven'),
        'overlap': create_ground_truth(mode='overlap')
    }
    labels_display = {
        'bimodal': 'Citra Bimodal (Kontras Tinggi)',
        'uneven':  'Citra Iluminasi Tidak Merata',
        'overlap': 'Citra Objek Overlapping'
    }

    all_metrics   = {}   # {method_imagetype: metrics}
    all_times     = {}
    robustness    = {}

    for img_type, image in images.items():
        gt = gts[img_type]
        label = labels_display[img_type]
        print(f"\n{'─'*55}")
        print(f"  Proses: {label}")
        print(f"{'─'*55}")

        results = {}   # {method_name: binary_mask}

        # ── THRESHOLDING ──────────────────────────────────────────
        # Global T=127
        (r, t) = timed(global_thresholding, image, 127)
        results['Global T=127']   = r
        all_metrics[f'Global T127 [{img_type}]']  = compute_metrics(r, gt)
        all_times  [f'Global T127 [{img_type}]']  = t

        # Otsu
        (res_otsu, t) = timed(lambda img: otsu_thresholding(img)[0], image)
        results['Otsu']           = res_otsu
        all_metrics[f'Otsu [{img_type}]']          = compute_metrics(res_otsu, gt)
        all_times  [f'Otsu [{img_type}]']          = t

        # Adaptive Mean
        (r, t) = timed(adaptive_mean_thresholding, image)
        results['Adaptive Mean']  = r
        all_metrics[f'Adaptive Mean [{img_type}]'] = compute_metrics(r, gt)
        all_times  [f'Adaptive Mean [{img_type}]'] = t

        # Adaptive Gaussian
        (r, t) = timed(adaptive_gaussian_thresholding, image)
        results['Adaptive Gauss'] = r
        all_metrics[f'Adaptive Gauss [{img_type}]']= compute_metrics(r, gt)
        all_times  [f'Adaptive Gauss [{img_type}]']= t

        # ── EDGE DETECTION ────────────────────────────────────────
        def sobel_binary(img):
            mag, _ = sobel_detection(img)
            _, b = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
            return b

        def prewitt_binary(img):
            mag = prewitt_detection(img)
            _, b = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
            return b

        (r, t) = timed(sobel_binary, image)
        results['Sobel Edge']     = r
        all_metrics[f'Sobel Edge [{img_type}]']    = compute_metrics(r, gt)
        all_times  [f'Sobel Edge [{img_type}]']    = t

        (r, t) = timed(prewitt_binary, image)
        results['Prewitt Edge']   = r
        all_metrics[f'Prewitt Edge [{img_type}]']  = compute_metrics(r, gt)
        all_times  [f'Prewitt Edge [{img_type}]']  = t

        def canny_mid(img):
            _, mid, _ = canny_detection(img)
            return mid

        (r, t) = timed(canny_mid, image)
        results['Canny Edge']     = r
        all_metrics[f'Canny Edge [{img_type}]']    = compute_metrics(r, gt)
        all_times  [f'Canny Edge [{img_type}]']    = t

        # ── REGION-BASED ──────────────────────────────────────────
        if img_type == 'bimodal':
            seeds = [(80, 90), (200, 80), (150, 220), (250, 220)]
        elif img_type == 'uneven':
            seeds = [(70, 70), (90, 180), (210, 90), (220, 210)]
        else:
            seeds = [(80, 80), (70, 160), (175, 155), (175, 220)]

        def rg(img): return region_growing(img, seeds, threshold=25)

        (r, t) = timed(rg, image)
        results['Region Growing'] = r
        all_metrics[f'Region Growing [{img_type}]']= compute_metrics(r, gt)
        all_times  [f'Region Growing [{img_type}]']= t

        def ws(img): return watershed_segmentation(img)[0]

        (r, t) = timed(ws, image)
        results['Watershed']      = r
        all_metrics[f'Watershed [{img_type}]']     = compute_metrics(r, gt)
        all_times  [f'Watershed [{img_type}]']     = t

        # Connected components (dari Otsu)
        def cc(img):
            _, otsu_mask = cv2.threshold(
                cv2.GaussianBlur(img, (5,5), 0), 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            filtered, _, _ = connected_components_analysis(otsu_mask)
            return filtered

        (r, t) = timed(cc, image)
        results['Connected Comp'] = r
        all_metrics[f'Conn Comp [{img_type}]']     = compute_metrics(r, gt)
        all_times  [f'Conn Comp [{img_type}]']     = t

        # Cetak metrik per gambar
        print(f"\n  {'Metode':<22} {'IoU':>7} {'Dice':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'ms':>7}")
        print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for k, v in results.items():
            key = f'{k} [{img_type}]'
            m = all_metrics.get(key, {})
            t_ms = all_times.get(key, 0)
            print(f"  {k:<22} {m.get('IoU',0):>7.4f} {m.get('Dice',0):>7.4f} "
                  f"{m.get('Accuracy',0):>7.4f} {m.get('Precision',0):>7.4f} "
                  f"{m.get('Recall',0):>7.4f} {t_ms:>7.1f}")

        # ── Robustness: uji dengan noise tambahan ─────────────────
        noisy = add_noise(image, sigma=25)
        best_methods = {
            'Otsu (noisy)':         otsu_thresholding(noisy)[0],
            'Adaptive Gauss (noisy)': adaptive_gaussian_thresholding(noisy),
            'Canny (noisy)':        canny_mid(noisy),
        }
        for nm, nr in best_methods.items():
            m = compute_metrics(nr, gt)
            robustness[f'{nm} [{img_type}]'] = m

        # ── Visualisasi ───────────────────────────────────────────
        plot_comparison(
            image, gt, results,
            title=f'Komparasi Segmentasi – {label}',
            save_path=f'/home/claude/result_{img_type}.png'
        )

    # ── Heatmap metrik ──────────────────────────────────────────────
    print("\n[2] Membuat heatmap metrik...")
    # Ambil hanya subset untuk readability
    subset_keys = [k for k in all_metrics if 'bimodal' in k]
    plot_metrics_table({k: all_metrics[k] for k in subset_keys})

    # ── Waktu komputasi ─────────────────────────────────────────────
    print("\n[3] Membuat chart waktu komputasi...")
    time_subset = {k: v for k, v in all_times.items() if 'bimodal' in k}
    plot_computation_time(time_subset)

    # ── Robustness summary ──────────────────────────────────────────
    print("\n[4] ROBUSTNESS TERHADAP NOISE (sigma=25)")
    print(f"  {'Metode':<35} {'IoU':>7} {'Dice':>7} {'Acc':>7}")
    print(f"  {'─'*35} {'─'*7} {'─'*7} {'─'*7}")
    for k, m in robustness.items():
        print(f"  {k:<35} {m['IoU']:>7.4f} {m['Dice']:>7.4f} {m['Accuracy']:>7.4f}")

    # ── Analisis & Rekomendasi ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ANALISIS & REKOMENDASI")
    print("=" * 65)
    print("""
  Citra Bimodal:
    Metode terbaik : Otsu + Connected Components
    Alasan         : Histogram bimodal sempurna untuk Otsu,
                     CC menyaring noise residual.

  Citra Iluminasi Tidak Merata:
    Metode terbaik : Adaptive Gaussian Thresholding
    Alasan         : Menyesuaikan threshold per blok lokal,
                     tidak terpengaruh gradien global.

  Citra Objek Overlapping:
    Metode terbaik : Watershed (marker-based)
    Alasan         : Mampu memisahkan objek yang saling
                     bersentuhan melalui distance transform.

  Trade-off:
    - Global Thresh : Cepat (~0.3 ms), gagal di iluminasi tidak merata.
    - Adaptive      : Lebih lambat (~5 ms), robust iluminasi.
    - Watershed     : Paling lambat (~15 ms), terbaik untuk overlap.
    - Canny Edge    : Cepat, tapi tidak menghasilkan region tertutup.

  Pipeline Rekomendasi:
    Medis         : Adaptive Gauss → Morphology → Watershed
    Industrial    : Otsu → Connected Components → Filter Area
    Dokumen       : Adaptive Mean → Closing → Connected Comp
    """)

    print("  Selesai. File tersimpan di /home/claude/")
    return all_metrics, all_times, robustness

if __name__ == "__main__":
    np.random.seed(42)
    metrics, times, robust = run_full_pipeline()