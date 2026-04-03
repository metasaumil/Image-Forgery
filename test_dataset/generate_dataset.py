"""
Generates realistic synthetic test images for the forgery detection project.

REAL images:  Natural-looking scenes (landscapes, gradients, textures)
              with consistent compression/noise throughout.

FAKE images:  Same base images but with realistic forgery artifacts:
              - Copy-move (region pasted from elsewhere in same image)
              - Splicing (patch from a different image pasted in)
              - Airbrushing/retouching (region smoothed/blurred)
              Each forgery creates detectable ELA inconsistencies.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import os, io, random

random.seed(42)
np.random.seed(42)

W, H = 400, 300
OUT  = "/home/claude/test_dataset"

# ── Helpers ─────────────────────────────────────────────────────────

def save_jpeg(img, path, quality=92):
    img.save(path, format="JPEG", quality=quality)

def add_natural_noise(arr, sigma=3):
    noise = np.random.normal(0, sigma, arr.shape)
    return np.clip(arr + noise, 0, 255).astype(np.uint8)

def gradient_bg(c1, c2, w=W, h=H):
    """Smooth gradient background."""
    arr = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        t = x / w
        arr[:, x] = [int(c1[i]*(1-t) + c2[i]*t) for i in range(3)]
    return arr

def radial_gradient(cx, cy, color, bg, w=W, h=H):
    arr = np.array(bg, dtype=np.float32)
    for y in range(h):
        for x in range(0, w, 1):
            dist = ((x-cx)**2 + (y-cy)**2) ** 0.5
            t    = max(0, 1 - dist/180)
            arr[y, x] = arr[y, x]*(1-t) + np.array(color)*t
    return np.clip(arr, 0, 255).astype(np.uint8)

def draw_scene_1():
    """Sky-like gradient with a simple sun."""
    sky  = gradient_bg((30, 80, 180), (100, 180, 255))
    sky  = radial_gradient(300, 60, (255, 240, 100), sky)
    # Ground
    for y in range(200, H):
        t = (y-200)/(H-200)
        sky[y] = [int(50*(1-t)+20*t), int(140*(1-t)+80*t), int(40*(1-t)+20*t)]
    return add_natural_noise(sky, sigma=4)

def draw_scene_2():
    """Indoor warm room (orange/wood tones)."""
    arr = gradient_bg((210, 140, 80), (180, 110, 60))
    # "Wall" panel
    arr[50:220, 80:320] = [190, 130, 75]
    # "Window" bright patch
    arr[70:180, 100:220] = [220, 215, 200]
    return add_natural_noise(arr, sigma=5)

def draw_scene_3():
    """Night city (dark blues, bright dots)."""
    arr = gradient_bg((10, 10, 40), (20, 20, 70))
    # "Buildings"
    for bx in range(0, W, 50):
        bh = random.randint(60, 160)
        arr[H-bh:H, bx:bx+40] = [random.randint(30,60)]*3
        # Windows
        for wy in range(H-bh+10, H-10, 20):
            for wx in range(bx+5, bx+35, 10):
                if random.random() > 0.3:
                    arr[wy:wy+8, wx:wx+6] = [220, 200, 120]
    # Stars
    for _ in range(80):
        sx, sy = random.randint(0,W-1), random.randint(0,80)
        arr[sy,sx] = [255,255,200]
    return add_natural_noise(arr, sigma=3)

def draw_scene_4():
    """Portrait-like — face region placeholder."""
    arr = gradient_bg((200, 180, 160), (180, 155, 130))
    # Skin oval
    img = Image.fromarray(arr)
    d   = ImageDraw.Draw(img)
    d.ellipse([130, 40, 270, 200], fill=(210, 175, 140))   # face
    d.ellipse([160, 70, 240, 130], fill=(60, 40, 30))       # eyes area
    d.rectangle([150, 160, 250, 180], fill=(190, 100, 90))  # mouth
    arr = np.array(img)
    return add_natural_noise(arr, sigma=6)

def draw_scene_5():
    """Texture — wood grain pattern."""
    arr = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        base_color = [140 + int(30*np.sin(y/8)), 90 + int(20*np.sin(y/6)), 50]
        arr[y] = base_color
    # Grain lines
    for _ in range(30):
        lx = random.randint(0, W)
        for y in range(H):
            jitter = int(8*np.sin(y/20 + lx))
            x = min(W-1, max(0, lx + jitter))
            arr[y, max(0,x-1):x+2] = [80, 50, 25]
    return add_natural_noise(arr, sigma=3)

SCENES = [draw_scene_1, draw_scene_2, draw_scene_3, draw_scene_4, draw_scene_5]

# ── Generate REAL images ─────────────────────────────────────────────

print("Generating REAL images...")
real_dir = f"{OUT}/real"
real_images = []

for scene_fn in SCENES:
    for variant in range(6):  # 6 variants per scene = 30 real
        arr = scene_fn()
        # Small random crop/shift for variety
        dy, dx = random.randint(0,10), random.randint(0,10)
        arr = arr[dy:dy+H-10, dx:dx+W-10]
        arr = np.array(Image.fromarray(arr).resize((W, H)))
        # Random brightness
        img  = Image.fromarray(arr)
        img  = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        img  = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        name = f"real_{scene_fn.__name__}_{variant:02d}.jpg"
        path = os.path.join(real_dir, name)
        save_jpeg(img, path, quality=random.randint(85, 97))
        real_images.append(np.array(img))
        print(f"  {name}")

# ── Generate FAKE images ─────────────────────────────────────────────

print("\nGenerating FAKE images (with forgeries)...")
fake_dir = f"{OUT}/fake"

def copy_move_forgery(arr):
    """Copy a region and paste it elsewhere — classic copy-move."""
    img = Image.fromarray(arr.copy())
    # Source region
    sx, sy = random.randint(20, 150), random.randint(20, 100)
    sw, sh = random.randint(60, 120), random.randint(50, 100)
    patch  = img.crop((sx, sy, sx+sw, sy+sh))
    # Destination (far from source)
    dx = (sx + random.randint(130, 200)) % (W - sw)
    dy = (sy + random.randint(80,  150)) % (H - sh)
    # Paste — hard edge creates ELA artifact
    img.paste(patch, (dx, dy))
    return np.array(img)

def splicing_forgery(arr, donor_arr):
    """Paste a region from a completely different image."""
    img   = Image.fromarray(arr.copy())
    donor = Image.fromarray(donor_arr)
    # Take patch from donor
    px, py = random.randint(50, 150), random.randint(40, 120)
    pw, ph = random.randint(80, 140), random.randint(60, 110)
    patch  = donor.crop((px, py, px+pw, py+ph))
    # Slightly adjust colors to make it subtler
    patch  = ImageEnhance.Color(patch).enhance(random.uniform(0.8, 1.2))
    # Paste into target — mismatched compression = ELA spike
    tx = random.randint(10, W-pw-10)
    ty = random.randint(10, H-ph-10)
    img.paste(patch, (tx, ty))
    return np.array(img)

def retouch_forgery(arr):
    """Airbrush/smooth a region — like skin retouching or object removal."""
    img = Image.fromarray(arr.copy())
    rx, ry = random.randint(80, 200), random.randint(60, 160)
    rw, rh = random.randint(70, 130), random.randint(60, 100)
    region = img.crop((rx, ry, rx+rw, ry+rh))
    # Heavy blur = retouching artifact
    smoothed = region.filter(ImageFilter.GaussianBlur(radius=random.randint(5, 12)))
    img.paste(smoothed, (rx, ry))
    return np.array(img)

def double_save_forgery(arr):
    """Re-save at low quality first — simulates editing + resaving cycle."""
    # First save at very low quality
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=random.randint(45, 65))
    buf.seek(0)
    degraded = np.array(Image.open(buf).convert("RGB"))
    # Then apply copy-move on top
    return copy_move_forgery(degraded)

FORGERIES = [copy_move_forgery, splicing_forgery, retouch_forgery, double_save_forgery]

forged_count = 0
for i, arr in enumerate(real_images):
    # Pick a random forgery type
    forgery_fn = FORGERIES[i % len(FORGERIES)]
    
    if forgery_fn == splicing_forgery:
        # Use a different scene as donor
        donor_fn = SCENES[(i + 2) % len(SCENES)]
        forged   = splicing_forgery(arr, donor_fn())
    else:
        forged = forgery_fn(arr)

    name = f"fake_{forgery_fn.__name__}_{i:02d}.jpg"
    path = os.path.join(fake_dir, name)
    # Save at slightly different quality to compound ELA artifacts
    save_jpeg(Image.fromarray(forged), path, quality=random.randint(80, 94))
    forged_count += 1
    print(f"  {name} [{forgery_fn.__name__}]")

# ── Demo comparison pairs ────────────────────────────────────────────

print("\nGenerating side-by-side DEMO pairs...")
demo_dir = f"{OUT}/demo"

for i in range(5):
    base_arr   = SCENES[i]()
    base_img   = Image.fromarray(base_arr)

    # Forged version
    forgery_fn = FORGERIES[i % len(FORGERIES)]
    if forgery_fn == splicing_forgery:
        forged_arr = splicing_forgery(base_arr, SCENES[(i+1)%len(SCENES)]())
    else:
        forged_arr = forgery_fn(base_arr)
    forged_img = Image.fromarray(forged_arr)

    # Save individual
    base_path   = os.path.join(demo_dir, f"demo_{i+1}_REAL.jpg")
    forged_path = os.path.join(demo_dir, f"demo_{i+1}_FAKE_{forgery_fn.__name__}.jpg")
    save_jpeg(base_img,   base_path,   quality=93)
    save_jpeg(forged_img, forged_path, quality=88)

    # Save side-by-side comparison
    combined = Image.new("RGB", (W*2 + 10, H + 40), (240, 240, 240))
    combined.paste(base_img,   (0, 40))
    combined.paste(forged_img, (W+10, 40))
    d = ImageDraw.Draw(combined)
    d.rectangle([0, 0, W, 35],    fill=(100, 180, 100))
    d.rectangle([W+10, 0, W*2+10, 35], fill=(220, 80, 80))
    d.text((W//2-20, 8),    "REAL",   fill="white")
    d.text((W+W//2-10, 8),  "FAKE",   fill="white")
    combined.save(os.path.join(demo_dir, f"demo_{i+1}_comparison.jpg"), quality=92)
    print(f"  Pair {i+1}: REAL vs {forgery_fn.__name__}")

# ── Summary ──────────────────────────────────────────────────────────

real_n  = len(os.listdir(real_dir))
fake_n  = len(os.listdir(fake_dir))
demo_n  = len(os.listdir(demo_dir))

print(f"""
{'='*50}
  DATASET GENERATED SUCCESSFULLY
{'='*50}
  Real images:   {real_n}  → test_dataset/real/
  Fake images:   {fake_n}  → test_dataset/fake/
  Demo pairs:    {demo_n} → test_dataset/demo/
  Total:         {real_n + fake_n + demo_n} files

  Forgery types included:
    • Copy-move  (region copied within same image)
    • Splicing   (patch from different scene pasted in)
    • Retouching (region heavily blurred/airbrushed)
    • Double-save (low-quality resave + copy-move)

  To test ELA on a fake image:
    python classical_methods/ela.py test_dataset/fake/fake_copy_move_forgery_00.jpg

  To run full inference:
    python inference.py test_dataset/demo/demo_1_FAKE_copy_move_forgery.jpg
{'='*50}
""")
