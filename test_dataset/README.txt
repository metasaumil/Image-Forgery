IMAGE FORGERY DETECTION — TEST DATASET
=======================================

75 images total, ready to use with your forgery detection project.

STRUCTURE
---------
real/   — 30 authentic images (5 scene types × 6 variants)
fake/   — 30 forged images (4 forgery types)
demo/   — 15 files = 5 REAL + 5 FAKE + 5 side-by-side comparisons

FORGERY TYPES IN fake/
-----------------------
copy_move   — A region is copied and pasted elsewhere in the same image
splicing    — A patch from a completely different scene is pasted in
retouch     — A region is heavily blurred (like airbrushing/smoothing)
double_save — Image re-saved at low quality first, then copy-moved

HOW TO USE
----------
1. ELA analysis on a fake image:
   python classical_methods/ela.py test_dataset/fake/fake_copy_move_forgery_00.jpg

2. ELA analysis on a real image (should show low score):
   python classical_methods/ela.py test_dataset/real/real_draw_scene_1_00.jpg

3. Copy-move detection:
   python classical_methods/copy_move.py test_dataset/fake/fake_copy_move_forgery_00.jpg

4. Full inference pipeline:
   python inference.py test_dataset/demo/demo_1_FAKE_copy_move_forgery.jpg

5. Use as training data:
   Organize into data/processed/train|val|test/real|fake/
   Then run: python training/train.py --data_dir data/processed

WHAT TO EXPECT
--------------
- ELA maps: fake images will show bright/noisy patches at forged regions
- Real images: ELA maps should be uniformly dark/low
- Copy-move detector: should find matches in copy_move fakes
- Demo comparisons: visually compare real vs fake side-by-side
