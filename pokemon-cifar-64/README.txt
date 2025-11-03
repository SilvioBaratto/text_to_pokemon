Pokemon CIFAR Dataset (BALANCED with Stratified Sampling)
==============================================================

This dataset contains 898 Pokemon classes in CIFAR format.
Uses STRATIFIED SAMPLING to ensure balanced distributions across splits.

Dataset Structure:
------------------
- Training batches: data_batch_1 to data_batch_5 (includes train + val)
- Test batch: test_batch (balanced to match train distribution)
- Metadata: batches.meta

Balancing Method:
-----------------
✅ Multi-attribute balancing: Equal samples per Type × Color × Body Shape combination
✅ Oversampling rare combinations (e.g., Flying + Orange + Serpentine)
✅ Undersampling common combinations (e.g., Water + Blue + Aquatic)
✅ Stratified train/val/test split maintains balanced distributions
✅ No distribution shift between splits
✅ Expected test loss close to validation loss (~80-100 L1)

Total Images:
-------------
- Training + Val: 2592 images
- Test: 458 images
- Total: 3050 images

Pokemon Classes: 898
