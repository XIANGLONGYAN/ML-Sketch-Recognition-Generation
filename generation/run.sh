# The pretrained model of 17-category are provided in the './model_save/' fold. You can train your own dataset as follows.
# Encoder and Decoder. We recommend using more categories.
CUDA_VISIBLE_DEVICES=python -u Train.py
# Noise predicetor U-Net.
python -u Diffusion_Train.py


# Visualize the groudtruth sketches to './GroundTruth/' fold.
python -u DrawAll.py
# Reconstruct sketches with original stroke locations.
python -u Inference.py
# Reconstruct sketches with generated stroke locations.
python -u Diffusion_Inference.py


# Calculate metrics.
cd evaluations
python CLIP_score.py ../results/ ../GroundTruth/ --real_flag img --fake_flag img --device cuda
python fid_score.py ../results/ ../GroundTruth/ --gpu 0
python lpips_score.py --path1 ../results/ --path2 ../GroundTruth/
python CLIP_score.py ../diffusion_results/ ../GroundTruth/ --real_flag img --fake_flag img --device cuda
python fid_score.py ../diffusion_results/ ../GroundTruth/ --gpu 0
python lpips_score.py --path1 ../diffusion_results/ --path2 ../GroundTruth/

# Record the strokes and their embeddings to './stroke/' fold.
python -u Dataset.py
python -u save_embedding.py

# Replace strokes, inteplorate between strokes, and add strokes.
# First, select the to be edited sketch and referenced sketch from the './GroundTruth/' fold and record their id, e.g. 7886 (the 386th sketches in the test set of 'angel') and 8196.
# Second, according to the shape of strokes, select the corresponding id of to be edited strokes and referenced strokes from the './stroke/' fold.
# Third, record the selected strokes' id, e.g. the 3rd stroke of 7886 and the 4th stroke of 8196. If you want to add a stroke, trying to select the first padding stroke as to be edited stroke, e.g. 5th stroke of 7886.
# Finally, modify the parameters in "Replace.py" and run the code. You will find the creative sketches in './sample_tmp/' fold.

python -u Replace.py

The parameters in "Replace.py" are followings:
    sketch_idx = 7886  # to be edited sketch
    sketch_stroke_idx = [3] # to be edited strokes
    template_idx =8196 # referenced sketch
    template_stroke_idx = [4] # referenced strokes from referenced sketch