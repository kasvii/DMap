# Step 1: synthesize the normal for the invisible back.
python ./step1_back_normal_guide.py 

# Step 2: predict the UV coordinates and depth from the normal.
python ./step2_uv_mapping_FB.py 

# Step 3: fit ISP to the incomplete mask to recover the complete mask/rest garment.
python ./step3_isp_seq_one_mask.py  

# Step 4: fit the diffusion prior to the incomplete UV positional maps to recover the complete UV maps/deformed garment.
python ./step4_uv_inpainting.py

# Step 5: refine the recovered garment mesh using a set of observations and constraints (normal, mask, points, physical energy, temporal consistency, etc.).
python ./step5_post_opt.py