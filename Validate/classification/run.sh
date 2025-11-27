python eval_classification_pyshp.py --model_name "5. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6 (new emb)/epoch99_loss0.00010_rmse0.00982.pth"
python eval_classification_pyshp.py --model_name "6. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed)/epoch99_loss0.00000_rmse0.00066.pth"
python eval_classification_pyshp.py --model_name "6.2. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed) gamma=3/epoch99_loss0.00000_rmse0.00003.pth"
python eval_classification_pyshp.py --model_name "6.3. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed) gamma=3 beta=1e4/epoch99_loss0.00672_rmse0.08198.pth"

