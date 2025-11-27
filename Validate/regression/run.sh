#!/bin/bash
# python eval_forecast.py
# python eval_all.py --model_dir "../Train/checkpoints/5. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6 (new emb)" --model_name "epoch99_loss0.00010_rmse0.00982.pth"
# python eval_all.py --model_dir "../Train/checkpoints/6. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed)" --model_name "epoch99_loss0.00000_rmse0.00066.pth"
# python eval_all.py --model_dir "../Train/checkpoints/6.2. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed) gamma=3" --model_name "epoch99_loss0.00000_rmse0.00066.pth"


# python eval_by_hour.py --model_dir "../../Train/checkpoints/diff_gamma3_beta1e2_p1" --diff True
# python eval_by_hour.py --model_dir "../../Train/checkpoints/diff_mse" --diff True

# python save_pred_output.py --model_dir "../../Train/checkpoints/6.3. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed) gamma=3 beta=1e4" --diff False
# python save_pred_output.py --model_dir "../../Train/checkpoints/diff_gamma3_beta1e2_p1" --diff True

python eval_all.py --model_dir "../../Train/checkpoints/diff_mse" --split test --diff True --model_name "epoch96_loss0.00009_rmse0.00922.pth" --reduction 1 --save_outputs True --results_dir "results_red1" --split "test" --batch_size 8
# python eval_all.py --model_dir "../../Train/checkpoints/diff_mse" --split test --diff True --model_name "epoch96_loss0.00009_rmse0.00922.pth"