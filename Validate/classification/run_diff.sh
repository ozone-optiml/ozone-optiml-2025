CUDA_VISIBLE_DEVICES=1 python eval_classification_pyshp_diff.py --model_name "diff_gamma3_beta1e1_p1/epoch99_loss0.00010_rmse0.01010.pth" &
CUDA_VISIBLE_DEVICES=2 python eval_classification_pyshp_diff.py --model_name "diff_gamma3_beta1e2_p1/epoch98_loss0.00008_rmse0.00916.pth" &
CUDA_VISIBLE_DEVICES=3 python eval_classification_pyshp_diff.py --model_name "diff_gamma3_beta1e3_p1/epoch99_loss0.00009_rmse0.00924.pth" &
CUDA_VISIBLE_DEVICES=4 python eval_classification_pyshp_diff.py --model_name "diff_gamma3_beta1e4_p1/epoch98_loss0.00008_rmse0.00920.pth" &
CUDA_VISIBLE_DEVICES=5 python eval_classification_pyshp_diff.py --model_name "diff_gamma3_beta1e3_p2/epoch99_loss0.00011_rmse0.01047.pth" &
CUDA_VISIBLE_DEVICES=6 python eval_classification_pyshp_diff.py --model_name "diff_gamma3_beta1e4_p2/epoch96_loss0.00009_rmse0.00931.pth"
CUDA_VISIBLE_DEVICES=6 python eval_classification_pyshp_diff.py --model_name "diff_mse/epoch96_loss0.00009_rmse0.00922.pth"
CUDA_VISIBLE_DEVICES=6 python eval_classification_pyshp_diff.py --model_name "diff_mse/epoch96_loss0.00009_rmse0.00922.pth" --reduction 1