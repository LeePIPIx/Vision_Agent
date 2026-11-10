"""Isolated nnU-Net v1 runner for official Task009_Spleen, one fold, no TTA."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Only the pinned official model is loaded, never a user-uploaded checkpoint.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ.setdefault("nnUNet_n_proc_DA", "1")
    from nnunet.inference.predict import predict_from_folder
    predict_from_folder(args.model, args.input, args.output, folds=[0], save_npz=False,
                        num_threads_preprocessing=1, num_threads_nifti_save=1,
                        lowres_segmentations=None, part_id=0, num_parts=1, tta=False,
                        mixed_precision=False, overwrite_existing=True, mode="normal",
                        overwrite_all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint")


if __name__ == "__main__":
    main()
