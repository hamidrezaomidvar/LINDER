import os

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from eolearn.core import (
    EOExecutor,
    EOPatch,
    EOTask,
    FeatureType,
    LinearWorkflow,
    LoadFromDisk,
    OverwritePermission,
    SaveToDisk,
)
from eolearn.io import ExportToTiff

from .sent_util import LULC, ConcatenateData, PredictPatch
from ._env import path_module

from pathlib import Path


def predict_image_one(path_out, patch_n, scale):
    path_out = Path(path_out)
    model_path = path_module / "model.pkl"
    model = joblib.load(model_path)

    cnt = "n"
    while cnt == "n":

        pic_n = int(input("Please choose the desired image number: "))
        # TASK TO LOAD EXISTING EOPATCHES
        load = LoadFromDisk(path_out)

        # TASK FOR CONCATENATION
        concatenate = ConcatenateData("FEATURES", ["BANDS", "NDVI", "NDWI", "NORM"])
        # concatenate = ConcatenateData('FEATURES', ['BANDS'])
        # TASK FOR FILTERING OUT TOO CLOUDY SCENES
        # keep frames with > 80 % valid coverage
        # valid_data_predicate = ValidDataFractionPredicate(0.8)
        # filter_task = SimpleFilterTask((FeatureType.MASK, 'IS_VALID'), valid_data_predicate)

        save = SaveToDisk(
            path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH
        )

        workflow = LinearWorkflow(
            load,
            concatenate,
            #    filter_task,
            #    linear_interp,
            #     erosion,
            #     spatial_sampling,
            save,
        )

        execution_args = []
        for idx in range(0, 1):
            execution_args.append(
                {
                    load: {"eopatch_folder": f"eopatch_{patch_n}"},
                    save: {"eopatch_folder": f"eopatch_{patch_n}"},
                }
            )

        print("Saving the features . . .")
        executor = EOExecutor(workflow, execution_args, save_logs=False)
        executor.run(workers=5, multiprocess=False)

        executor.make_report()

        # TASK TO LOAD EXISTING EOPATCHES
        load = LoadFromDisk(path_out)

        # TASK FOR PREDICTION
        predict = PredictPatch(
            model, (FeatureType.DATA, "FEATURES"), "LBL", pic_n, "SCR"
        )

        # TASK FOR SAVING
        # path_out_sampled = './eopatches_sampled_small_'+cname+'_2/' if use_smaller_patches else './eopatches_sampled_large/'
        # if not os.path.isdir(path_out_sampled):
        #    os.makedirs(path_out_sampled)
        save = SaveToDisk(
            str(path_out), overwrite_permission=OverwritePermission.OVERWRITE_PATCH
        )

        # TASK TO EXPORT TIFF
        export_tiff = ExportToTiff((FeatureType.MASK_TIMELESS, "LBL"))
        tiff_location = path_out / f"predicted_tiff" / f"patch{patch_n}"

        if not os.path.isdir(tiff_location):
            os.makedirs(tiff_location)

        workflow = LinearWorkflow(load, predict, export_tiff, save)

        # create a list of execution arguments for each patch
        execution_args = []
        for i in range(0, 1):
            execution_args.append(
                {
                    load: {"eopatch_folder": f"eopatch_{patch_n}"},
                    export_tiff: {
                        "filename": tiff_location / f"prediction_eopatch_{i}.tiff"
                    },
                    save: {"eopatch_folder": f"eopatch_{patch_n}"},
                }
            )

        # run the executor on 2 cores
        executor = EOExecutor(workflow, execution_args)

        # uncomment below save the logs in the current directory and produce a report!
        # executor = EOExecutor(workflow, execution_args, save_logs=True)
        print("Predicting the land cover . . .")
        executor.run(workers=5, multiprocess=False)
        executor.make_report()

        PATH = path_out / "predicted_tiff" / f"patch{patch_n}"
        if os.path.exists(PATH / "merged_prediction.tiff"):
            os.remove(PATH / "merged_prediction.tiff")
        cmd = (
            "gdal_merge.py -o "
            + str(PATH)
            + "/merged_prediction.tiff -co compress=LZW "
            + str(PATH)
            + "/prediction_eopatch_*"
        )
        os.system(cmd)

        # Reference colormap things
        lulc_cmap = mpl.colors.ListedColormap([entry.color for entry in LULC])
        lulc_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 3, 1), lulc_cmap.N)

        size = 20
        fig, ax = plt.subplots(
            figsize=(2 * size * 1, 1 * size * scale), nrows=1, ncols=2
        )
        eopatch = EOPatch.load(path_out / f"eopatch_{patch_n}", lazy_loading=True)
        im = ax[0].imshow(
            eopatch.mask_timeless["LBL"].squeeze(), cmap=lulc_cmap, norm=lulc_norm
        )
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_aspect("auto")

        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(0, 1):
            eopatch = EOPatch.load(path_out / f"eopatch_{patch_n}", lazy_loading=True)
            ax = ax[1]
            plt.imshow(
                np.clip(
                    eopatch.data["BANDS"][pic_n, :, :, :][..., [2, 1, 0]] * 3.5, 0, 1
                )
            )
            plt.xticks([])
            plt.yticks([])
            ax.set_aspect("auto")
            del eopatch

        print("saving the predicted image . . .")
        plt.savefig(path_out / f"predicted_vs_real_{patch_n}.png")

        cnt = input(
            "Is the first stage predicted image good for patch "
            + str(patch_n)
            + "? (y/n):"
        )
        while cnt not in ["y", "n"]:
            cnt = input(
                "Is the first stage predicted image good for patch "
                + str(patch_n)
                + "? (y/n):"
            )


def predict_image_all(path_out, patch_n, scale):
    path_out = Path(path_out)
    model_path = path_module / "model.pkl"
    model = joblib.load(model_path)

    eopatch = EOPatch.load(path_out / f"eopatch_{patch_n}", lazy_loading=True)

    n_pics = eopatch.data["BANDS"].shape[0]

    # cnt = "n"
    # while cnt == "n":

    # pic_n = int(input("Please choose the desired image number: "))
    for pic_n in range(n_pics):
        # TASK TO LOAD EXISTING EOPATCHES
        load = LoadFromDisk(path_out)

        # TASK FOR CONCATENATION
        concatenate = ConcatenateData("FEATURES", ["BANDS", "NDVI", "NDWI", "NORM"])
        # concatenate = ConcatenateData('FEATURES', ['BANDS'])
        # TASK FOR FILTERING OUT TOO CLOUDY SCENES
        # keep frames with > 80 % valid coverage
        # valid_data_predicate = ValidDataFractionPredicate(0.8)
        # filter_task = SimpleFilterTask((FeatureType.MASK, 'IS_VALID'), valid_data_predicate)

        save = SaveToDisk(
            path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH
        )

        workflow = LinearWorkflow(
            load,
            concatenate,
            #    filter_task,
            #    linear_interp,
            #     erosion,
            #     spatial_sampling,
            save,
        )

        execution_args = []
        for idx in range(0, 1):
            execution_args.append(
                {
                    load: {"eopatch_folder": f"eopatch_{patch_n}"},
                    save: {"eopatch_folder": f"eopatch_{patch_n}"},
                }
            )

        print("Saving the features . . .")
        executor = EOExecutor(workflow, execution_args, save_logs=False)
        executor.run(workers=5, multiprocess=False)

        executor.make_report()

        # TASK TO LOAD EXISTING EOPATCHES
        load = LoadFromDisk(path_out)

        # TASK FOR PREDICTION
        predict = PredictPatch(
            model, (FeatureType.DATA, "FEATURES"), "LBL", pic_n, "SCR"
        )

        # TASK FOR SAVING
        # path_out_sampled = './eopatches_sampled_small_'+cname+'_2/' if use_smaller_patches else './eopatches_sampled_large/'
        # if not os.path.isdir(path_out_sampled):
        #    os.makedirs(path_out_sampled)
        save = SaveToDisk(
            str(path_out), overwrite_permission=OverwritePermission.OVERWRITE_PATCH
        )

        # TASK TO EXPORT TIFF
        export_tiff = ExportToTiff((FeatureType.MASK_TIMELESS, "LBL"))
        tiff_location = (
            path_out / f"predicted_tiff" / f"patch{patch_n}" / f"picture{pic_n}"
        )

        if not os.path.isdir(tiff_location):
            os.makedirs(tiff_location)

        workflow = LinearWorkflow(load, predict, export_tiff, save)

        # create a list of execution arguments for each patch
        execution_args = []
        for i in range(0, 1):
            execution_args.append(
                {
                    load: {"eopatch_folder": f"eopatch_{patch_n}"},
                    export_tiff: {
                        "filename": tiff_location / f"prediction_eopatch_{i}.tiff"
                    },
                    save: {"eopatch_folder": f"eopatch_{patch_n}"},
                }
            )

        # run the executor on 2 cores
        executor = EOExecutor(workflow, execution_args)

        # uncomment below save the logs in the current directory and produce a report!
        # executor = EOExecutor(workflow, execution_args, save_logs=True)
        print("Predicting the land cover . . .")
        executor.run(workers=5, multiprocess=False)
        executor.make_report()

        # PATH = path_out / "predicted_tiff" / f"patch{patch_n}"
        if os.path.exists(tiff_location / "merged_prediction.tiff"):
            os.remove(tiff_location / "merged_prediction.tiff")
        cmd = (
            "gdal_merge.py -o "
            + str(tiff_location)
            + "/merged_prediction.tiff -co compress=LZW "
            + str(tiff_location)
            + "/prediction_eopatch_*"
        )
        os.system(cmd)

        # Reference colormap things
        lulc_cmap = mpl.colors.ListedColormap([entry.color for entry in LULC])
        lulc_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 3, 1), lulc_cmap.N)

        size = 20
        fig, ax = plt.subplots(
            figsize=(2 * size * 1, 1 * size * scale), nrows=1, ncols=2
        )
        eopatch = EOPatch.load(path_out / f"eopatch_{patch_n}", lazy_loading=True)
        im = ax[0].imshow(
            eopatch.mask_timeless["LBL"].squeeze(), cmap=lulc_cmap, norm=lulc_norm
        )
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_aspect("auto")

        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(0, 1):
            eopatch = EOPatch.load(path_out / f"eopatch_{patch_n}", lazy_loading=True)
            ax = ax[1]
            plt.imshow(
                np.clip(
                    eopatch.data["BANDS"][pic_n, :, :, :][..., [2, 1, 0]] * 3.5, 0, 1
                )
            )
            plt.xticks([])
            plt.yticks([])
            ax.set_aspect("auto")
            del eopatch

        print("saving the predicted image . . .")
        plt.savefig(path_out / f"predicted_vs_real_{patch_n}-{pic_n}.png")

    # cnt = input(
    #     "Is the first stage predicted image good for patch "
    #     + str(patch_n)
    #     + "? (y/n):"
    # )
    # while cnt not in ["y", "n"]:
    #     cnt = input(
    #         "Is the first stage predicted image good for patch "
    #         + str(patch_n)
    #         + "? (y/n):"
    #     )
