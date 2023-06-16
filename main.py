import argparse

from utils.logger import get_logger
from utils.arg_parser import Argments
from loader.data_loader import (
    GeneralDataLoaderCls,
    NbsDataLoaderCls,
    GeneralDataLoaderSeg,
    NbsDataLoaderSeg,
)
from runners.cnn_runner import CnnRunner
from runners.nbs_runner import NbsRunner
from runners.mcd_runner import McdRunner

from pathlib import Path
import wandb


argparser = argparse.ArgumentParser()
argparser.add_argument("yaml")
argparser.add_argument("--local_rank", default=0, type=int)
cmd_args = argparser.parse_args()


def main():
    arg = Argments(f"scripts/{cmd_args.yaml}.yaml", cmd_args)
    setup = arg["setup"]
    model_path = arg["path/model_path"]
    infer_path = arg["path/infer_path"]
    logger = get_logger(f"{model_path}/log.txt")
    setup['rank'] = 0
    if setup["rank"] == 0:
        logger.info(arg)

    model_type = setup["model_type"]
    dataset = arg["path/dataset"]

    if "nbs" in model_type:
        _data_loader = NbsDataLoaderCls
        
        
        data_loader = _data_loader(
            dataset, setup["batch_size"], setup["n_a"], setup["cpus"], setup["seed"]
        )
        runner = NbsRunner(
            data_loader,
            **arg.module,
            num_epoch=setup["num_epoch"],
            logger=logger,
            model_path=model_path,
            infer_path=infer_path,
            rank=setup["rank"],
            epoch_th=setup["epoch_th"],
            num_mc=setup["num_mc"],
        )

    else:
        _data_loader = GeneralDataLoaderCls
        data_loader = _data_loader(
            dataset, setup["batch_size"], setup["cpus"], setup["seed"]
        )

        runner = CnnRunner(
            data_loader,
            **arg.module,
            num_epoch=setup["num_epoch"],
            logger=logger,
            model_path=model_path,
            infer_path=infer_path,
            rank=setup["rank"],
        )

    if setup["phase"] == "train":
        runner.train()
        runner.test()
    elif setup["phase"] == "test":
        runner.test()


if __name__ == "__main__":
    main()