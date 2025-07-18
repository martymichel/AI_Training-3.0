import argparse
from yolo.yolo_train import start_training


def main():
    parser = argparse.ArgumentParser(description="Run YOLO training")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=float, required=True)
    parser.add_argument("--lr0", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--cos_lr", action="store_true")
    parser.add_argument("--close_mosaic", type=int, required=True)
    parser.add_argument("--momentum", type=float, required=True)
    parser.add_argument("--warmup_epochs", type=int, required=True)
    parser.add_argument("--warmup_momentum", type=float, required=True)
    parser.add_argument("--box", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    start_training(
        args.data,
        args.epochs,
        args.imgsz,
        args.batch,
        args.lr0,
        args.resume,
        args.multi_scale,
        args.cos_lr,
        args.close_mosaic,
        args.momentum,
        args.warmup_epochs,
        args.warmup_momentum,
        args.box,
        args.dropout,
        args.project,
        args.experiment,
    )


if __name__ == "__main__":
    main()