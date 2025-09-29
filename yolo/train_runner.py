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
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--copy_paste", type=float, default=0.0)
    parser.add_argument("--mask_ratio", type=int, default=4)
    args = parser.parse_args()

    # Use the unified training function which automatically selects the right pipeline
    from yolo.yolo_train import start_training
    start_training(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        resume=args.resume,
        multi_scale=args.multi_scale,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        momentum=args.momentum,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        box=args.box,
        dropout=args.dropout,
        copy_paste=args.copy_paste,
        mask_ratio=args.mask_ratio,
        project=args.project,
        name=args.experiment,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()