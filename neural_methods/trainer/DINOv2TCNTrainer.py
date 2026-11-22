import torch
import torch.optim as optim

from neural_methods.model.dino2 import DINOv2TCN
from neural_methods.loss.TorchLossComputer import Hybrid_Loss
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.trainer.SSMambaTrainer import SSMambaTrainer


class DINOv2TCNTrainer(SSMambaTrainer):
    """Trainer using DINOv2TCN with the existing rPPG training workflow."""

    def __init__(self, config, data_loader):
        BaseTrainer.__init__(self)
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = int(config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized")

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = DINOv2TCN(input_format="rgb255",
                                    freeze_backbone=False,
                                    checkpoint_backbone=True,
                                    frame_chunk_size=96,).to(self.device)

            # Freeze all but the last two transformer blocks of the backbone and the normalization layer.
            self.model.backbone.requires_grad_(False)
            for block in list(self.model.backbone.blocks)[-2:]:
                block.requires_grad_(True)
                # Unfreeze the normalization layer in the last block.
            self.model.backbone.norm.requires_grad_(True)

            # Separate trainable parameters
            backbone_params = [
                p for p in self.model.backbone.parameters()
                if p.requires_grad
            ]

            head_params = [
                p for name, p in self.model.named_parameters()
                if p.requires_grad and not name.startswith("backbone.")
            ]
# -------------------above are changes--------------------------
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
            )
            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()   #Hybrid_Loss()
            # self.optimizer = optim.AdamW(
            #     self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0
            # )

            head_lr = config.TRAIN.LR
            backbone_lr = head_lr * 0.1

            self.optimizer = optim.AdamW(
                        [
                            {
                                "params": head_params,
                                "lr": head_lr,
                                "weight_decay": 0.0,
                            },
                            {
                                "params": backbone_params,
                                "lr": backbone_lr,
                                "weight_decay": 0.0,
                            },
                        ]
                    )
            # Learning rate scheduler: OneCycleLR
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches,
            )
        elif config.TOOLBOX_MODE == "only_test":
            # self.model = DINOv2TCN().to(self.device)
            self.model = DINOv2TCN(
                input_format="rgb255",
                freeze_backbone=False,
                checkpoint_backbone=False,
                frame_chunk_size=96,
            ).to(self.device)

            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
            )
        else:
            raise ValueError("DINOv2TCN trainer initialized in incorrect toolbox mode!")
