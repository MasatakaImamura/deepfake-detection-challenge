# Version 0
# BaseModel: Efficientnet-b0 - Resnet3D
# Optimizer: Adam
# Scheduler: StepLR(step_size=5, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# img_num = 15
# batch_size = 4
# img_size = 120
# epoch = 100


# Version 1
# BaseModel: Facenet - Resnet3D
# Optimizer: Adam
# Scheduler: StepLR(step_size=5, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# img_num = 15
# batch_size = 4
# img_size = 120
# epoch = 100


# Version 2
# BaseModel: Facenet - Resnet3D
# Optimizer: Adam
# Scheduler: StepLR(step_size=5, gamma=0.5)
# criterion = F.binary_cross_entropy(F.sigmoid(preds), targets)
# img_num = 15
# batch_size = 4
# img_size = 120
# epoch = 100


# Version 3
# BaseModel: Efficientnet-b4 - Resnet3D
# Optimizer: Adam
# Scheduler: StepLR(step_size=5, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# img_num = 15
# batch_size = 4
# img_size = 120
# epoch = 100


# Version 4
# BaseModel: Efficientnet-b0 - Resnet3D
# Optimizer: Adam
# Scheduler: StepLR(step_size=5, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# img_num = 15
# batch_size = 4
# img_size = 120
# epoch = 100
# Add CenterLoss


# Version 5
# BaseModel: Efficientnet-b0 (dim=3)
# Optimizer: Adam
# Scheduler: StepLR(step_size=5, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# batch_size = 4
# img_size = 120
# epoch = 100


# Version 6
# BaseModel: Efficientnet-b4 (dim=3)
# Optimizer: Adam
# Scheduler: StepLR(step_size=10, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# batch_size = 4
# img_size = 120
# epoch = 100
# ckpt


# Version 7
# BaseModel: Efficientnet-b7 (dim=3)
# Optimizer: Adam
# Scheduler: StepLR(step_size=10, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# batch_size = 4
# img_size = 120
# epoch = 100
# ckpt_3


# Version 8
# BaseModel: Efficientnet-b0 (dim=3)
# Optimizer: RAdam  New Optimizer
# Scheduler: StepLR(step_size=5, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# batch_size = 4
# img_size = 120
# epoch = 100
# ckpt_2


# Version 9
# BaseModel: Efficientnet-b0 (dim=3)
# Optimizer: RAdam  New Optimizer
# Scheduler: CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
# criterion = nn.BCEWithLogitsLoss()
# batch_size = 4
# img_size = 120
# epoch = 100
# ckpt


# Version 10
# BaseModel: resnext50_32x4d
# Optimizer: Adam
# Scheduler: StepLR(step_size=10, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# batch_size = 4
# img_size = 120
# epoch = 100
# ckpt_4
# 全然Lossが下がらない


# Version 11
# BaseModel: Efficientnet-b7 (dim=3)
# Optimizer: RAdam
# Scheduler: StepLR(step_size=10, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# batch_size = 4
# img_size = 120
# epoch = 100
# FineTuning  _blocks.48._expand_conv.weight
# ckpt_4


# Version 12
# BaseModel: Efficientnet-b0 (dim=3)
# Preprocessing: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# Optimizer: RAdam  New Optimizer
# Scheduler: StepLR(step_size=5, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# batch_size = 4
# img_size = 120
# epoch = 100
# ckpt
# Version 8と比べて前処理のところを変えてみた
