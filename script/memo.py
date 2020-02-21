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
# BaseModel: Facenet - Resnet3D
# Optimizer: Adam
# Scheduler: StepLR(step_size=5, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()
# img_num = 15
# batch_size = 4
# img_size = 120
# epoch = 100
# Add "Logloss" (tensorboard)
