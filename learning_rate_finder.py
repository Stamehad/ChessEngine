from chessengine.model.engine_pl import ChessLightningModule
from chessengine.model.datamodule import ChessDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
import matplotlib.pyplot as plt
import yaml
import os
from dotenv import load_dotenv

load_dotenv()
TEST_DATASET = os.getenv("TEST_DATASET")
training_paths = os.getenv("TRAINING_DATASET").split(", ")


# load config from engine_config.yaml
with open("engine_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Update config with mps settings
config['train']['accelerator'] = "mps"
config['dataloader']['num_workers'] = 0
config['dataloader']['persistent_workers'] = False

print(config)
data_paths = [TEST_DATASET]
data_paths = training_paths[:1]  # Use only the first path for testing
dm = ChessDataModule(config, data_paths)
model = ChessLightningModule(config)

trainer = Trainer(accelerator="mps")
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, datamodule=dm)

# Plot suggestion
fig = lr_finder.plot(suggest=True)
fig.show()

# The new learning rate is updated in model.hparams.lr
print("Suggested LR:", model.hparams.lr)
print(lr_finder.suggestion())

# # Save best lr back to config
# config['train']['lr'] = lr_finder.suggestion()