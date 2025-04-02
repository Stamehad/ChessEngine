import yaml
from chessengine.engine_pl import ChessLightningModule
from chessengine.dataclass import ChessPositionDataset
import torch

def test_training_step_runs():

    ########## Get config from YAML ##########
    with open("engine_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ########## Initialize dataset and model ##########
    dataset = ChessPositionDataset(["../chess_engine/data/test_positions.pt"])
    model = ChessLightningModule(config)
    
    ########## Random sample ##########
    N = len(dataset)
    idx = torch.randint(0, N, (1,)).item()
    batch = dataset[idx]
    x, labels = batch  # (8, 8, 21), Dictionary of labels

    ########## Add batch dim ##########
    x = x.unsqueeze(0)  
    labels = {k: v.unsqueeze(0) for k, v in labels.items()}
    labels["true_index"] = labels["true_index"].squeeze(-1)
    batch = (x, labels)
    
    loss = model.training_step(batch, batch_idx=0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar loss