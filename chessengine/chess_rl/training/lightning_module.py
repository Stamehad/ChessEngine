# chess_rl/training/lightning_module.py
import torch
import torch.optim as optim
import pytorch_lightning as pl
from importlib import import_module # To load model dynamically
from .loss import compute_loss

class ChessRLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config) # Save config to checkpoint

        # Dynamically load the model class from the specified path
        model_module_path = config['model']['model_path'].replace('/', '.').replace('.py', '')
        model_class_name = self._find_model_class_name(config['model']['model_path']) # Helper needed
        ModelClass = getattr(import_module(model_module_path), model_class_name)

        # Load model configuration (e.g., dimensions, layers) if needed
        # model_params = load_json_config(config['model']['config_path']) # Helper needed
        # self.model = ModelClass(**model_params)
        # Simplified: Assume ModelClass can be instantiated without extra args for now
        self.model = ModelClass() # Adjust instantiation based on your model definition


        # --- Important: Load weights ---
        # Load latest weights if available (from self-play or previous training)
        checkpoint_path = os.path.join(config['model']['checkpoint_dir'], config['model']['latest_checkpoint_name'])
        if os.path.exists(checkpoint_path):
            print(f"Loading model weights from: {checkpoint_path}")
            try:
                 # Load state_dict directly (handles cases where it was saved from nn.Module)
                 state_dict = torch.load(checkpoint_path, map_location='cpu')
                 # Adjust keys if saved from LightningModule (e.g., remove 'model.')
                 if list(state_dict.keys())[0].startswith('model.'):
                     state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items()}
                 self.model.load_state_dict(state_dict)

            except Exception as e:
                 print(f"Warning: Failed to load weights from {checkpoint_path}. Starting with initial weights. Error: {e}")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting with initial model weights.")
        # ------------------------------

    def _find_model_class_name(self, model_path):
        """Helper to find the nn.Module class name in the model file."""
        # Basic implementation: assumes one class inherits nn.Module
        import inspect
        import sys
        module_name = model_path.replace('/', '.').replace('.py', '')
        # Add directory to path if necessary
        model_dir = os.path.dirname(model_path)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        module = import_module(module_name)
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
                 print(f"Found model class: {name}")
                 # Remove added path
                 if model_dir in sys.path and sys.path[0] == model_dir:
                    sys.path.pop(0)
                 return name
        raise NameError(f"Could not find nn.Module class in {model_path}")


    def forward(self, x):
        # Assumes your model returns (policy_logits, value)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, policy_targets, value_targets = batch
        policy_logits, value_preds = self(states) # Calls forward

        loss, policy_loss, value_loss = compute_loss(
            policy_logits, value_preds, policy_targets, value_targets, self.config
        )

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('policy_loss', policy_loss, on_step=True, on_epoch=True, logger=True)
        self.log('value_loss', value_loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        # Optional: Add learning rate scheduler here if needed
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        # return [optimizer], [scheduler]
        return optimizer

    def on_train_epoch_end(self):
        # --- Important: Save core model weights after training ---
        checkpoint_dir = self.config['model']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, self.config['model']['latest_checkpoint_name'])
        try:
            # Save only the state_dict of the underlying model (not the LightningModule)
            torch.save(self.model.state_dict(), save_path)
            print(f"Saved core model weights to {save_path}")
        except Exception as e:
            print(f"Error saving model state_dict: {e}")
        # -------------------------------------------------------