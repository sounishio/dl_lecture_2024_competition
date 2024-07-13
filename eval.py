import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm

from src.datasets_4th import ThingsMEGDataset
from src.models_4 import BasicConvClassifier
from src.utils import set_seed

@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config_4")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------    
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        test_set.num_classes, test_set.seq_len, test_set.num_channels
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model.eval()
    for X, _, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device), None).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission.npy"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")

if __name__ == "__main__":
    run()
