import torch
from torch.utils.data import DataLoader
import numpy as np

from DiffCrimeModel import DDPM, HamNet
from RiskmapDataset import RiskMapTest

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_feat = 256
    n_T = 400
    batch_size = 1

    ddpm = DDPM(nn_model=HamNet(in_channels=1, n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load("./Result/Output/best_network_with_early_stopping.pth"))

    ddpm.to(device)

    ddpm.eval()

    dataset = RiskMapTest()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)

    ws_test = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # strength of generative guidance

    batchCount = 0
    with torch.no_grad():
        for x in dataloader:
            batchCount += 1
            filename = x['file_name']
            label = x["image"]
            cond_map = x["cond_map"]
            cond_satellite = x["cond_satellite"]
            c = torch.cat((cond_map, cond_satellite), dim=3)
            history = x['riskmap_171819']
            history = history.to(device)

            label = label.to(device)
            c = c.to(device)

            for w_i, w in enumerate(ws_test):
                x_gen = ddpm.crime_sample(len(label), (1, 16, 16), c, history, device, guide_w=w)

                for i in range(len(filename)):
                    matrix = x_gen[i, 0].cpu()
                    npyFileName = f'./Result/Riskmaps/result_test_{w}_{filename[i]}.npy'
                    np.save(npyFileName, matrix)
