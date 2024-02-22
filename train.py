from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from DiffCrimeModel import DDPM, HamNet
from RiskmapDataset import RiskMapTrain, RiskMapValidation
from early_stopping import EarlyStopping

if __name__ == '__main__':
    n_epoch = 1000
    batch_size = 1
    n_T = 400
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_feat = 256
    lrate = 1e-4

    ddpm = DDPM(nn_model=HamNet(in_channels=1, n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    ddpm.to(device)

    dataset = RiskMapTrain()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    early_stopping = EarlyStopping(save_path="./Result/Output", patience=200, verbose=True)

    for ep in range(n_epoch):
        print(f'train epoch {ep}')
        ddpm.train()
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()

            label = x["image"]
            cond_map = x["cond_map"]
            cond_satellite = x["cond_satellite"]
            c = torch.cat((cond_map, cond_satellite), dim=3)
            history = x['riskmap_171819']
            label = label.to(device)
            c = c.to(device)
            history = history.to(device)

            loss = ddpm(label, c, history)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # Validation
        ddpm.eval()
        datasetVal = RiskMapValidation()
        dataloaderVal = DataLoader(datasetVal, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)
        batchCount = 0
        with torch.no_grad():
            eval_loss = 0
            for y in dataloaderVal:
                batchCount += 1
                filename = y['file_name']
                label = y["image"]
                label = label.to(device)
                cond_map = y["cond_map"]
                cond_satellite = y["cond_satellite"]
                c = torch.cat((cond_map, cond_satellite), dim=3)
                c = c.to(device)

                history = y['riskmap_171819']
                history = history.to(device)

                loss = ddpm(label, c, history)
                eval_loss += loss

            early_stopping(eval_loss / datasetVal.__len__(), ddpm)
            if early_stopping.early_stop:
                print("Early stopping")
                break
