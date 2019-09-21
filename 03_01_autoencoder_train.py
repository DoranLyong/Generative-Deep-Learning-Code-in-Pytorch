import torch as t
import torch.nn.functional as F
from tqdm import tqdm
from models.Autoencoder import Autoencoder
from torchvision import datasets, transforms

bs = 512
train_ds = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
train_dl = t.utils.data.DataLoader(dataset=train_ds, batch_size=bs, shuffle=True, drop_last=True)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = Autoencoder(train_ds[0][0][None], in_c=1, enc_out_c=[32, 64, 64, 64],
                    enc_ks=[3, 3, 3, 3], enc_pads=[1, 1, 0, 1], enc_strides=[1, 2, 2, 1],
                    dec_out_c=[64, 64, 32, 1], dec_ks=[3, 3, 3, 3], dec_strides=[1, 2, 2, 1],
                    dec_pads=[1, 0, 1, 1], dec_op_pads=[0, 1, 1, 0], z_dim=2)
model.cuda(device)

optimizer = t.optim.Adam(model.parameters(), lr=5e-4, betas=(.9, .99), weight_decay=1e-2)
model.train()

for epoch in tqdm(range(20)):
    if epoch == 10:
        optimizer = t.optim.Adam(model.parameters(), lr=2e-4, betas=(.9, .99), weight_decay=1e-2)
    for i, (data, _) in enumerate(train_dl):
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, data)
        loss.backward()
        optimizer.step()
        if i % 33 == 0:
            print(loss)

print(loss)
t.save(model.state_dict(), 'models/state_dicts/03_01.pth')
