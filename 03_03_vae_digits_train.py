import torch as t
import torch.nn.functional as F
from tqdm import tqdm
from models.VariationalAutoencoder import VariationalAutoencoder
from torchvision import datasets, transforms

bs = 32
train_ds = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
train_dl = t.utils.data.DataLoader(dataset=train_ds, batch_size=bs, shuffle=True, drop_last=True)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = VariationalAutoencoder(train_ds[0][0][None], in_c=1, enc_out_c=[32, 64, 64, 64],
                    enc_ks=[3, 3, 3, 3], enc_pads=[1, 1, 0, 1], enc_strides=[1, 2, 2, 1],
                    dec_out_c=[64, 64, 32, 1], dec_ks=[3, 3, 3, 3], dec_strides=[1, 2, 2, 1],
                    dec_pads=[1, 0, 1, 1], dec_op_pads=[0, 1, 1, 0], z_dim=2)
model.cuda(device)
model.train()

def vae_kl_loss(mu, log_var):
    return -.5 * t.sum(1 + log_var - mu ** 2 - log_var.exp())

def vae_loss(y_pred, mu, log_var, y_true):
    r_loss = F.binary_cross_entropy(y_pred, y_true, reduction='sum')
    kl_loss = vae_kl_loss(mu, log_var)
    return r_loss + kl_loss

lr = .0005
for epoch in tqdm(range(20)):
    optimizer = t.optim.Adam(model.parameters(), lr=lr / (epoch * 2 + 1), betas=(.9, .99), weight_decay=1e-2)
    for i, (data, _) in enumerate(train_dl):
        data = data.to(device)
        optimizer.zero_grad()
        pred, mu, log_var = model(data)
        loss = vae_loss(pred, mu, log_var, data)
        loss.backward()
        optimizer.step()
        if i % 33 == 0:
            print(loss)

print(loss)
t.save(model.state_dict(), 'models/state_dicts/03_03.pth')
