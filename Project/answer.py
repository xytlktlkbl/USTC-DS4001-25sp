import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h)).view(z.size(0), 28, 28)
    
    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, labels)
        return recon_x, mu, log_var

def vae_loss(recon_x, x, mu, log_var, var=0.5):
    recon_loss = F.mse_loss(recon_x.view(-1, 28, 28), x.view(-1, 28, 28), reduction='sum') / (2 * var)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + kl_loss) / x.size(0)

class GenModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(GenModel, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x, labels):
        x = torch.cat([x, labels], dim=1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        if len(labels.shape) == 1:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        z = torch.cat([z, labels], dim=1)
        h = F.relu(self.fc2(z))
        recon_x = torch.sigmoid(self.fc3(h)).view(z.size(0), 28, 28)
        return recon_x
    
    def forward(self, x, labels):
        if len(labels.shape) == 1:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        x = x.view(x.size(0), -1)
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, labels)
        return recon_x, mu, log_var