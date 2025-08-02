import torch
import torch.nn as nn
import math
import pytorch_lightning as pl
from torch.optim import Adam
import torch.nn.functional as F

class VAEErrorModel(nn.Module):

    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, q_dim, kv_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.query_proj = nn.Linear(q_dim, embed_dim)
        self.key_proj = nn.Linear(kv_dim, embed_dim)
        self.value_proj = nn.Linear(kv_dim, embed_dim)

        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_size, query_len, embed_dim)
        :param key: (batch_size, key_len, embed_dim)
        :param value: (batch_size, value_len, embed_dim)
        :param mask: (batch_size, query_len, key_len)
        :return: output, attention_weights
        """
        B, N1, _ = query.shape
        _, N2, _ = key.shape

        Q = self.query_proj(query).reshape(B, N1, self.num_heads, int(self.embed_dim / self.num_heads)).permute(0, 2, 1, 3)  # (batch, num_heads, N, dim)
        K = self.key_proj(key).reshape(B, N2, self.num_heads, int(self.embed_dim / self.num_heads)).permute(0, 2, 1, 3)  # (batch, num_heads, N, dim)
        V = self.value_proj(value).reshape(B, N2, self.num_heads, int(self.embed_dim / self.num_heads)).permute(0, 2, 1, 3)  # (batch, num_heads, N, dim)

        att = (Q @ K.transpose(-2, -1)) * self.scale  # (batch, num_heads, N, N)
        att = att.softmax(dim=-1)
        att = self.dropout(att)
        attention_output = (att @ V).transpose(1, 2).flatten(2)  # B,N,dim

        output_ = self.output_proj(attention_output)  # (batch_size, query_len, embed_dim)
        output_ = self.norm(output_)
        return output_

class CrossAttentionModel(nn.Module):
    def __init__(self, embed_dim, his_dim, ge_dim, num_heads=8):
        super(CrossAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.cross_attention1 = CrossAttentionLayer(embed_dim=embed_dim, q_dim=his_dim, kv_dim=ge_dim, num_heads=num_heads)
        self.cross_attention2 = CrossAttentionLayer(embed_dim=embed_dim, q_dim=ge_dim, kv_dim=his_dim, num_heads=num_heads)

        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, img, exp):

        his_attended = self.cross_attention1(img, exp, exp)
        gene_attended = self.cross_attention2(exp, img, img)
        fused_feat = self.fusion_proj(torch.cat([his_attended[:, 0, :], gene_attended[:, 0, :]], dim=-1))

        fused_feat = self.fusion_norm(fused_feat)

        return fused_feat

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        original_shape = x.shape

        if x.dim() == 3:
            batch_size, neighbors, dims = x.shape
            x = x.view(-1, dims)

        output = self.fc(x)

        if len(original_shape) == 3:
            output = output.view(original_shape[0], original_shape[1], -1)

        return output

class ContrastiveModel(pl.LightningModule):
    def __init__(self, lr, img_dim, gene_dim, proj_dim, temperature=0.15):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.temperature = temperature
        self.img_encoder = MLPEncoder(img_dim, proj_dim)
        self.gene_encoder = MLPEncoder(gene_dim, proj_dim)

    def forward(self, img, gene):
        img_z = self.img_encoder(img)
        gene_z = self.gene_encoder(gene)
        return F.normalize(img_z, dim=1), F.normalize(gene_z, dim=1)

    def contrastive_loss(self, img_z, gene_z):

        logits = torch.mm(img_z, gene_z.T) / self.temperature

        targets = torch.arange(logits.size(0)).to(logits.device)

        loss_img_to_gene = F.cross_entropy(logits, targets)

        loss_gene_to_img = F.cross_entropy(logits.T, targets)

        lamuda = 0.5
        loss = (lamuda * loss_img_to_gene + (1-lamuda) * loss_gene_to_img)

        return loss

    def training_step(self, batch, batch_idx):
        img, gene, _ = batch
        img_z, gene_z = self.forward(img, gene)
        loss = self.contrastive_loss(img_z, gene_z)
        self.log('contrastive_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class ReconstructionModel(pl.LightningModule):
    def __init__(self, contrastive_checkpoint_path, proj_dim, img_dim, gene_dim, lr):
        super().__init__()
        self.lr = lr

        self.CL = ContrastiveModel.load_from_checkpoint(
            contrastive_checkpoint_path,
            lr=lr,
            img_dim=img_dim,
            gene_dim=gene_dim,
            proj_dim=proj_dim
        )

        self.BCA = CrossAttentionModel(embed_dim=512, his_dim=proj_dim, ge_dim=proj_dim)

        self.vae_error_model = VAEErrorModel(input_dim=512, latent_dim=64)

        self.target_mu_0 = nn.Parameter(torch.randn(64) * 0.01)
        self.target_mu_1 = nn.Parameter(torch.randn(64) * 0.01)

        self.decoder = nn.Sequential(
            nn.Linear(64+64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.scores = []

    def forward(self, img, gene):
        img_z, gene_z = self.CL(img, gene)

        multimodal = self.BCA(img_z, gene_z)

        recon_multimodal, mu, logvar = self.vae_error_model(multimodal)

        combined_feature = torch.cat([mu, logvar], dim=1)

        logit = self.decoder(combined_feature)

        return multimodal, recon_multimodal, logit, mu, logvar, img_z, gene_z

    def training_step(self, batch, batch_idx):
        img, gene, _, y = batch
        multimodal, recon_multimodal, logits, mu, logvar, img_z, gene_z = self.forward(img, gene)
        logits = logits.squeeze(1)

        vae_recon_loss = F.mse_loss(recon_multimodal, multimodal)

        target_mu = torch.where(y.unsqueeze(1) == 1, self.target_mu_1, self.target_mu_0)
        clu_kl_loss = 0.5 * torch.sum(logvar.exp() + (mu - target_mu).pow(2) - logvar - 1) / mu.size(0)
        separation_loss = - torch.norm(self.target_mu_0 - self.target_mu_1, p=2)

        vae_total_loss = vae_recon_loss + 0.5 * clu_kl_loss + 0.1 * separation_loss

        if self.current_epoch < 50:
            self.log('vae_total_loss', vae_total_loss, prog_bar=True)
            return vae_total_loss

        else:
            classification_loss = F.binary_cross_entropy_with_logits(logits, y.float())
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean()

            total_loss = classification_loss + 0.1 * (vae_total_loss)

            self.log('classifier_loss', classification_loss, prog_bar=True)
            self.log('acc', acc, prog_bar=True)

            return total_loss

    def test_step(self, batch, batch_idx):
        img, gene, _, y = batch
        multimodal, recon_multimodal, logits, mu, logvar, img_z, gene_z = self.forward(img, gene)
        logits = logits.squeeze(1)

        self.scores.append(logits.cpu())

        return self.scores

    def on_test_epoch_end(self):
        self.scores = torch.cat(self.scores, dim=0)

        print(f"Test finished, total scores shape: {self.scores.shape}")


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

