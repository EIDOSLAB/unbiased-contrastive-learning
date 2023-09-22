import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EpsilonSupInfoNCE(nn.Module):
    """e-SupInfoNCE: https://arxiv.org/pdf/2211.05568.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Based on: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, form='out', epsilon=0):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.omega = np.exp(-epsilon)
        self.form = form
        print(self)

    def __repr__(self):
        return f'{self.__class__.__name__}_{self.form} ' \
               f'(t={self.temperature}, ' \
               f'epsilon={-np.log(self.omega)}, ' \
               f'omega={self.omega}) ' \

    def forward_in(self, logits, positive_mask, inv_diagonal):
        """ 
        $ \log \left( \sum_i \left( \frac{\exp(s_i^+)}{\frac{1}{P} \sum_i \exp(s_i^+ - \epsilon)) +  \sum_j \exp(s_j^-)} \right) \right) $
        """
        alignment = torch.log((torch.exp(logits) * positive_mask).sum(1, keepdim=True)) 
        
        uniformity = torch.exp(logits) * inv_diagonal 
        uniformity = ((self.omega * uniformity * positive_mask) / \
                        torch.max(positive_mask.sum(1, keepdim=True), 
                                  torch.ones_like(positive_mask.sum(1, keepdim=True)))) + \
                     (uniformity * (~positive_mask) * inv_diagonal)
        uniformity = torch.log(uniformity.sum(1, keepdim=True))

        log_prob = alignment - uniformity
        loss = - (self.temperature / self.base_temperature) * log_prob
        return loss.mean()

    def forward_out(self, logits, positive_mask, inv_diagonal):
        """ 
        $ - \sum_i \log \left( \frac{\exp(s_i^+)}{\exp(s_i^+ - \epsilon)) +  \sum_j \exp(s_j^-)} \right) $
        """
        alignment = logits 

        # uniformity term = torch.log(sum(exp(logits - diagonal)))
        uniformity = torch.exp(logits) * inv_diagonal 
        uniformity = self.omega*uniformity*positive_mask + \
                     (uniformity*(~positive_mask)*inv_diagonal).sum(1, keepdim=True)
        uniformity = torch.log(uniformity + 1e-6) #prevent nan

        # log(alignment/uniformity) = log(alignment) - log(uniformity)
        log_prob = alignment - uniformity
        
        # compute mean of log-likelihood over positive
        log_prob = (positive_mask * log_prob).sum(1, keepdim=True) / \
                    torch.max(positive_mask.sum(1, keepdim=True), torch.ones_like(positive_mask.sum(1, keepdim=True)))

        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob
        loss = loss.mean()

        if torch.isnan(loss):
            print('alignment', alignment)
            print('uniformity', uniformity)
            print('log_prob', log_prob)

        return loss
    
    def forward(self, features, labels=None):
        """Compute loss for model. If `labels` is None, 
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, n_features]. 
                input has to be rearranged to [bsz, n_views, n_features] and labels [bsz],
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) != 3:
            raise ValueError('`features` needs to be [bsz, n_views, n_feats],'
                             '3 dimensions are required')

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is None:
            mask = torch.eye(batch_size, device=device).bool()
        
        else:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).bool()

        view_count = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            features = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            features = features
            anchor_count = view_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Tile mask
        mask = mask.repeat(anchor_count, view_count)

        # Inverse of torch-eye to remove self-contrast (diagonal)
        inv_diagonal = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size*n_views, device=device).view(-1, 1),
            0
        ).bool()
        # mask now contains the anchor-positive pairs
        # excluding <self,self> on the diagonal
        positive_mask = mask * inv_diagonal

        # compute similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if self.form == 'in':
            loss = self.forward_in(logits, positive_mask, inv_diagonal)
        elif self.form == 'out':
            loss = self.forward_out(logits, positive_mask, inv_diagonal)
        else:
            raise ValueError('Unknown loss form', self.form)
        return loss

class EpsilonSupCon(EpsilonSupInfoNCE):

    def forward_out(self, logits, positive_mask, inv_diagonal):
        # Eq. 18
        """ 
        $ - \sum_i \log \left( \frac{\exp(s_i^+)}{\sum_i \exp(s_i^+ - \epsilon) +  \sum_j \exp(s_j^-)}  \right) $
        """
        alignment = logits 

        # uniformity term = torch.log(sum(exp(logits - diagonal)))
        uniformity = torch.exp(logits) * inv_diagonal 
        uniformity = self.omega*(uniformity*positive_mask) + uniformity*(~positive_mask)*inv_diagonal
        uniformity = torch.log(uniformity.sum(1, keepdim=True) + 1e-6) #prevent nan

        # log(alignment/uniformity) = log(alignment) - log(uniformity)
        log_prob = alignment - uniformity
        
        # compute mean of log-likelihood over positive
        log_prob = (positive_mask * log_prob).sum(1) / \
                    torch.max(positive_mask.sum(1), torch.ones_like(positive_mask.sum(1)))
 
        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob
        return loss.mean()
    
    def forward_in(self, logits, positive_mask, inv_diagonal):
        raise NotImplementedError()


def fairkl(feat, labels, bias_labels, temperature=1.0, kld=0.):
    # feat must be normalized!
    bsz = feat.shape[0]

    if labels.shape[0] != bsz:
        raise ValueError('Num of labels does not match num of features')
    if bias_labels.shape[0] != bsz:
        raise ValueError('Num of bias_labels does not match num of features')

    similarity = torch.div(
        torch.matmul(feat, feat.T),
        temperature
    )

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T)

    bias_labels = bias_labels.view(-1, 1)
    aligned_mask = torch.eq(bias_labels, bias_labels.T)
    conflicting_mask = ~aligned_mask

    pos_conflicting = positive_mask * conflicting_mask
    conflicting_sim = similarity * pos_conflicting
    mu_conflicting = conflicting_sim.sum() / max(pos_conflicting.sum(), 1)
    
    pos_aligned = positive_mask * aligned_mask
    aligned_sim = similarity * pos_aligned
    mu_aligned = aligned_sim.sum() / max(pos_aligned.sum(), 1)

    neg_aligned = (~positive_mask) * aligned_mask
    neg_aligned_sim = similarity * neg_aligned
    mu_neg_aligned = neg_aligned_sim.sum() / max(neg_aligned.sum(), 1)

    neg_conflicting = (~positive_mask) * conflicting_mask
    neg_conflicting_sim = similarity * neg_conflicting
    mu_neg_conflicting = neg_conflicting_sim.sum() / max(neg_conflicting.sum(), 1)

    if mu_conflicting > 1 or mu_conflicting < -1:
        print("mu_conflicting", mu_conflicting.item())

    if mu_aligned > 1 or mu_aligned < -1: 
        print("mu_aligned", mu_aligned.item())

    mu_loss = torch.pow(mu_conflicting - mu_aligned, 2)
    mu_loss += torch.pow(mu_neg_aligned - mu_neg_conflicting, 2)
  
    var_aligned = torch.std(aligned_sim)
    var_conflicting = torch.std(conflicting_sim)
    kld_loss = torch.pow(var_aligned - var_conflicting, 2)

    var_neg_aligned = torch.std(neg_aligned_sim)
    var_neg_conflicting = torch.std(neg_conflicting_sim)
    kld_loss += torch.pow(var_neg_aligned - var_neg_conflicting, 2)

    if torch.isnan(mu_loss):
        print('mu_conflicting:', mu_conflicting)
        print('mu_aligned:', mu_aligned)

    return mu_loss + kld*kld_loss - mu_conflicting + mu_neg_aligned
