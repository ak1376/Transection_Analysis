"""

This code is adapted from the original work by Yonglong Tian (yonglong@mit.edu).
Original code available at: https://github.com/giakou4/classification, https://github.com/HobbitLong/SupContrast/tree/master 

"""
# from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if torch.isnan(loss).any():
            raise ValueError("Detected NaN in tensor!")
            
        loss = loss.view(anchor_count, batch_size).mean()
        
        # I want to visualize the similarity scores for each anchor sample and
        # the negative samples. I will first create a "negative samples" mask,
        # take the anchor_dot_contrast values for each of them, and then plot a
        # histogram of those values. 
        
        negative_mask = logits_mask * (1-mask)
        positive_mask = logits_mask*mask
        
        a = anchor_dot_contrast*negative_mask
        i, j = np.triu_indices(a.shape[0], k=1)
        upper_triangle_values = a[i, j]
        
        # Any value of upper_triangle_values that equals 0 means that it belongs to the masked index
        negative_similarities = upper_triangle_values[upper_triangle_values !=0]
        
        # I also want to extract the similarity scores for the positive samples. The distribution of these values should increase, ideally to 1/temperature. This will allow me to see what the model is doing. For instance,
        # 1. Is the model learning to represent the positive samples 
        #    dissimilarly from the negative samples? 
        # 2. Is the model learning to represent the positive samples similarly?
        # 3. Both 1-2
        
        b = anchor_dot_contrast*positive_mask
        positive_similarities = b[b!=0]
        

        return loss, negative_similarities, positive_similarities
