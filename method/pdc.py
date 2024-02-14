import torch
import torch.nn as nn
import torch.nn.functional as F

# class PDCLoss(nn.Module):
#     def __init__(self, temperature=0.06):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, features, labels, patient_ids, domain_ids):
#         """
#         Custom Supervised Contrastive Loss

#         :param features: Tensor of shape (batch_size, feature_dim) - The feature representations of the samples.
#         :param labels: Tensor of shape (batch_size,) - Class labels of the samples.
#         :param patient_ids: Tensor of shape (batch_size,) - Patient IDs for each sample.
#         :param domain_ids: Tensor of shape (batch_size,) - Domain IDs for each sample.
#         :return: The computed loss value.
#         """

#         # print("labels_ids :",labels)
#         # print("patient_ids :",patient_ids)
#         # print("domain_ids :",domain_ids)
#         features = F.normalize(features, p=2, dim=1)
#         # print("features_size:",features)
        
#         batch_size = features.shape[0]
        
#         # print("batch_size:",batch_size)

#         # Compute the similarity matrix
#         similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
#         # print("similarity_matrix:",similarity_matrix)

#         # Masks for positive and negative pairs
#         labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)
#         patient_dff = patient_ids.unsqueeze(1) != patient_ids.unsqueeze(0)
#         domain_diff = domain_ids.unsqueeze(1) != domain_ids.unsqueeze(0)

#         # print("labels_equal:",labels_equal)
#         # print("patient_equal:",patient_dff)
#         # print("domain_diff:",domain_diff)
#         # Positive pairs: Same class and (different patient or different domain)
#         positive_mask = labels_equal & (patient_dff | domain_diff)

#         # print("positive_mask:", positive_mask)
#         # Negative pairs: Different class
#         negative_mask = ~labels_equal
        
#         # print("negative_mask:", negative_mask)

#         # Mask out the diagonal (self-similarity)
#         identity_mask = ~torch.eye(batch_size, dtype=torch.bool, device=labels.device)
#         # print("identity_mask :",identity_mask)
#         positive_mask = positive_mask & identity_mask
#         negative_mask = negative_mask & identity_mask

#         # Log-sum-exp trick for numerical stability
#         max_sim, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
#         # print("max_sim:", max_sim)
#         exp_similarities = torch.exp(similarity_matrix - max_sim)
#         # print("exp_similarities:", exp_similarities)
#         # Sum of exp similarities for negative pairs
#         sum_negatives = torch.sum(exp_similarities * negative_mask, dim=1)
#         # print("sum_negativess:", sum_negatives)    
#         # Compute loss
#         pos_similarities = exp_similarities * positive_mask
#         # print("pos_similarites_dim1 :" ,torch.sum(pos_similarities, dim=1))
#         # print("pos_similarities:", pos_similarities)   
#         epsilon = 1e-6  # Small constant
#         losses = -torch.log(torch.sum(pos_similarities, dim=1) / sum_negatives + epsilon)
#         print(" losses :",  losses )   
#         loss = torch.mean(losses)
#         print("loss :",loss)
        
#         return loss
    
class PDCLoss(nn.Module):
    def __init__(self, temperature=0.06):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, patient_ids, domain_ids):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # print("similarity_matrix:",similarity_matrix)
        

        # Masks for positive and negative pairs
        labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)
        patient_diff = patient_ids.unsqueeze(1) != patient_ids.unsqueeze(0)
        domain_diff = domain_ids.unsqueeze(1) != domain_ids.unsqueeze(0)

        # Positive and negative masks
        positive_mask = labels_equal & (patient_diff | domain_diff)
        negative_mask = ~labels_equal

        # Mask out the diagonal (self-similarity)
        identity_mask = ~torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        positive_mask = positive_mask & identity_mask
        negative_mask = negative_mask & identity_mask

        # Check for batches with zero positive or negative pairs
        if not positive_mask.any() or not negative_mask.any():
            return torch.tensor(0.0).to(features.device)

        # Log-sum-exp trick for numerical stability
        max_sim, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        exp_similarities = torch.exp(similarity_matrix - max_sim)
        
        # print("exp_similarities:",exp_similarities)


        # Sum of exp similarities for negative pairs
        sum_negatives = torch.sum(exp_similarities * negative_mask, dim=1)

        # Normalize the sums based on the number of pairs
        num_positives = positive_mask.sum(dim=1).float()
        num_negatives = negative_mask.sum(dim=1).float()

        # Avoid division by zero
        epsilon = 1e-6
        num_positives = torch.clamp(num_positives, min=epsilon)
        num_negatives = torch.clamp(num_negatives, min=epsilon)

        # print("num_pos :",num_positives)
        # print("num_neg :",num_negatives)
        
        # Compute loss
        pos_similarities = exp_similarities * positive_mask
        pos_sum = torch.sum(pos_similarities, dim=1) / num_positives
        neg_sum = sum_negatives / num_negatives
        
        # print("pos_sum :",pos_sum)
        # print("neg_sum :",neg_sum)

        losses = -torch.log(pos_sum / neg_sum + epsilon)
        # print("losses : ", losses)
        loss = torch.mean(losses)

        return loss
    