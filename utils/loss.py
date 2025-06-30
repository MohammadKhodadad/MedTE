import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive):
        # Normalize the embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        
        # Compute cosine similarity matrix (anchor-positive, including self-similarity)
        similarity_matrix = torch.matmul(anchor, positive.T) / self.temperature
        # with torch.cuda.amp.autocast(dtype=torch.float32):  # Ensure stable computation
        #     similarity_matrix = torch.matmul(anchor, positive.T) / self.temperature

        # Create labels: each anchor should be most similar to its corresponding positive
        batch_size = anchor.size(0)
        labels = torch.arange(batch_size).to(anchor.device)

        # Apply cross-entropy loss on the similarity matrix
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class SupervisedInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positives, negatives):
        """
        anchor: Tensor of shape (batch_size, embedding_dim)
        positives: Tensor of shape (batch_size, embedding_dim)
        negatives: Tensor of shape (batch_size, num_negatives, embedding_dim)
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)  # (batch_size, embedding_dim)
        positives = F.normalize(positives, p=2, dim=-1)  # (batch_size, embedding_dim)
        negatives = F.normalize(negatives, p=2, dim=-1)  # (batch_size, num_negatives, embedding_dim)

        # Compute positive similarity (dot product with anchor)
        positive_sim = (anchor * positives).sum(dim=-1, keepdim=True) / self.temperature  # (batch_size, 1)

        # Compute negative similarities using batch matrix multiplication
        # negatives: (batch_size, num_negatives, embedding_dim)
        # anchor.unsqueeze(2): (batch_size, embedding_dim, 1)
        # print(negatives.shape,anchor.shape)
        negative_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / self.temperature  # (batch_size, num_negatives)

        # Concatenate positive and negative similarities
        logits = torch.cat([positive_sim, negative_sim], dim=1)  # (batch_size, 1 + num_negatives)

        # Create labels (positive is always at index 0)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss




if __name__ == "__main__":
    # Example usage
    batch_size = 16
    embedding_dim = 128
    negative_dim = 5
    anchor = torch.randn(batch_size, embedding_dim)  # Embedding for anchor (sentence 1)
    positive = torch.randn(batch_size, embedding_dim)  # Embedding for positive (sentence 2)
    negatives = torch.randn(batch_size, negative_dim, embedding_dim)  # Embedding for positive (sentence 2)
    # loss_fn = InfoNCELoss(temperature=0.07)
    # loss = loss_fn(anchor, positive)
    loss_fn = SupervisedInfoNCELoss(temperature=0.07)
    loss = loss_fn(anchor, positive, negatives)
    print("Modified InfoNCE Loss:", loss.item())
