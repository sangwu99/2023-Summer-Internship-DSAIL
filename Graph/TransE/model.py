import torch
import torch.nn as nn 

class TransE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations
        self.k = args.embedding_dim
        self.entity_embedding = nn.Embedding(args.num_entities, args.embedding_dim)
        self.relation_embedding = nn.Embedding(args.num_relations, args.embedding_dim)
        self.initialize()
        
    def initialize(self):
        nn.init.uniform_(self.entity_embedding.weight.data, -6/self.k, 6/self.k)
        nn.init.uniform_(self.relation_embedding.weight.data, -6/self.k, 6/self.k)
        
        self.relation_embedding.weight.data = self.relation_embedding.weight.data / torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
    
    def norm_entity(self):
        self.entity_embedding.weight.data = self.entity_embedding.weight.data / torch.norm(self.entity_embedding.weight.data, dim=1, keepdim=True)
        
    def forward(self, head: torch.Tensor, tail: torch.Tensor, label: torch.Tensor):
        head_embedding = self.entity_embedding(head)
        tail_embedding = self.entity_embedding(tail)
        relation_embedding = self.relation_embedding(label)
        
        score = torch.norm(head_embedding + relation_embedding - tail_embedding, dim=1)
        return score
    