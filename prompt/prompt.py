# copy and modified from 
# https://github.com/ucinlp/autoprompt/blob/master/autoprompt/create_trigger.py

import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, embs, lr, weight_decay):
        super().__init__()
        self.lr = lr
        self.static_embs = embs
        self.embs = nn.Parameter(embs.clone(), requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)




class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        # gradient_dot_embedding_matrix = gradient_dot_embedding_matrix.mean(dim=-1)
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.sum(dim=-1).topk(num_candidates)

    return top_k_ids

