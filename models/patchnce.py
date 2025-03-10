from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        #print('feat_q,k', feat_q.size(), feat_k.size())
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        #print('qk', feat_q.device, feat_k.device)
        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit -- current batch
        # reshape features to batch size
        feat_q = feat_q.view(self.opt.batch_size, -1, dim)
        feat_k = feat_k.view(self.opt.batch_size, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) # b*np*np

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss

# class NCELoss(nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         self.opt = opt
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

#     def forward(self, feat_a, feat_p, feat_n):#[BX,C] anchor, positive, negative
#         print('feat_apn', feat_a.size(), feat_p.size(), feat_n.size())
#         batchSize = feat_a.shape[0]
#         dim = feat_a.shape[1]
#         feat_p = feat_p.detach()
#         feat_n = feat_n.detach()
        
#         # pos logit
#         l_pos = torch.bmm(feat_a.view(batchSize, 1, -1), feat_p.view(batchSize, -1, 1))#BX,1,C  BX,C,1  -> BX, 1, 1
#         l_pos = l_pos.view(batchSize, 1)#BX,1

#         # neg logit -- current batch
#         # reshape features to batch size
#         feat_a = feat_a.view(self.opt.batch_size, -1, dim)#B,X,C
#         feat_n = feat_n.view(self.opt.batch_size, dim, -1)#B,C,X
#         npatches = feat_a.size(1)
#         l_neg_curbatch = torch.bmm(feat_a, feat_n) # b*np*np  B,X,X

#         # diagonal entries are similarity between same features, and hence meaningless.
#         # just fill the diagonal with very small number, which is exp(-10) and almost zero
#         diagonal = torch.eye(npatches, device=feat_a.device, dtype=self.mask_dtype)[None, :, :]# 1,X,X
#         l_neg_curbatch.masked_fill_(diagonal, -10.0)
#         l_neg = l_neg_curbatch.view(-1, npatches)#BX,X

#         out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T #BX,X+1

#         loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
#                                                         device=feat_a.device))
#         print('nceloss',loss.size())
#         return loss

class domainNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_a, feat_p, feat_n):#[BX,C] anchor, positive, negative
        batchSize = feat_a.shape[0]
        dim = feat_a.shape[1]
        feat_p = feat_p.detach()
        feat_n = feat_n.detach()#??
        #print(feat_a.size(), feat_p.size(), feat_n.size())
        # pos logit
        l_pos = torch.bmm(feat_a.view(batchSize, 1, -1), feat_p.view(batchSize, -1, 1))#BX,1,C  BX,C,1  -> BX, 1, 1
        l_pos = l_pos.view(batchSize, 1)#BX,1
        #('l_pos', l_pos.size(), batchSize)

        # neg logit -- current batch
        # reshape features to batch size
        feat_a = feat_a.view(self.opt.batch_size, -1, dim)#B,X,C
        feat_n = feat_n.view(self.opt.batch_size, dim, -1)#B,C,X
        npatches = feat_a.size(1)
        l_neg_curbatch = torch.bmm(feat_a, feat_n) # b*np*np  B,X,X

        l_neg = l_neg_curbatch.view(-1, npatches)#BX,X
        #print('l_neg', l_neg.size())

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T #BX,X+1

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_a.device))
        return loss