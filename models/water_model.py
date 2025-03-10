import numpy as np
import torch
import torchvision.transforms as T
from .base_model import BaseModel
from . import networks_global, networks_local, networks_local_global, networks_water
from .patchnce import PatchNCELoss, domainNCELoss
import util.util as util
from torch import nn
from models.vgg import vgg16

class WaterModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--QS_mode', type=str, default="water", choices='(global, local, local_global)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_domain', type=float, default=1.0, help='domain loss')
        parser.add_argument('--lambda_per', type=float, default=1.0, help='perception loss')
        parser.add_argument('--lambda_idt', type=float, default=1.0, help='idt loss')
        parser.add_argument('--type_idt', type=str, default='L1', help='idt loss')
        parser.add_argument('--nce_idt', type=util.str2bool, default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'NCE_domain']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.type_idt = opt.type_idt
        
        self.lambda_per = opt.lambda_per
        self.lambda_idt = opt.lambda_idt
        self.lambda_domain = opt.lambda_domain
        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.loss_names += ['per']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'W','HC','HW']
        else:  # during test time, only load G
            self.model_names = ['G']

        if self.opt.QS_mode == 'global':
            networks = networks_global
        elif self.opt.QS_mode == 'local':
            networks = networks_local
        elif self.opt.QS_mode == 'water':
            networks = networks_water
        else:
            networks = networks_local_global

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                       opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, 'context',opt)
        self.netW=networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                         opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, 'style',opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, 
                                      opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netHC = networks.define_H(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, 
                                      opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netHW = networks.define_H(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, 
                                      opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)      
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.vgg = vgg16(pretrained=True).to(self.device)
            if self.type_idt == 'L1':
                self.l1_loss = torch.nn.L1Loss().to(self.device)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.domainNCE=domainNCELoss(opt).to(self.device)
            self.perloss=nn.SmoothL1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_W = torch.optim.Adam(self.netW.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_HC = torch.optim.Adam(self.netW.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_HW = torch.optim.Adam(self.netW.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_W)    
            self.optimizers.append(self.optimizer_HC)  
            self.optimizers.append(self.optimizer_HW)  

    def data_dependent_initialize(self,epoch):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = self.real_A.size(0) // len(self.opt.gpu_ids)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_D()                  # calculate gradients for D
            self.backward_G(epoch)                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self,epoch):
        # forward
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.optimizer_W.zero_grad()
        self.optimizer_HW.zero_grad()
        self.optimizer_HC.zero_grad()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0:
            self.optimizer_F.zero_grad()
        self.backward_G(epoch)                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.optimizer_W.step()
        self.optimizer_HW.step()
        self.optimizer_HC.step()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0:
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.aug_A = input['aug_A'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.isTrain:
            if self.opt.flip_equivariance:
                self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
                if self.flipped_for_equivariance:
                    self.real_A = torch.flip(self.real_A, [3])
                    self.real_B = torch.flip(self.real_B, [3])
            self.A = torch.cat((self.real_A, self.aug_A), dim=0)
            self.styA = self.netW(self.A)
            self.fake, self.cxt = self.netG(self.A)
            self.idt_B, self.cxt_B = self.netG(self.real_B)

            self.fake_B = self.fake[:self.real_A.size(0)]
            self.cxt_A, self.cxt_Ag = self.cxt[:self.real_A.size(0)], self.cxt[self.real_A.size(0):]
            self.sty_A, self.sty_Ag = self.styA[:self.real_A.size(0)], self.styA[self.real_A.size(0):]
        else:
            self.fake_B, _=self.netG(self.real_A)
            # from util.uiqm_batch import parallel_compute_uiqm
            # B,C,H,W=self.fake_B.size()
            # uiqm_batch_mean = parallel_compute_uiqm(self.fake_B.view(B,H,W,C))
            # self.uiqm_batch.append(uiqm_batch_mean)
            # self.uiqm_mean = np.mean(self.uiqm_batch)
            # print(uiqm_batch_mean,self.uiqm_mean)

    def backward_D(self):
        if self.opt.lambda_GAN > 0.0:
            """Calculate GAN loss for the discriminator"""
            fake = self.fake_B.detach()
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD(fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            # Real
            pred_real = self.netD(self.real_B)
            loss_D_real_unweighted = self.criterionGAN(pred_real, True)
            self.loss_D_real = loss_D_real_unweighted.mean()

            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        else:
            self.loss_D_real, self.loss_D_fake, self.loss_D = 0.0, 0.0, 0.0

    def backward_G(self,epoch):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
           self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
           self.loss_NCE = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
           self.loss_NCE_Y = self.lambda_idt*self.calculate_idt_loss(self.real_B, self.idt_B)
           loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
           loss_NCE_both = self.loss_NCE
        
        loss_cxt = self.calc_cxtNCE(self.cxt_A, self.cxt_Ag, self.sty_A)
        loss_idt_cxt = self.calc_cxtNCE(self.cxt_Ag, self.cxt_A, self.sty_Ag)
        #loss_sty = self.calc_styNCE(self.sty_A, self.sty_Ag, self.cxt_A)
        #loss_idt_sty = self.calc_styNCE(self.sty_Ag, self.sty_A, self.cxt_Ag)
        self.loss_NCE_domain = self.lambda_domain*(loss_cxt + loss_idt_cxt)
        self.loss_per =  self.lambda_per * self.calc_per(self.real_A, self.fake_B)
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_NCE_domain + self.loss_per
        self.loss_G.backward()
    def calculate_idt_loss(self, src, tgt):
        if self.type_idt == 'L1':
            return self.l1_loss(src, tgt)
        else:
            return self.calculate_NCE_loss(src, tgt)
    def calculate_NCE_loss(self, src, tgt):
        #print('src,tgt', src.size(), tgt.size())
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids, attn_mats = self.netF(feat_k, self.opt.num_patches, None, None)
        feat_q_pool, _, _ = self.netF(feat_q, self.opt.num_patches, sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calc_cxtNCE(self, anchor, pos, neg):
        B,C,H,W=anchor.size()
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            anchor = torch.flip(anchor, [3])
        # _anchor = anchor.view(B*H*W,C)
        # _pos = pos.view(B*H*W,C)
        # _neg = neg.view(B*H*W,C)
        _anchor = self.netHC(anchor)
        _pos = self.netHC(pos)
        ##################################
        # neg=neg.detach()
        ###################################
        _neg = self.netHW(neg)
        domain_loss = self.domainNCE(_anchor, _pos, _neg).mean()
        return domain_loss

    def calc_styNCE(self, anchor, pos, neg):
        B,C,H,W=anchor.size()
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            anchor = torch.flip(anchor, [3])
        # _anchor = anchor.view(B*H*W,C)
        # _pos = pos.view(B*H*W,C)
        # _neg = neg.view(B*H*W,C)
        _anchor = self.netHW(anchor)
        _pos = self.netHW(pos)
        #############################################
        # pos=pos.detach()
        ##################################################
        _neg = self.netHC(neg)
        domain_loss = self.domainNCE(_anchor, _pos, _neg).mean()
        return domain_loss
    
    def calc_per(self, src, tgt):
        feat_A = self.vgg(src).detach()
        feat_B = self.vgg(tgt)
        return self.perloss(feat_A, feat_B)
    
    def calc_per_ec(self, src, tgt):
        feat_A = self.netG(src,encode_only=True).detach()
        feat_B = self.netG(tgt,encode_only=True)
        return self.perloss(feat_A, feat_B)
