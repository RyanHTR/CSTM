import torch
import torch.nn.functional as F
from torch import nn

from .backbones.resnet import resnet50
from .backbones.pytorch_i3d import I3D
from .components.cmam import CMAM


class CSTMModel(nn.Module):
    def __init__(self, opt):
        super(CSTMModel, self).__init__()
        self.opt = opt

        # load pretrained temporal encoder
        temporal_encoder = self.build_i3d()
        self.t_layer1 = nn.Sequential(
            temporal_encoder.conv3d_1a_7x7
        )
        self.t_layer2 = nn.Sequential(
            temporal_encoder.maxPool3d_2a_3x3,
            temporal_encoder.conv3d_2b_1x1,
            temporal_encoder.conv3d_2c_3x3
        )
        self.t_layer3 = nn.Sequential(
            temporal_encoder.maxPool3d_3a_3x3,
            temporal_encoder.mixed_3b,
            temporal_encoder.mixed_3c
        )
        self.t_layer4 = nn.Sequential(
            temporal_encoder.maxPool3d_4a_3x3,
            temporal_encoder.mixed_4b,
            temporal_encoder.mixed_4c,
            temporal_encoder.mixed_4d,
            temporal_encoder.mixed_4e,
            temporal_encoder.mixed_4f
        )
        self.t_layer5 = nn.Sequential(
            temporal_encoder.maxPool3d_5a_2x2,
            temporal_encoder.mixed_5b,
            temporal_encoder.mixed_5c
        )

        # load pretrained spatial encoder
        spatial_encoder = resnet50(pretrained=True)
        self.s_layer1 = nn.Sequential(
            spatial_encoder.conv1,
            spatial_encoder.bn1,
            spatial_encoder.relu
        )
        self.s_layer2 = nn.Sequential(
            spatial_encoder.maxpool,
            spatial_encoder.layer1,
        )
        self.s_layer3 = spatial_encoder.layer2
        self.s_layer4 = spatial_encoder.layer3
        self.s_layer5 = spatial_encoder.layer4

        # txt encoder
        self.txt_encoder = nn.GRU(input_size=300, hidden_size=self.opt.dim_semantic,
                                bias=False, batch_first=True)
        self.txt_norm = nn.Sequential(
            nn.LayerNorm(self.opt.dim_semantic),
            nn.Tanh()
        )

        self.video_dims = [1024, 832, 480, 192, 64, 64]
        self.image_dims = [2048, 1024, 512, 256, 64, 64]

        # CMAM
        self.dec_dim = 256
        self.cmam_s = nn.ModuleList()
        self.cmam_t = nn.ModuleList()
        self.ires_trans = nn.ModuleList()
        self.vres_trans = nn.ModuleList()

        for i in range(5):
            self.cmam_s.append(CMAM(self.image_dims[4-i], self.image_dims[4-i], self.opt.dim_semantic, zero_init=True))
            self.cmam_t.append(CMAM(self.video_dims[4-i], self.video_dims[4-i], self.opt.dim_semantic, num_dim=3, zero_init=True))
            self.ires_trans.append(nn.Sequential(
                nn.Conv2d(self.image_dims[4-i], self.dec_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.dec_dim),
                nn.ReLU(inplace=True)
            ))
            self.vres_trans.append(nn.Sequential(
                nn.Conv2d(self.video_dims[4-i], self.dec_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.dec_dim),
                nn.ReLU(inplace=True)
            ))

        self.conv_fuse1 = nn.ModuleList()
        for i in range(4, -1, -1):
            self.conv_fuse1.append(nn.Sequential(
                nn.Conv2d(self.dec_dim * 2, self.dec_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.dec_dim),
                nn.ReLU(inplace=True)
            ))

        self.conv_fuse2 = nn.ModuleList()
        for i in range(3, -1, -1):
            self.conv_fuse2.append(nn.Sequential(
                nn.Conv2d(self.dec_dim * 2, self.dec_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.dec_dim),
                nn.ReLU(inplace=True)
            ))

        # response map
        self.conv_mask = nn.ModuleList()
        for i in range(6):
            mid_dims = self.dec_dim
            self.conv_mask.append(nn.Sequential(
                nn.Conv2d(in_channels=mid_dims, out_channels=mid_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=mid_dims, out_channels=mid_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=mid_dims, out_channels=1, kernel_size=1)
            ))

    def build_i3d(self):
        i3d_model = I3D(400, modality=self.opt.modality, skip_connect=self.opt.skip)
        state_dict = torch.load('{}/pretrained_weights/i3d_rgb.pth'.format(self.opt.project_root))
        model_dict_keys = i3d_model.state_dict().keys()
        state_dict_keys = state_dict.keys()
        trash_vars = [k for k in model_dict_keys if k not in state_dict_keys]
        print('missing keys:', trash_vars)
        i3d_model.load_state_dict(state_dict, strict=False)
        return i3d_model

    def forward_encoder(self, video, txt, txt_mask, spatials, spatials_3d):
        t_res = []
        s_res = []
        # txt: [B, 512, 20]
        # txt_mask: [B, 20]
        t_txt = txt
        s_txt = txt

        # process video input with multiple frames
        # video: [bs, 3, nf, h, w]
        img = video[:, :, 4, :, :]  # target frame for spatial encoder

        t_1 = self.t_layer1(video)
        # t_1: [B, 64, T, H//2, W//2]
        s_1 = self.s_layer1(img)
        # s_1: [B, 64, H//2, W//2]
        t_1_c = self.cmam_t[0](t_1, t_txt, spatials_3d[-1])
        s_1_c = self.cmam_s[0](s_1, s_txt, spatials[-1])
        t_1 = t_1 + t_1_c
        s_1 = s_1 + s_1_c
        t_res.append(self.vres_trans[0](t_1[:, :, t_1.shape[2] // 2, :, :]))
        s_res.append(self.ires_trans[0](s_1))
        
        t_2 = self.t_layer2(t_1)
        # t_2: [B, 192, T, H//4, W//4]
        s_2 = self.s_layer2(s_1)
        # s_2: [B, 256, H//4, W//4]
        t_2_c = self.cmam_t[1](t_2, t_txt, spatials_3d[-2])
        s_2_c = self.cmam_s[1](s_2, s_txt, spatials[-2])
        t_2 = t_2 + t_2_c
        s_2 = s_2 + s_2_c
        t_res.append(self.vres_trans[1](t_2[:, :, t_2.shape[2] // 2, :, :]))
        s_res.append(self.ires_trans[1](s_2))

        t_3 = self.t_layer3(t_2)
        # t_3: [B, 480, T, H//8, W//8]
        s_3 = self.s_layer3(s_2)
        # s_3: [B, 512, H//8, W//8]
        t_3_c = self.cmam_t[2](t_3, t_txt, spatials_3d[-3])
        s_3_c = self.cmam_s[2](s_3, s_txt, spatials[-3])
        t_3 = t_3 + t_3_c
        s_3 = s_3 + s_3_c
        t_res.append(self.vres_trans[2](t_3[:, :, t_3.shape[2] // 2, :, :]))
        s_res.append(self.ires_trans[2](s_3))

        t_4 = self.t_layer4(t_3)
        # t_4: [B, 832, T, H//16, W//16]
        s_4 = self.s_layer4(s_3)
        # s_4: [B, 1024, H//16, W//16]
        t_4_c = self.cmam_t[3](t_4, t_txt, spatials_3d[-4])
        s_4_c = self.cmam_s[3](s_4, s_txt, spatials[-4])
        t_4 = t_4 + t_4_c
        s_4 = s_4 + s_4_c
        t_res.append(self.vres_trans[3](t_4[:, :, t_4.shape[2] // 2, :, :]))
        s_res.append(self.ires_trans[3](s_4))

        t_5 = self.t_layer5(t_4)
        # t_5: [B, 1024, T, H//32, W//32]
        s_5 = self.s_layer5(s_4)
        # s_5: [B, 2048, H//32, W//32]
        t_5_c = self.cmam_t[4](t_5, t_txt, spatials_3d[-5])
        s_5_c = self.cmam_s[4](s_5, s_txt, spatials[-5])
        t_5 = t_5 + t_5_c
        s_5 = s_5 + s_5_c
        t_5 = self.vres_trans[4](t_5[:, :, t_5.shape[2] // 2, :, :])
        s_5 = self.ires_trans[4](s_5)

        return t_5, s_5, t_res, s_res

    def forward(self, video, txt, txt_mask, spatials):
        # txt encoder
        txt = self.txt_encoder(txt)[0]  
        txt = self.txt_norm(txt).transpose(1, 2)
        # txt: [B, 300, 20]

        # 3d spatials for i3d backbone
        spatials_3d = []
        spatial_size = [4, 4, 4, 2, 1]
        for i in range(4, -1, -1):
            spatials_3d.append(spatials[4 - i].unsqueeze(2).repeat(1, 1, spatial_size[i], 1, 1))

        # visual encoder
        encoder_results = self.forward_encoder(video, txt, txt_mask, spatials, spatials_3d)
        temporal_features, spatial_features, temporal_res_features, spatial_res_features = encoder_results

        vi_im_feat = self.conv_fuse1[4](torch.cat([spatial_features, temporal_features], 1))
        video_features = [vi_im_feat]
        for i in range(5):
            if i >= 1:
                s_t_add = self.conv_fuse1[4-i](torch.cat([temporal_res_features[4-i], spatial_res_features[4-i]], 1))
                video_features[i] = self.conv_fuse2[4-i](torch.cat([video_features[i], s_t_add], 1))
            video_features.append(F.interpolate(video_features[i], scale_factor=2, mode='bilinear', align_corners=True))

        response_maps = []
        for i in range(6):
            cur_mask = self.conv_mask[i](video_features[i])    # [B, 1, H, W]
            if self.opt.up_mask and i > 1:
                cur_mask = F.interpolate(cur_mask, (320, 320))
            response_maps.append(cur_mask.squeeze(1))

        return response_maps
