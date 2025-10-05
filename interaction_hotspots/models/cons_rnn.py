import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from interaction_hotspots.models import backbones


def attentionConsistencyLoss(attention_maps, gaze_maps, sigma):
    gaze_maps = gaze_maps.mean(1)
    return (1 / (2 * sigma**2)) * torch.mean( (attention_maps - gaze_maps)**2) + torch.log(sigma)


class FrameLSTM(nn.Module):

    def __init__(self, num_classes, max_len, hidden_size, pool_fn="L2", ant_loss=None):
        super().__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.name = "GazeLSTM"

        self.pool_fn = pool_fn
        self.ant_loss = ant_loss

    def init_backbone(self, backbone):
        self.backbone = backbone()
        self.spatial_dim = self.backbone.spatial_dim
        # NOTE: The LSTM is used as the time aggregation function
        self.rnn = nn.LSTM(
            self.backbone.feat_dim, self.hidden_size, batch_first=True
        )  # (B, T, num_maps)
        # NOTE: Head used for prediciting the action itself
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        
        self.attention_sigma = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        feat_dim = self.backbone.feat_dim
        # NOTE: Used in the process of learning projection of inactive object image to its active equivalent
        self.project = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(True),
        )
        self.backbone_fn = backbone.__name__

        # LSTM hidden state
        n_layers = 1
        h0 = torch.zeros(n_layers, 1, self.hidden_size)
        c0 = torch.zeros(n_layers, 1, self.hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain("relu"))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

        print(
            "FrameLSTM created with out_ch: %d and spatial dim: %d"
            % (self.hidden_size, self.spatial_dim)
        )

    def get_hidden_state(self, B, device):
        h = (
            self.h0.expand(self.h0.shape[0], B, self.h0.shape[2])
            .contiguous()
            .to(device)
        )
        c = (
            self.c0.expand(self.c0.shape[0], B, self.c0.shape[2])
            .contiguous()
            .to(device)
        )
        return (h, c)

    # (B, T, X) --> (B*T, X) --module--> (B*T, Y) --> (B, T, Y)
    def flatten_apply(self, tensor, module):
        shape = tensor.shape
        flat = (shape[0] * shape[1],) + shape[2:]
        tensor = tensor.view(flat)
        out = module(tensor)
        uflat = (
            shape[0],
            shape[1],
        ) + out.shape[1:]
        out = out.view(uflat)
        return out

    # NOTE: Terrible naming for an LSTM layer
    # (B, T, 2048) --> LSTM --> (B, hidden_dim)
    def embed_clip(self, frame_feats, **kwargs):
        # NOTE: Lenghts = number of elements in the batch
        lengths = kwargs["length"].cpu().to(torch.int64)

        # sort clip features by length
        B = frame_feats.shape[0]
        packed_input = pack_padded_sequence(
            frame_feats, lengths, batch_first=True, enforce_sorted=False
        )
        self.rnn.flatten_parameters()  # Otherwise it throws a warning?
        packed_output, (hn, cn) = self.rnn(
            packed_input, self.get_hidden_state(B, frame_feats.device)
        )

        clip_feats = hn[-1]
        output, _ = pad_packed_sequence(packed_output)
        output = output.transpose(0, 1)  # (B, T, 2048)
        return clip_feats, output

    # NOTE: Used for pooling of the feature maps produced by the backbone CNN
    def pool(self, frame_feats):
        if self.pool_fn == "L2":
            pool_feats = F.lp_pool2d(frame_feats, 2, self.spatial_dim)
        elif self.pool_fn == "avg":
            pool_feats = F.avg_pool2d(frame_feats, self.spatial_dim)
        return pool_feats

    def anticipation_loss(self, frame_feats, lstm_feats, batch):
        B = frame_feats.shape[0]
        T = lstm_feats.shape[1]

        # NOTE: Positive features represent the actual instance of "active" action, while negatives
        # are randomly choosen examples. According to the authors of the original paper, this is performed
        # to strengthen the models anticipation of "active features" using inactive object of correct classes.
        # This is to ensure that the model can predict this features using correct inactive features better than
        # when using incorrect classes. This is specifically used in the loss functions
        positive, negative = batch["positive"], batch["negative"]
        target, length = batch["verb"], batch["length"]

        # select the active frame from the clip
        lstm_preds = self.fc(lstm_feats)  # (B, max_L, #classes)
        lstm_preds = lstm_preds.view(B * T, -1)
        target_flat = (
            target.unsqueeze(1).expand(target.shape[0], T).contiguous().view(-1)
        )


        # NOTE: Choosing the frame for which the model is most confident is the representation of the true action.
        pred_scores = -F.cross_entropy(lstm_preds, target_flat, reduction="none").view(
            B, T
        )

        # NOTE: Choose element with the lowest cost associated with it. I.e. the element in which
        # the model is most confident is representative if the true action.
        _, frame_idx = pred_scores.max(1)
        frame_idx = torch.min(
            frame_idx, length - 1
        )  # don't select a padding frame! I.e. padding is used to handle sequences of variable length
        # NOTE: Active feats as in features associated with the active representation?
        active_feats = frame_feats[torch.arange(B), frame_idx]  # (B, 256, H, W)
        # Original code: 
        active_pooled = self.pool(active_feats).view(B, -1)
        # New fix: Use adaptive pooling to handle variable input sizes
        # active_pooled = F.adaptive_avg_pool2d(active_feats, (1, 1)).squeeze(-1).squeeze(-1)

        # NOTE: Function used for creation of the embedding of inactive frame ("State")
        def embed(x):
            pred_frame = self.project(self.backbone(x))
            # Original code: 
            pooled = self.pool(pred_frame).view(B, -1)
            # New fix: Use adaptive pooling to handle variable input sizes
            # pooled = F.adaptive_avg_pool2d(pred_frame, (1, 1)).squeeze(-1).squeeze(-1)
            return pooled

        # NOTE: The actual features used for prediction. I.e. positives are the
        # indented input to the network. This
        positive_pooled = embed(positive)

        # NOTE: Passing "positive" features through the lstm
        _, (hn, cn) = self.rnn(
            positive_pooled.unsqueeze(1),
            self.get_hidden_state(B, positive_pooled.device),
        )
        # NOTE: Generate the prediction of active state image after one time step
        preds = self.fc(hn[-1])
        # aux_loss = F.cross_entropy(preds, target, reduction="none")

        if self.ant_loss == "mse":
            ant_loss = 0.1 * ((positive_pooled - active_pooled) ** 2).mean(1)
        elif self.ant_loss == "triplet":
            negative_pooled = self.backbone(negative)
            # Original code: 
            negative_pooled = self.pool(negative_pooled).view(B, -1)
            # New fix: Use adaptive pooling to handle variable input sizes
            # negative_pooled = F.adaptive_avg_pool2d(negative_pooled, (1, 1)).squeeze(-1).squeeze(-1)
            anc, pos, neg = (
                F.normalize(positive_pooled, 2),
                F.normalize(active_pooled, 2),
                F.normalize(negative_pooled, 2),
            )
            # NOTE: Triplet margin loss ensures that the model maxizes the distance between postives and negatives,
            # while minimizing the distance between positives :
            # "a metric learning loss function that pushes embeddings of similar (anchor-positive) samples
            # closer together while pushing embeddings of dissimilar (anchor-negative) samples further apart
            # by a specified margin, ensuring a minimum separation between positive and negative clusters."
            ant_loss = F.triplet_margin_loss(
                anc, pos, neg, margin=0.5, reduction="none"
            )

        return {"ant_loss": ant_loss} #"aux_loss": aux_loss}

    def refine_cams(self, cam_original, image_shape):
    
        if image_shape[0] != cam_original.size(2) or image_shape[1] != cam_original.size(3):
            cam_original = F.interpolate(cam_original, image_shape, mode="bilinear", align_corners=True) 
        
        B, C, H, W = cam_original.size()
        cams = []
        for idx in range(C):
            cam = cam_original[:, idx, :,:]
            cam = cam.view(B, -1)
            cam_min = cam.min(dim=1, keepdim=True)[0]
            cam_max = cam.max(dim=1, keepdim=True)[0]
            norm = cam_max - cam_min
            norm[norm == 0] = 1e-5
            cam = (cam - cam_min) / norm
            cam = cam.view(B, H, W).unsqueeze(1)
            cams.append(cam)
        cams = torch.cat(cams, dim=1)
        # sigmoid_cams = torch.sigmoid(100 * (cams -0.4))
        return cams

    def forward(self, batch, **kwargs):
        frames, length, gazemaps = batch["frames"], batch["length"], batch["gazemaps"]
        B, T = frames.shape[:2]
        S = self.spatial_dim

        # NOTE: Flatten the features from backbone CNN before passing to next layer
        frame_feats = self.flatten_apply(
            frames, lambda t: self.backbone(t)
        )  # (B, T, 256, 28, 28)

        shape = frame_feats.shape
        new_shape = (shape[0] * shape[1],) + shape[2:]
        fmaps = frame_feats.view(new_shape)

        # NOTE: Global average or L2 pooling applied before the LSTM layer
        # Original code: 
        pool_feats = self.flatten_apply(frame_feats, lambda t: self.pool(t)).view(B, T, -1)  # (B, T, 256)
        # New fix: Use adaptive pooling to handle variable input sizes
        # pool_feats = self.flatten_apply(frame_feats, lambda t: F.adaptive_avg_pool2d(t, (1, 1)).squeeze(-1).squeeze(-1)).view(
        #    B, T, -1
        #)  # (B, T, 2048)

        # NOTE: Embedding of clip is equivalent to passing the output of the backbone through
        # the LSTM layers
        clip_feats, lstm_feats = self.embed_clip(pool_feats, length=length)  # (B, 2048)

        # NOTE: Take the output from LSTM and pass through a fully connected layer
        preds = self.fc(clip_feats)

        fc_params = self.fc.weight.detach().unsqueeze(-1).unsqueeze(-1)
        cams = F.conv2d(fmaps, fc_params, bias=None) #  [B*T, Num_classes, H, W]
        
        cams = self.refine_cams(cams, image_shape=gazemaps.shape[3:])
        
        cams = F.relu(cams)

        # Extract the correct cams: 
        labels = torch.repeat_interleave(torch.argmax(preds, dim=1), 8)
        
        cams = cams[torch.arange(B*T), labels]
        shape = gazemaps.size() 
        attention_loss = attentionConsistencyLoss(cams, gazemaps.view(shape[0]*shape[1], shape[2], shape[3], shape[4]), self.attention_sigma)

        loss_dict = {}
        target = batch["verb"]
        # If provided use class weighting
        if kwargs["class_weights"] is not None:
            cls_loss = F.cross_entropy(preds, target, reduction="none", weight=kwargs["class_weights"])
        else:
            cls_loss = F.cross_entropy(preds, target, reduction="none")

        # NOTE: Save class loss to dict
        loss_dict.update({"cls_loss": cls_loss})
        loss_dict.update({"attention_loss" : attention_loss})

        if not self.training or self.ant_loss is None:
            return preds, loss_dict

        loss_dict.update(self.anticipation_loss(frame_feats, lstm_feats, batch))

        return preds, loss_dict

    # NOTE: What does ic stand for?
    # forward/backward passes for class activation mapping

    # NOTE: This is the forward pass in the backbone dilated CNN
    def ic_features(self, image, **kwargs):
        feat = self.backbone(image)
        return feat.unsqueeze(1)

    # NOTE: This is the anticipation network that is trained in "anticipation_loss".
    # This classifier specifically learns how to anticipate and active frame from an inactive one
    def ic_classifier(self, frame_feats, **kwargs):
        B = frame_feats.shape[0]
        frame_feats = frame_feats.squeeze(1)

        # NOTE: The projection used to learn the features associated with
        # transformation of inactive frame to an active one
        if self.ant_loss is not None:
            frame_feats = self.project(frame_feats)

        # Original code: 
        pool_feats = self.pool(frame_feats).view(B, -1)
        # New fix: Use adaptive pooling to handle variable input sizes
        # pool_feats = F.adaptive_avg_pool2d(frame_feats, (1, 1)).squeeze(-1).squeeze(-1)

        _, (hn, cn) = self.rnn(
            pool_feats.unsqueeze(1), self.get_hidden_state(B, pool_feats.device)
        )
        pred = self.fc(hn[-1])
        return pred


def cons_frame_lstm(num_classes, max_len, backbone, hidden_size=2048, ant_loss="mse"):
    net = FrameLSTM(num_classes, max_len, hidden_size, ant_loss=ant_loss)
    net.init_backbone(backbone)
    print("Using backbone class: %s" % backbone)
    print("Using ant loss fn: %s" % ant_loss)
    return net
