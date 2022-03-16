import torch
from torch import nn
import torch.nn.functional as F
from .models import New_Audio_Guided_Attention
from .models import EncoderLayer, Encoder, DecoderLayer, Decoder
from torch.nn import MultiheadAttention
from .Dual_lstm import Dual_lstm


class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, int(d_model / 2), num_layers=num_layers, batch_first=True,
                                 bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
                                  dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output





class CAS_Module(nn.Module):
    def __init__(self, d_model, num_class=28):
        super(CAS_Module, self).__init__()
        self.d_model = d_model
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class+1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, content):

        content = content.permute(0, 2, 1)

        out = self.classifier(content)
        out = out.permute(0, 2, 1)
        return out


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        # self.cas_model = CAS_Module(d_model=d_model, num_class=28)

    # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        # class_logits = self.event_classifier(fused_content.transpose(1,0))
        # sorted_scores_base,_ = class_logits.sort(descending=True, dim=1)
        # topk_scores_base = sorted_scores_base[:, :4, :]
        # class_logits = torch.mean(topk_scores_base, dim=1)
        class_scores = class_logits

        return logits, class_scores


class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim  # need to equal d_model
        self.classifier = nn.Linear(self.hidden_dim, 1)  # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        fused_content = fused_content.transpose(0, 1)
        max_fused_content, _ = fused_content.max(1)
        # confident scores
        is_event_scores = self.classifier(fused_content)
        # classification scores
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        # fused
        fused_logits = is_event_scores.sigmoid() * raw_logits
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores

class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output


class weak_main_model(nn.Module):
    def __init__(self, config):
        super(weak_main_model, self).__init__()
        self.config = config
        self.beta = self.config["beta"]
        self.alpha = self.config["alpha"]
        self.gamma = self.config["gamma"]
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta).cuda()
        self.video_input_dim = self.config["video_inputdim"]
        self.video_fc_dim = self.config["video_inputdim"]
        self.d_model = self.config["d_model"]
        self.audio_input_dim = self.config["audio_inputdim"]
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=2048)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.audio_input_dim, d_model=self.d_model, feedforward_dim=2048)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.audio_input_dim, d_model=self.d_model, feedforward_dim=1024)
        #self.audio_visual_rnn_layer = RNNEncoder(audio_dim=128, video_dim=512, d_model=256, num_layers=1)
        self.AVInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        #self.localize_module = WeaklyLocalizationModule(self.d_model)
        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),

            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.CAS_model = CAS_Module(d_model=self.d_model, num_class=28)
        self.classifier = nn.Linear(self.d_model, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.audio_cas = nn.Linear(self.d_model, 29)
        self.video_cas = nn.Linear(self.d_model, 29)

    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]
        # this fc is optinal, that is used for adaption of different visual features (e.g., vgg, resnet).
        #audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        visual_feature = visual_feature.transpose(1, 0).contiguous()
        #visual_rnn_input = visual_feature


        # audio query
        video_key_value_feature = self.video_encoder(visual_feature)
        audio_query_output = self.audio_decoder(audio_feature, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_feature)
        video_query_output = self.video_decoder(visual_feature, audio_key_value_feature)

        audio_gate = self.audio_gated(video_key_value_feature)
        video_gate = self.video_gated(audio_key_value_feature)

        av_gate = (audio_gate + video_gate) / 2
        av_gate = av_gate.permute(1, 0, 2)

        video_query_output = (1 - self.alpha)*video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = (1 - self.alpha)*audio_query_output + video_gate * audio_query_output * self.alpha

        video_cas = self.video_cas(video_query_output)
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)

        video_cas_gate = video_cas.sigmoid()
        audio_cas_gate = audio_cas.sigmoid()
        #
        # video_cas_gate = (video_cas_gate > 0.01).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.01).float()*audio_cas_gate

        # video_cas = audio_cas_gate.unsqueeze(1) * video_cas
        # audio_cas = video_cas_gate.unsqueeze(1) * audio_cas
        #
        # sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        # topk_scores_video = sorted_scores_video[:, :4, :]
        # score_video = torch.mean(topk_scores_video, dim=1)
        # sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        # topk_scores_audio = sorted_scores_audio[:, :4, :]
        # score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 29]
        #
        # video_cas_gate = score_video.sigmoid()
        # audio_cas_gate = score_audio.sigmoid()
        # video_cas_gate = (video_cas_gate > 0.5).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.5).float()*audio_cas_gate

        #
        # av_score = (score_video + score_audio) / 2


        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)
        #scores = self.localize_module((video_query_output+audio_query_output)/2)


        fused_content = (video_query_output+audio_query_output)/2
       # fused_content = video_query_output
        fused_content = fused_content.transpose(0, 1)
        #is_event_scores = self.classifier(fused_content)

        cas_score = self.CAS_model(fused_content)
        #cas_score = cas_score + 0.2*video_cas_gate.unsqueeze(1)*cas_score + 0.2*audio_cas_gate.unsqueeze(1)*cas_score
        cas_score = self.gamma*video_cas_gate*cas_score + self.gamma*audio_cas_gate*cas_score
        #cas_score = cas_score*2
        sorted_scores, _ = cas_score.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :4, :]
        raw_logits = torch.mean(topk_scores, dim=1)[:, None, :]       #[32, 29]

        #fused_logits = is_event_scores.sigmoid() * raw_logits
        fused_logits = av_gate * raw_logits
        # fused_scores, _ = fused_logits.sort(descending=True, dim=1)
        # topk_scores = fused_scores[:, :3, :]
        # logits = torch.mean(topk_scores, dim=1)
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        event_scores = event_scores

        return av_gate.squeeze(), raw_logits.squeeze(), event_scores


class supv_main_model(nn.Module):
    def __init__(self, config):
        super(supv_main_model, self).__init__()
        self.config = config
        self.beta = self.config["beta"]
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta).cuda()
        self.video_input_dim = self.config['video_inputdim']
        self.audio_input_dim = self.config['audio_inputdim']

        self.video_fc_dim = 512
        self.d_model = self.config['d_model']

        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim, d_model=self.d_model, num_layers=1)

        self.audio_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )
        self.video_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )

        self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.localize_module = SupvLocalizeModule(self.d_model)
        self.video_norm = nn.LayerNorm(self.d_model)
        self.audio_norm = nn.LayerNorm(self.d_model)
        self.audio_cas = nn.Linear(self.d_model, 28)
        self.video_cas = nn.Linear(self.d_model, 28)
        self.alpha = self.config['alpha']
        self.gamma = self.config['gamma']


    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        visual_rnn_input = visual_feature

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 256]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 512]


        # audio query
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        audio_query_output = self.audio_decoder(audio_encoder_input1, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_query_output = self.video_decoder(visual_encoder_input1, audio_key_value_feature)


        audio_gate = self.audio_gated(audio_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature)

        audio_visual_gate = audio_gate * video_gate

        video_query_output = video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.alpha


        video_cas = self.video_cas(video_query_output)  # [10, 32, 28]
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)
        sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        topk_scores_video = sorted_scores_video[:, :4, :]
        score_video = torch.mean(topk_scores_video, dim=1)
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 28]

        # event_visual_gate = score_video.sigmoid()
        # event_audio_gate = score_audio.sigmoid()

        av_score = (score_video + score_audio) / 2

        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)

        is_event_scores, event_scores = self.localize_module((video_query_output + audio_query_output)/2)
        event_scores = event_scores + self.gamma*av_score
        #event_scores = event_scores + self.gamma * (event_visual_gate * event_audio_gate) * event_scores


        return is_event_scores, event_scores, audio_visual_gate, av_score

