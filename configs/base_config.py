# coding=utf-8

class config(object):
    def adjust_parm(self, value):
        pass

    def get_txt_encoder_num(self, text_encoding):
        encoder_num = 0
        for name in text_encoding:
            encoder_value = text_encoding[name]['name']
            if 'no' not in encoder_value:
                encoder_num += 1

        return encoder_num

    model_name = 'w2vpp_mutivis_attention' 

    text_encoding = {
        'bow_encoding': {'name': 'bow_nsw'},  # [nobow_nsw, bow_nsw]
        'w2v_encoding': {'name': 'w2v_nsw'},  # [now2v_nsw, w2v_nsw]
        'rnn_encoding': {'name': 'gru_mean'},  # [gru_mean, bigru_mean, nogru_mean]
        'bert_encoding': {'name': 'noBert',  # [noBert, bert-base-uncased, \bert_name, ...]
                          'dir_name': 'bert-base-uncased'
                          },
        'CLIP_encoding': {'name': 'noCLIP',  # [noCLIP, ViT-B/32, \CLIP_name, ...]
                          'dir_name': 'CLIP_ViT-B32'
                          },
        'NetVLAD_encoding': {'name': 'noNetVLAD'},  # [noNetVLAD, NetVLAD]
    }
    text_encoder_num = 3
    threshold = 5
    bow_norm = 0
    we_dim = 500
    rnn_size = 1024
    rnn_layer = 1
    txt_fc_layers = '0-2048'
    txt_norm = 2  # L_2 norm


    bert_size = 768
    bert_frozen = False
    bert_do_lower_case = True
    bert_transform_batch_norm = True
    bert_transform_dropout = 0
    bert_transform_activation = 'tanh'
    clip_opt = {
        'size': 512, 'transform_batch_norm': False, 'transform_dropout': 0.0,
        'transform_activation': 'tanh', 'frozen': True
    }
    # if text_encoding includes NetVLAD
    NetVLAD_opt = {
        'num_clusters': 32, 'alpha': 100, 'normalize_pooling': False,
    }

    # visual transform
    vis_fc_layers = ['0', 2048]
    vis_norm = 2  # L_2 norm
    batch_norm = False
    batch_norm_momentum = 0.1
    batch_norm_eps = 1e-05
    # dropout
    dropout = 0.2  # 0.4 is better
    last_dropout = 0.2
    # activation
    activation = 'tanh'
    last_activation = 'tanh'


    # loss
    loss = 'mrl'  # [dsl]
    margin = 0.2
    direction = 't2i'  # ['t2i', 'bidir']. only valid for mrl loss
    # Use max instead of sum in the rank loss
    max_violation = True  # only valid for mrl loss
    # cost style (sum|mean)
    cost_style = 'sum'  # only valid for mrl loss
    # Similarity measure used (cosine|order)
    measure = 'cosine'

    # optimizer
    optimizer = 'rmsprop'
    # Initial learning rate.
    lr = 0.0001
    lr_decay_rate = 0.99
    # Gradient clipping threshold
    grad_clip = 2

    # half model: float16 tensor, half the memory. Recommend to True.
    float16 = False

    attention_types = ('attention_noAverageMul_Ave', 
                      'average_AverageMul_noAve', 
                      'con_attention',
                      'fc_attention',
                       'just_average',  # 4
                       'muti_head_attention',
                      'attention3',
                      'attention_noAveNoAverageMul', 
                      'concat',  
                      'attention_averageMul', 
                      'muti_head_attention_official', 
                      'my_self_attention', 
                      'Multi_head_MyApply_Attention',  
                      'Multi_head_MyApply_FusionAttention', 
                      'Multi_head_Attention_layer_norm', 
                      'Multi_head_Attention_distinct_fc', 
                      'Attention_MMT',  # 16 Attention_MMT
                      )
    attention_l2norm = False
    muti_head_attention_official = {'agg': 'mean'}
    vis_attentions = attention_types

    vis_no_transform = []  # ['clip_finetune_8frame_uniform_1103']#[ 'clip_finetune_0922']
    txt_no_transform = []  # ['CLIP_encoder']


    my_self_attention_output_types = ['mean', 'max', 'first', 'last', 'cls_embedding',
                                      'concat', 'max_embedding', 'mean_embedding', 'random', 'second',
                                      'third', 'Attention_1']
    my_self_attention_output_type = my_self_attention_output_types[0]


    # Txt Attention
    txt_attentions = attention_types
    txt_attention = attention_types[1]

    txt_attention_global_decay_rate = 0.8 
    txt_expert_embedding = {'expert': False, 'l2norm': False}

    # visual Attention
    vid_feats = ['mean_resnext101_resnet152', 'irCSN_152_ig65m_16frms',
                 'mean_pyresnext-101_rbps13k,flatten0_output,os', 'ipcsn_sports1m_32frms',
                 'mean_C3d_resneXt101_16f', 'mean_resnext101_32x48d_wsl,avgpool,os',
                 # 21.5.7
                 'mean_clip_frame_feat_ViT-B_32,os', 'HowTo100M_TimeSformer_divST_96x4_224',
                 'X3D_L', 'I3D_NLN_8x8_R50',
                 ]
    vis_feat_add_concat = False  
    vis_attention = attention_types[1]
    vis_attention_global_decay_rate = 0.8
    vis_expert_embedding = {'expert': False, 'l2norm': False}

    multi_head_attention = {  # if attention include muti_head_attention
        'dropout': 0.0, 'heads': 4, 'embed_dim_qkv': 2048 // 4,

    }
    attention_param_each_head = {
        "with_ave": True, "mul": False, 'split_head': True,
    }
    multi_space = True 

    # visual frame feats
    max_frame = 200
    frame_feat_input = False
    frame_feat_with_video_feat = False 
    vid_frame_feats = ['pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os',
                       ]
    vis_frame_attention = attention_types[1]
    vis_frame_addFC = True


    tranformer_encoder_opt = {
        'nhead': 4, 'num_layers': 4
    }
    add_vid_feats = False 


    csn = False

    SGRAF = False
    muti_feat = 'vg_label_feat_36dim_repeat'  # ['vg_label_feat_36dim_repeat', 'vg_label_feat']
    img_dim = 2048
    word_dim = 300
    embed_size = 1024
    sim_dim = 256
    num_layers = 1  # Number of GRU layers.
    bi_gru = False
    no_imgnorm = True
    no_txtnorm = True
    module_name = 'SGR'  # SGR, SAF
    sgr_step = 3  # step of SGR

    task2 = False
    # task2 text embbeding
    txt_feature_task2 = 'bow'
    txt_fc_layers_task2 = '0-0' 
    # task2 multi label
    text_encoding_task2 = 'bow_nsw'
    threshold_task2 = 5 
    bow_norm_task2 = 0 
    batch_norm_task2 = True
    activation_task2 = 'sigmoid' 
    dropout_task2 = 0.1
    # task2 visual embbeding
    vis_fc_layers_task2 = '0-0' 

    task3_neg_weight=1
    task3_start=-1
    task3_end=100
    task3_loss_weight=1
    task3_margin = 0.2
    # loss
    loss_lambda = 0.2
    # Similarity measure used (cosine|order|hist)  # hist is jaccard sim
    measure_task2 = 'hist'
    # parameter that balance latent space and task2 space (concept space)
    alpha = 0.2

    frame_loader = False
    frame_sample_type_train = 'random'  # ['uniform', 'random']
    frame_sample_type_test = 'uniform'
    sample_frame = 8 

    # Feature re-learning
    txt_fc_same_with_vis_fc = False 
    txt_fc_same_with_vis_fc_dict = {
        'CLIP_encoder': 'clip2video_global_visual_output_MSVD', 
    }


    attack_scales = [1024]  #  "[1024], [300,400,500,600,700,800,900,1024], [300,350,400,450,500,550,600,650,700,750,800,850,900,950,1024]
    attack_iters = 300
    attack_lr = 0.01
    attack_lam = 1
    attack_sigma_blur = 0.0  # no blur if 0.0
    attack_mode = 'global'  # "global | tensor | hist"
    attack_variant = ""
    patch_ratio = 0.35
    only_keyword = False
