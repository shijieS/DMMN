import torch
from torch import nn

from ..models import resnet, pre_act_resnet, wide_resnet, resnext, densenet
from ..models import extra_net

def generate_base_model(opt):
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False

    assert opt.model_name in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']

    if opt.model_name == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 18:
            model = resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 34:
            model = resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 50:
            model = resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 101:
            model = resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 152:
            model = resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 200:
            model = resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
    elif opt.model_name == 'wideresnet':
        assert opt.model_depth in [50]

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, k=opt.wide_resnet_k,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
    elif opt.model_name == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        if opt.model_depth == 50:
            model = resnext.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 101:
            model = resnext.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
        elif opt.model_depth == 152:
            model = resnext.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
    elif opt.model_name == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
    elif opt.model_name == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        if opt.model_depth == 121:
            model = densenet.densenet121(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 169:
            model = densenet.densenet169(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 201:
            model = densenet.densenet201(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 264:
            model = densenet.densenet264(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model


def generate_resnext101(num_classes, frame_size, num_frames, cuda):
    model = resnext.resnet101(num_classes=num_classes,
                              shortcut_type="B",
                              # cardinality=32,
                              cardinality=8,
                              frame_size=frame_size,
                              frame_duration=num_frames,
                              last_fc=None)



    if cuda:
        model = model.cuda()
        # motion_model = nn.DataParallel(motion_model, device_ids=None)

    return model

def generate_extra_model(cuda, inplanes = 1024):
    extra_model = extra_net.ExtraNet(resnext.ResNeXtBottleneck, [3, 3], shortcut_type="B", cardinality=1, inplanes=inplanes)

    if cuda:
        extra_model = extra_model.cuda()
        # extra_model = nn.DataParallel(extra_model, device_ids=None)

    return extra_model
