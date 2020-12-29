import torchvision.models as models
import timm


def get_classifier(model: str):
    if model in [
        "alexnet",
        "AlexNet",
        "resnet",
        "ResNet",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "vgg",
        "VGG",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19_bn",
        "vgg19",
        "squeezenet",
        "SqueezeNet",
        "squeezenet1_0",
        "squeezenet1_1",
        "inception",
        "Inception3",
        "inception_v3",
        "InceptionOutputs",
        "densenet",
        "DenseNet",
        "densenet121",
        "densenet169",
        "densenet201",
        "densenet161",
        "googlenet",
        "GoogLeNet",
        "mobilenet",
        "MobileNetV2",
        "mobilenet_v2",
        "mnasnet",
        "MNASNet",
        "mnasnet0_5",
        "mnasnet0_75",
        "mnasnet1_0",
        "mnasnet1_3",
        "shufflenetv2",
        "ShuffleNetV2",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
    ]:
        model = eval(f"models.{model}(pretrained=False)")
    elif model in [
        "adv_inception_v3",
        "cspdarknet53",
        "cspresnet50",
        "cspresnext50",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "densenetblur121d",
        "dla34",
        "dla46_c",
        "dla46x_c",
        "dla60",
        "dla60_res2net",
        "dla60_res2next",
        "dla60x",
        "dla60x_c",
        "dla102",
        "dla102x",
        "dla102x2",
        "dla169",
        "dpn68",
        "dpn68b",
        "dpn92",
        "dpn98",
        "dpn107",
        "dpn131",
        "ecaresnet50d",
        "ecaresnet50d_pruned",
        "ecaresnet101d",
        "ecaresnet101d_pruned",
        "ecaresnetlight",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b1_pruned",
        "efficientnet_b2",
        "efficientnet_b2_pruned",
        "efficientnet_b2a",
        "efficientnet_b3",
        "efficientnet_b3_pruned",
        "efficientnet_b3a",
        "efficientnet_em",
        "efficientnet_es",
        "efficientnet_lite0",
        "ens_adv_inception_resnet_v2",
        "ese_vovnet19b_dw",
        "ese_vovnet39b",
        "fbnetc_100",
        "tresnet_l",
        "tresnet_l_448",
        "tresnet_m",
        "tresnet_m_448",
        "tresnet_xl",
        "tresnet_xl_448",
        "tv_densenet121",
        "tv_resnet34",
        "tv_resnet50",
        "tv_resnet101",
        "tv_resnet152",
        "tv_resnext50_32x4d",
        "vit_base_patch16_224",
        "vit_base_patch16_384",
        "vit_base_patch32_384",
        "vit_large_patch16_224",
        "vit_large_patch16_384",
        "vit_large_patch32_384",
        "vit_small_patch16_224",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "xception",
        "xception41",
        "xception65",
        "xception71",
    ]:
        model = eval(f"timm.create_model('{model}', pretrained=False)")
    else:
        raise NotImplementedError()
    return model
