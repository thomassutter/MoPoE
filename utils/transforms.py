
import PIL.Image as Image
from torchvision import transforms


def get_transform_celeba(flags):
    offset_height = (218 - flags.crop_size_img) // 2
    offset_width = (178 - flags.crop_size_img) // 2
    crop = lambda x: x[:, offset_height:offset_height + flags.crop_size_img,
                     offset_width:offset_width + flags.crop_size_img]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(crop),
                                    transforms.ToPILImage(),
                                    transforms.Resize(size=(flags.img_size, flags.img_size),
                                                      interpolation=Image.BICUBIC),
                                    transforms.ToTensor()])

    return transform;


def get_transform_mimic(flags):
    offset_height = (218 - flags.crop_size_img) // 2
    offset_width = (178 - flags.crop_size_img) // 2
    crop = lambda x: x[:, offset_height:offset_height + flags.crop_size_img,
                     offset_width:offset_width + flags.crop_size_img]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(crop),
                                    transforms.ToPILImage(),
                                    transforms.Resize(size=(flags.img_size, flags.img_size),
                                                      interpolation=Image.BICUBIC),
                                    transforms.ToTensor()])

    return transform;


def get_transform_dsprites(flags):
    return None;


def get_transform_mnist(flags):
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.ToPILImage(),
                                          transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),
                                          transforms.ToTensor()])
    return transform_mnist;


def get_transform_svhn(flags):
    transform_svhn = transforms.Compose([transforms.ToTensor()])
    return transform_svhn;
