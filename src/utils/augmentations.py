import albumentations as A

# train_transform = None
train_transform = A.Compose([
                            #  A.Resize(config.input_H, config.input_W, p=1.0),
                            # #  A.CoarseDropout(max_holes=2, max_height=128, max_width=128, p=0.5),
                            #  A.VerticalFlip(p=0.5),
                            #  A.HorizontalFlip(p=0.5),
                            # #  A.JpegCompression(quality_lower=50, p=0.25),
                            # #  A.Downscale(p=0.25),
                            # #  A.RandomResizedCrop(384, 384, p=0.5),
                            #  A.ShiftScaleRotate(scale_limit=.15, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                            # #  A.GaussianBlur(blur_limit=3, p=0.05),
                            # #  A.GaussNoise(p=0.1),
                            # # #  A.RandomBrightness(limit=0.2, p=1),
                            #  A.OneOf([
                            #   A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                            #   A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20)
                            #  ], p=0.5),
                            #  A.Normalize(p=1.0),

                            #  A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                             A.GaussNoise(p=0.1),
                             A.GaussianBlur(blur_limit=3, p=0.05),
                             A.HorizontalFlip(p=0.5),
                            #  A.VerticalFlip(),
                            #  A.Resize(config.input_H, config.input_W, p=1.0),
                            #  A.IsotropicResize(max_side=config.input_W),
                             A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
                             A.ToGray(p=0.2),
                            #  A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                             A.Normalize(p=1.0),
])
val_transform = A.Compose([
                            #  A.Resize(config.input_H, config.input_W, p=1.0),
                             A.Normalize(p=1.0)
])