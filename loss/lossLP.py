from loss import loss_base
from transformer.Master_base import Lap_Pyramid_Conv


class LossLP_SSIM(loss_base.LossBase):
    def __init__(self, args):
        super(LossLP_SSIM, self).__init__(
            args
        )

        self.lap_pyramid = Lap_Pyramid_Conv(2)  # !!!!!

    def get_loss(self):
        return self.loss

    def forward(self, model_out, target):
        target_list = self.lap_pyramid.pyramid_decom(img=target)
        target_out = [target, target_list[0], target_list[1], target_list[2], target]
        return super(LossLP_SSIM, self).forward(model_out, target_out)