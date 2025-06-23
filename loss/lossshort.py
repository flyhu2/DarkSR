from loss import loss_base


class LossShort(loss_base.LossBase):
    def __init__(self, args):
        super(LossShort, self).__init__(
            args
        )

    def get_loss(self):
        return self.loss

    def forward(self, model_out, target):
        # r, g, b = target.split(1, 1)
        return super(LossShort, self).forward(model_out, target)
