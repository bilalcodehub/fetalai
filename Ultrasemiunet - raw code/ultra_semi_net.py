import torch
import torch.nn as nn
from simple_unet import SimpleUNet  # Adjust import path as necessary

class UltraSemiNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, alpha=0.99):
        super(UltraSemiNet, self).__init__()
        # Student and Teacher share the same architecture
        self.student_net = SimpleUNet(in_channels, num_classes)
        self.teacher_net = SimpleUNet(in_channels, num_classes)
        
        # Initialize teacher weights to match student initially
        self._update_teacher(0.0)
        # Exponential moving average factor for teacher update
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, x):
        return self.student_net(x)

    @torch.no_grad()
    def _update_teacher(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        for teacher_param, student_param in zip(self.teacher_net.parameters(), 
                                                self.student_net.parameters()):
            teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
