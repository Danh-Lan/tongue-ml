import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
	def __init__(self, in_channels=3, num_classes=2):
		super(UNet, self).__init__()

		self.down1 = DownSample(in_channels, 64)
		self.down2 = DownSample(64, 128)
		self.down3 = DownSample(128, 256)
		self.down4 = DownSample(256, 512)

		self.bottleneck = DoubleConv(512, 1024)

		self.up4 = UpSample(1024, 512)
		self.up3 = UpSample(512, 256)
		self.up2 = UpSample(256, 128)
		self.up1 = UpSample(128, 64)

		self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

	def forward(self, x):
		down1, p1 = self.down1(x)
		down2, p2 = self.down2(p1)
		down3, p3 = self.down3(p2)
		down4, p4 = self.down4(p3)

		bottleneck = self.bottleneck(p4)

		up4 = self.up4(bottleneck, down4)
		up3 = self.up3(up4, down3)
		up2 = self.up2(up3, down2)
		up1 = self.up1(up2, down1)

		out = self.final_conv(up1)

		return out

if __name__ == "__main__":
	input_image = torch.randn((1, 3, 128, 128)) 
	model = UNet(in_channels=3, num_classes=2)
	output = model(input_image)

	print("Output shape:", output.shape)