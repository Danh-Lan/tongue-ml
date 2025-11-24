import torch
import torch.nn as nn

class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class DownSample(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.conv = DoubleConv(in_channels, out_channels)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x):
		down = self.conv(x)
		p = self.pool(down)
		return down, p

class UpSample(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x, skip):
		x = self.up(x)
		x = torch.cat([x, skip], 1)
		return self.conv(x)