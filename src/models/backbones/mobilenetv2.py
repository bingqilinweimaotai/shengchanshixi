""" 此文件改编自 https://github.com/thuyngch/Human-Segmentation-PyTorch"""

import math
import json
from functools import reduce

import torch
from torch import nn


#------------------------------------------------------------------------------
#  实用函数
#------------------------------------------------------------------------------

def _make_divisible(v, divisor, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# 确保向下取整不会减少超过10%
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def conv_bn(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


def conv_1x1_bn(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


#------------------------------------------------------------------------------
#  倒残差块类
#------------------------------------------------------------------------------

class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expansion, dilation=1):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = round(inp * expansion)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expansion == 1:
			self.conv = nn.Sequential(
				# 深度卷积
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# 点卷积-线性
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)
		else:
			self.conv = nn.Sequential(
				# 点卷积
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# 深度卷积
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# 点卷积-线性
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


#------------------------------------------------------------------------------
#  MobileNetV2类
#------------------------------------------------------------------------------

class MobileNetV2(nn.Module):
	def __init__(self, in_channels, alpha=1.0, expansion=6, num_classes=1000):
		super(MobileNetV2, self).__init__()
		self.in_channels = in_channels
		self.num_classes = num_classes
		input_channel = 32
		last_channel = 1280
		interverted_residual_setting = [
			# t, c, n, s
			[1        , 16, 1, 1],
			[expansion, 24, 2, 2],
			[expansion, 32, 3, 2],
			[expansion, 64, 4, 2],
			[expansion, 96, 3, 1],
			[expansion, 160, 3, 2],
			[expansion, 320, 1, 1],
		]

		# 构建第一层
		input_channel = _make_divisible(input_channel*alpha, 8)
		self.last_channel = _make_divisible(last_channel*alpha, 8) if alpha > 1.0 else last_channel
		self.features = [conv_bn(self.in_channels, input_channel, 2)]

		# 构建倒残差块
		for t, c, n, s in interverted_residual_setting:
			output_channel = _make_divisible(int(c*alpha), 8)
			for i in range(n):
				if i == 0:
					self.features.append(InvertedResidual(input_channel, output_channel, s, expansion=t))
				else:
					self.features.append(InvertedResidual(input_channel, output_channel, 1, expansion=t))
				input_channel = output_channel

		# 构建最后几层
		self.features.append(conv_1x1_bn(input_channel, self.last_channel))

		# 使其成为nn.Sequential
		self.features = nn.Sequential(*self.features)

		# 构建分类器
		if self.num_classes is not None:
			self.classifier = nn.Sequential(
				nn.Dropout(0.2),
				nn.Linear(self.last_channel, num_classes),
			)

		# 初始化权重
		self._init_weights()

	def forward(self, x):
		# 阶段1
		x = self.features[0](x)
		x = self.features[1](x)
		# 阶段2
		x = self.features[2](x)
		x = self.features[3](x)
		# 阶段3
		x = self.features[4](x)
		x = self.features[5](x)
		x = self.features[6](x)
		# 阶段4
		x = self.features[7](x)
		x = self.features[8](x)
		x = self.features[9](x)
		x = self.features[10](x)
		x = self.features[11](x)
		x = self.features[12](x)
		x = self.features[13](x)
		# 阶段5
		x = self.features[14](x)
		x = self.features[15](x)
		x = self.features[16](x)
		x = self.features[17](x)
		x = self.features[18](x)

		# 分类
		if self.num_classes is not None:
			x = x.mean(dim=(2,3))
			x = self.classifier(x)
			
		# 输出
		return x

	def _load_pretrained_model(self, pretrained_file):
		pretrain_dict = torch.load(pretrained_file, map_location='cpu')
		model_dict = {}
		state_dict = self.state_dict()
		print("[MobileNetV2] 加载预训练模型...")
		for k, v in pretrain_dict.items():
			if k in state_dict:
				model_dict[k] = v
			else:
				print(k, "被忽略")
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
