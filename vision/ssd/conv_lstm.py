import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


def conv_dw(inp, oup, kernel_size, padding, stride=1):
	return nn.Sequential(
		nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
		nn.BatchNorm2d(inp),
		nn.ReLU(inplace=True),

		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU(inplace=True),
	)




class ConvLSTMCell(nn.Module):
	"""
	Generate a convolutional LSTM cell
	"""

	def __init__(self, input_size, hidden_size):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.bottleneck_gate = nn.Sequential(
			nn.Conv2d(input_size + hidden_size, hidden_size, KERNEL_SIZE, padding=PADDING),
			nn.ReLU())
		self.Gates = nn.Conv2d(hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)
		self.prev_state = None
		self.device = torch.device("cuda")

	def forward(self, input_):

		# get batch and spatial sizes
		batch_size = input_.data.size()[0]
		spatial_size = input_.data.size()[2:]

		# generate empty prev_state, if None is provided
		if self.prev_state is None:
			state_size = [batch_size, self.hidden_size] + list(spatial_size)
			prev_state = (
				Variable(torch.zeros(state_size)),
				Variable(torch.zeros(state_size))
			)
		else:
			prev_state = self.prev_state

		prev_hidden, prev_cell = prev_state

		# data size is [batch, channel, height, width]
		prev_hidden = prev_hidden.to(self.device)
		input_ = input_.to(self.device)
		stacked_inputs = torch.cat((input_, prev_hidden), 1)

		stacked_inputs = self.bottleneck_gate(stacked_inputs)

		gates = self.Gates(stacked_inputs)

		# chunk across channel dimension
		in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

		# apply sigmoid non linearity
		in_gate = f.sigmoid(in_gate)
		remember_gate = f.sigmoid(remember_gate)
		out_gate = f.sigmoid(out_gate)

		# apply tanh non linearity
		cell_gate = f.tanh(cell_gate)

		# compute current cell and hidden state
		cell = (remember_gate.to("cpu") * prev_cell.to("cpu")) + (in_gate.to("cpu") * cell_gate.to("cpu"))
		hidden = out_gate * f.tanh(cell)

		self.prev_state = (hidden, cell)

		return hidden, cell



# class ConvLSTMCell(nn.Module):
# 	"""
# 	Generate a convolutional LSTM cell
# 	"""

# 	def __init__(self, input_size, hidden_size):
# 		super().__init__()
# 		self.input_size = input_size
# 		self.hidden_size = hidden_size
# 		self.prev_state = None
# 		#self.bottleneck_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size = KERNEL_SIZE, stride = 1)
# 		self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = KERNEL_SIZE, stride =1)

# 	def forward(self, input_):

# 		# get batch and spatial sizes
# 		batch_size = input_.data.size()[0]
# 		spatial_size = input_.data.size()[2:]

# 		# generate empty prev_state, if None is provided
# 		if self.prev_state is None:
# 			state_size = [batch_size, self.hidden_size] + list(spatial_size)
# 			prev_state = (
# 				Variable(torch.zeros(state_size)),
# 				Variable(torch.zeros(state_size))
# 			)
# 		else:
# 			prev_state = self.prev_state


# 		prev_hidden, prev_cell = prev_state

	
# 		# data size is [batch, channel, height, width]
# 		prev_hidden = prev_hidden.to('cpu')
# 		stacked_inputs = torch.cat((input_, prev_hidden), 1)

# 		#b_gate = self.bottleneck_gate(stacked_inputs)

# 		gates = self.Gates(stacked_inputs)

# 		print(gates.shape)
		
# 		# chunk across channel dimension
# 		in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
# 		print(in_gate.shape)
# 		print(remember_gate.shape)
# 		print(out_gate.shape)
# 		print(cell_gate.shape)

# 		# apply sigmoid non linearity
# 		in_gate = f.sigmoid(in_gate)
# 		remember_gate = f.sigmoid(remember_gate)
# 		out_gate = f.sigmoid(out_gate)

# 		# apply tanh non linearity
# 		cell_gate = f.tanh(cell_gate)

# 		# compute current cell and hidden state
# 		cell = (remember_gate.to('cpu') * prev_cell.to('cpu')) + (in_gate.to('cpu') * cell_gate.to('cpu'))
# 		hidden = out_gate * f.tanh(cell)

# 		self.prev_state = (hidden, cell)

# 		return hidden, cell


def _main():
	"""
	Run some basic tests on the API
	"""

	# define batch_size, channels, height, width
	b, c, h, w = 1, 3, 150, 150
	d = 5           # hidden state size
	lr = 1e-1       # learning rate
	T = 6           # sequence length
	max_epoch = 20  # number of epochs

	# set manual seed
	torch.manual_seed(0)

	print('Instantiate model')
	model = ConvLSTMCell(3, 512)
	print(repr(model))

	print('Create input and target Variables')
	x = Variable(torch.rand(T, b, c, h, w))
	y = Variable(torch.randn(T, b, d, h, w))

	print(x.shape)

	# print('Create a MSE criterion')
	# loss_fn = nn.MSELoss()

	print('Run for', max_epoch, 'iterations')

	state = None
	loss = 0
	for t in range(0, T):
		print(x[t].size())
		state = model(x[t])
		break
		# print(state[0].size())
		#loss += loss_fn(state[0], y[t])

		#print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.data[0]))

		# zero grad parameters
		# model.zero_grad()

		# compute new grad parameters through time!
		# loss.backward()

		# learning_rate step against the gradient
		# for p in model.parameters():
		#    p.data.sub_(p.grad.data * lr)

	print('Input size:', list(x.data.size()))
	print('Target size:', list(y.data.size()))
	print('Last hidden state size:', list(state[0].size()))
	print('Last cell state size:', list(state[1].size()))


if __name__ == '__main__':
	_main()
