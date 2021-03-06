import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from args import parser

args = parser.parse_args()


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2



def conv_dw(inp, oup, kernel_size, padding, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride,
                  padding, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )



class BottleNeckLSTM(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bottleneck_gate = nn.Sequential(
            conv_dw(input_size + hidden_size, hidden_size,
                    KERNEL_SIZE, padding=PADDING),
            nn.ReLU())
        self.Gates = conv_dw(hidden_size, 4 * hidden_size,
                             KERNEL_SIZE, padding=PADDING)
        self.hidden_state = None
        self.cell_state = None
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    def initialise_hidden_cell_state(self):
        self.cell_state = None
        self.hidden_state = None

    def forward(self, input_):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if self.hidden_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            self.hidden_state, self.cell_state = (
                torch.zeros(state_size),
                torch.zeros(state_size)
            )
            self.hidden_state = self.hidden_state.to(self.device)
            self.cell_state = self.cell_state.to(self.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, self.hidden_state), 1)

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
        cell = (remember_gate * self.cell_state) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        # self.prev_state = (hidden, cell)

        self.hidden_state = hidden
        self.cell_state = cell

        return hidden, cell


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size,
                               4 * hidden_size, KERNEL_SIZE, padding=PADDING)
        self.hidden_state = None
        self.cell_state = None
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    def initialise_hidden_cell_state(self):
        self.cell_state = None
        self.hidden_state = None

    def forward(self, input_):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if self.hidden_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            self.hidden_state, self.cell_state = (
                torch.zeros(state_size),
                torch.zeros(state_size)
            )
            self.hidden_state = self.hidden_state.to(self.device)
            self.cell_state = self.cell_state.to(self.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, self.hidden_state), 1)

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
        # compute current cell and hidden state
        cell = (remember_gate * self.cell_state) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        # self.prev_state = (hidden, cell)

        self.hidden_state = hidden
        self.cell_state = cell

        return hidden, cell



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

    # print(model.hidden_state.detach_())
    # print(model.cell_state.detach_())

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
