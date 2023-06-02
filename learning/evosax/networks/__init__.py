from evosax.networks import NetworkMapper
from learning.evosax.networks.lstm_cnn import LSTM_CNN
from learning.evosax.networks.lstm import LSTM
from learning.evosax.networks.mlp import MLP

NetworkMapperGiga = NetworkMapper.copy()
NetworkMapperGiga["LSTM_CNN"] = LSTM_CNN
NetworkMapperGiga["LSTM"] = LSTM
NetworkMapperGiga["MLP"] = MLP