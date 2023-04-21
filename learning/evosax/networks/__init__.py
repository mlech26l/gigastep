from evosax.networks import NetworkMapper
from learning.evosax.networks.lstm_cnn import LSTM_CNN


NetworkMapperGiga = NetworkMapper.copy()
NetworkMapperGiga["LSTM_CNN"] = LSTM_CNN