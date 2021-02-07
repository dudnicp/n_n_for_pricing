using NeuralNetwork.Common;
using NeuralNetwork.Common.Serialization;

namespace NeuralNetwork.Serialization
{
    public static class NetworkSerializer
    {
        public static SerializedNetwork Serialize(INetwork network)
        {
            var batchSize = network.BatchSize;
            var layers = network.Layers;
            var serializedLayers = new ISerializedLayer[layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                serializedLayers[i] = LayerSerializer.Serialize(layers[i]);
            }
            return new SerializedNetwork() { BatchSize = batchSize, SerializedLayers = serializedLayers };
        }
    }
}