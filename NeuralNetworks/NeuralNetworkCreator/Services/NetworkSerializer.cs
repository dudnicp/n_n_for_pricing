using NeuralNetwork.Activators;
using NeuralNetwork.Common.Serialization;
using NeuralNetworkCreator.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkCreator.Services
{
    public static class NetworkSerializer
    {
        public static SerializedNetwork Serialize(NetworkData network)
        {
            var batchSize = network.BatchSize;
            var layers = network.Layers;
            var serializedLayers = new ISerializedLayer[layers.Count];
            var inputSize = network.InputSize;
            var gradientAdjustmentParameters = network.GradientAdjustmentParameters;
            var activator = network.ActivatorType;
            for (int i = 0; i < layers.Count; i++)
            {
                int nextLayerSize = 1;
                if (i < layers.Count - 1) nextLayerSize = layers[i + 1].LayerSize;
                serializedLayers[i] = LayerSerializer.Serialize(layers[i], nextLayerSize, inputSize, activator, 
                    gradientAdjustmentParameters, new Random()) ;
            }
            return new SerializedNetwork() { BatchSize = batchSize, SerializedLayers = serializedLayers };
        }
    }
}
