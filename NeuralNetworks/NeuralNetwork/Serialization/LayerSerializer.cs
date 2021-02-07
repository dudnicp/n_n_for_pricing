using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    public static class LayerSerializer
    {
        public static ISerializedLayer Serialize(ILayer layer)
        {
            switch (layer)
            {
                case BasicStandardLayer standardLayer:
                    return SerializeStandardLayer(standardLayer);

                default:
                    throw new InvalidOperationException("Unknown layer type: " + layer.GetType());
            }
        }

        private static ISerializedLayer SerializeStandardLayer(BasicStandardLayer standardLayer)
        {
            var bias = standardLayer.Bias.ToColumnArrays()[0];
            var weights = standardLayer.Weights.ToArray();
            var activatorType = standardLayer.Activator.Type;
            return new SerializedStandardLayer(bias, weights, activatorType, standardLayer.GradientAdjustmentStrategy.Parameters);
        }
    }
}