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

                case InputStandardizingLayer inputNorm:
                    return SerializeInputStandardizingLayer(inputNorm);

                case DropoutLayer dropout:
                    return SerializeDropoutLayer(dropout);

                case L2PenaltyLayer penalty:
                    return SerializeL2PenaltyLayer(penalty);

                case WeightDecayLayer decay:
                    return SerializeWeightDecayLayer(decay);

                default:
                    throw new InvalidOperationException("Unknown layer type: " + layer.GetType());
            }
        }

        private static ISerializedLayer SerializeWeightDecayLayer(WeightDecayLayer layer)
        {
            var underlying = Serialize(layer.UnderlyingLayer);
            return new SerializedWeightDecayLayer(underlying, layer.DecayRate);
        }

        private static ISerializedLayer SerializeL2PenaltyLayer(L2PenaltyLayer layer)
        {
            var underlying = Serialize(layer.UnderlyingLayer);
            return new SerializedL2PenaltyLayer(underlying, layer.PenaltyCoefficient);
        }

        private static ISerializedLayer SerializeDropoutLayer(DropoutLayer layer)
        {
            return new SerializedDropoutLayer(layer.LayerSize, layer.KeepProbability);
        }

        private static ISerializedLayer SerializeInputStandardizingLayer(InputStandardizingLayer layer)
        {
            var mean = layer.Mean.ToArray();
            var stdDev = layer.StdDev.ToArray();
            var underlying = Serialize(layer.UnderlyingLayer);
            return new SerializedInputStandardizingLayer(underlying, mean, stdDev);
        }

        private static ISerializedLayer SerializeStandardLayer(BasicStandardLayer layer)
        {
            var bias = layer.Bias.ToColumnArrays()[0];
            var weights = layer.Weights.ToArray();
            return new SerializedStandardLayer(bias, weights, layer.Activator.Type, layer.GradientAdjustmentStrategy.Parameters);
        }
    }
}