using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers.AdditionalLayerParameters;
using NeuralNetworkCreator.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkCreator.Services
{
    public static class LayerSerializer
    {
        public static ISerializedLayer Serialize(LayerData layer, int nextLayerSize, int inputSize, ActivatorType activator, 
            IGradientAdjustmentParameters gradientAdjustmentParameters, Random rng)
        {
            switch (layer.LayerType)
            {
                case LayerType.Standard:
                    return SerializeStandardLayer(layer, nextLayerSize, inputSize, activator, 
                        gradientAdjustmentParameters, rng);
                case LayerType.InputStandardizing:
                    return SerializeInputStandardizingLayer(layer, nextLayerSize, inputSize, activator, 
                        gradientAdjustmentParameters, rng);
                case LayerType.Dropout:
                    return SerializeDropoutLayer(layer, nextLayerSize, inputSize, activator, 
                        gradientAdjustmentParameters, rng);
                case LayerType.L2Penalty:
                    return SerializeL2PenaltyLayer(layer, nextLayerSize, inputSize, activator, 
                        gradientAdjustmentParameters, rng);
                case LayerType.WeightDecay:
                    return SerializeWeightDecayLayer(layer, nextLayerSize, inputSize, activator, 
                        gradientAdjustmentParameters, rng);
                default:
                    throw new InvalidOperationException("Unknown layer type: " + layer.LayerType);
            }
        }

        private static ISerializedLayer SerializeStandardLayer(LayerData layer, int nextLayerSize, int inputSize, 
            ActivatorType activator, IGradientAdjustmentParameters gradientAdjustmentParameter, Random rng)
        {
            // weights and bias initialization with Xavier's method
            int layerSize = layer.LayerSize;
            double[,] weights = new double[inputSize, layerSize];
            double[] bias = new double[layerSize];

            double max = Math.Sqrt(6.0 / (layerSize + nextLayerSize));
            double interval = 2 * max;
            for (int i = 0; i < layerSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    weights[j, i] = rng.NextDouble() * interval - max;
                }
                bias[i] = rng.NextDouble() * interval - max;
            }

            return new SerializedStandardLayer(bias, weights, activator, gradientAdjustmentParameter);
        }

        private static ISerializedLayer SerializeInputStandardizingLayer(LayerData layer, int nextLayerSize, int inputSize, 
            ActivatorType activator, IGradientAdjustmentParameters gradientAdjustmentParameter, Random rng)
        {
            double[] mean = { 9.18302351e+01, 6.45327246e+00, 1.34619841e+02, 2.71564421e-01,
                1.09658178e+00, 3.00538796e-02, 3.58017618e-01 }; // nice hard encoding :smirk:
            double[] stdDev = { 4.05251300e+01, 2.58158032e+00, 6.59942893e+01, 7.90843930e-02,
                5.62299193e-01, 5.79778757e-03, 8.51935671e-02 };
            var underlying = SerializeStandardLayer(layer, nextLayerSize, inputSize, activator,
                gradientAdjustmentParameter, rng);
            return new SerializedInputStandardizingLayer(underlying, mean, stdDev);
        }

        private static ISerializedLayer SerializeDropoutLayer(LayerData layer, int nextLayerSize, int inputSize, 
            ActivatorType activator, IGradientAdjustmentParameters gradientAdjustmentParameter, Random rng)
        {
            var additionalParameters = layer.AdditionalParameters as DropoutParameters;
            return new SerializedDropoutLayer(layer.LayerSize, additionalParameters.KeepProbability);
        }

        private static ISerializedLayer SerializeL2PenaltyLayer(LayerData layer, int nextLayerSize, int inputSize, 
            ActivatorType activator, IGradientAdjustmentParameters gradientAdjustmentParameter, Random rng)
        {
            var additionalParameters = layer.AdditionalParameters as L2PenaltyParameters;
            var underlying = SerializeStandardLayer(layer, nextLayerSize, inputSize, activator, 
                gradientAdjustmentParameter, rng);
            return new SerializedL2PenaltyLayer(underlying, additionalParameters.PenaltyCoefficient);
        }

        private static ISerializedLayer SerializeWeightDecayLayer(LayerData layer, int nextLayerSize, int inputSize, 
            ActivatorType activator, IGradientAdjustmentParameters gradientAdjustmentParameter, Random rng)
        {
            var additionalParameters = layer.AdditionalParameters as WeightDecayParameters;
            var underlying = SerializeStandardLayer(layer, nextLayerSize, inputSize, activator,
                gradientAdjustmentParameter, rng);
            return new SerializedL2PenaltyLayer(underlying, additionalParameters.DecayRate);
        }
    }
}
