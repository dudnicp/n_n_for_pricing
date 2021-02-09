﻿using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    public static class LayerDeserializer
    {
        public static ILayer Deserialize(ISerializedLayer serializedLayer, int batchSize)
        {
            switch (serializedLayer.Type)
            {
                case LayerType.Standard:
                    var serializedStandard = serializedLayer as SerializedStandardLayer;
                    return DeserializeStandardLayer(serializedStandard, batchSize);
                case LayerType.InputStandardizing:
                    var serializedInputStandardized = serializedLayer as SerializedInputStandardizingLayer;
                    return DeserializeInputStandardizedLayer(serializedInputStandardized, batchSize);
                case LayerType.Dropout:
                    var serializedDropout = serializedLayer as SerializedDropoutLayer;
                    return DeserializeDropoutLayer(serializedDropout, batchSize);
                case LayerType.L2Penalty:
                    var serializedL2Penalty = serializedLayer as SerializedL2PenaltyLayer;
                    return DeserializeL2PenaltyLayer(serializedL2Penalty, batchSize);
                case LayerType.WeightDecay:
                    var serializedWeightDecay = serializedLayer as SerializedWeightDecayLayer;
                    return DeserializeWeightDecayLayer(serializedWeightDecay, batchSize);
                default:
                    throw new InvalidOperationException("Unknown layer type to deserialize: " + serializedLayer.Type);
            }
        }

        private static ILayer DeserializeStandardLayer(SerializedStandardLayer serializedStandard, int batchSize)
        {
            var weights = Matrix<double>.Build.DenseOfArray(serializedStandard.Weights);
            var bias = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { serializedStandard.Bias });
            var activator = ActivatorFactory.Build(serializedStandard.ActivatorType);
            var gradientAdjustmentParameters = serializedStandard.GradientAdjustmentParameters;
            return new BasicStandardLayer(weights, bias, batchSize, activator, gradientAdjustmentParameters);
        }

        private static ILayer DeserializeInputStandardizedLayer(SerializedInputStandardizingLayer serializedInputStandardized, int batchSize)
        {
            var underlying = Deserialize(serializedInputStandardized.UnderlyingSerializedLayer, batchSize);
            var mean = serializedInputStandardized.Mean;
            var stdDev = serializedInputStandardized.StdDev;
            return new InputStandardizingLayer(underlying, mean, stdDev, batchSize);
        }

        private static ILayer DeserializeDropoutLayer(SerializedDropoutLayer serializedDropout, int batchSize)
        {
            var layerSize = serializedDropout.LayerSize;
            var probability = serializedDropout.KeepProbability;
            return new DropoutLayer(layerSize, probability, batchSize);
        }

        private static ILayer DeserializeL2PenaltyLayer(SerializedL2PenaltyLayer serializedL2Penalty, int batchSize)
        {
            var underlying = Deserialize(serializedL2Penalty.UnderlyingSerializedLayer, batchSize);
            var penalty = serializedL2Penalty.PenaltyCoefficient;
            return new L2PenaltyLayer(underlying, penalty, batchSize);
        }

        private static ILayer DeserializeWeightDecayLayer(SerializedWeightDecayLayer serializedWeightDecay, int batchSize)
        {
            var underlying = Deserialize(serializedWeightDecay.UnderlyingSerializedLayer, batchSize);
            var decay = serializedWeightDecay.DecayRate;
            return new WeightDecayLayer(underlying, decay, batchSize);
        }
    }
}