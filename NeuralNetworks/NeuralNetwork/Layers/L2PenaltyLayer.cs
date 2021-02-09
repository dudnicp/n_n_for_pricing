using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    class L2PenaltyLayer : ILayer
    {
        public int LayerSize => UnderlyingLayer.LayerSize;

        public int InputSize => UnderlyingLayer.InputSize;

        public int BatchSize { get; set; }

        public Matrix<double> Activation => UnderlyingLayer.Activation;

        public Matrix<double> WeightedError => UnderlyingLayer.WeightedError;

        public BasicStandardLayer UnderlyingLayer { get; }

        public double PenaltyCoefficient { get; }

        public L2PenaltyLayer(BasicStandardLayer underlyingLayer, double penalty, int batchSize)
        {
            UnderlyingLayer = underlyingLayer;
            PenaltyCoefficient = penalty;
            BatchSize = batchSize;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);
            UnderlyingLayer.WeightsGradient.Add(UnderlyingLayer.Weights.Multiply(PenaltyCoefficient), UnderlyingLayer.WeightsGradient);
        }

        public void Propagate(Matrix<double> input)
        {
            UnderlyingLayer.Propagate(input);
        }

        public void UpdateParameters()
        {
            UnderlyingLayer.UpdateParameters();
        }
    }
}
