﻿using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    class WeightDecayLayer : ILayer
    {
        public int LayerSize => UnderlyingLayer.LayerSize;

        public int InputSize => UnderlyingLayer.InputSize;

        public int BatchSize { get; set; }

        public Matrix<double> Activation => UnderlyingLayer.Activation;

        public Matrix<double> WeightedError => UnderlyingLayer.WeightedError;

        public BasicStandardLayer UnderlyingLayer { get; }

        public double DecayRate { get; }

        public WeightDecayLayer(BasicStandardLayer underlyingLayer, double decay, int batchSize)
        {
            UnderlyingLayer = underlyingLayer;
            DecayRate = decay;
            BatchSize = batchSize;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);
        }

        public void Propagate(Matrix<double> input)
        {
            UnderlyingLayer.Propagate(input);
        }

        public void UpdateParameters()
        {
            UnderlyingLayer.Weights.Multiply(1 - DecayRate, UnderlyingLayer.Weights);
            UnderlyingLayer.Bias.Multiply(1 - DecayRate, UnderlyingLayer.Bias);
            UnderlyingLayer.UpdateParameters();
        }
    }
}
