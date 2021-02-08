﻿using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    class L2PenaltyLayer : ILayer
    {
        public int LayerSize { get; }

        public int InputSize { get; }

        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; }

        public Matrix<double> WeightedError { get; }

        public ILayer UnderlyingLayer { get; }

        public double PenaltyCoefficient { get; }

        public L2PenaltyLayer(ILayer underlyingLayer, double penalty)
        {
            UnderlyingLayer = underlyingLayer;
            PenaltyCoefficient = penalty;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            throw new NotImplementedException();
        }

        public void Propagate(Matrix<double> input)
        {
            throw new NotImplementedException();
        }

        public void UpdateParameters()
        {
            throw new NotImplementedException();
        }
    }
}
