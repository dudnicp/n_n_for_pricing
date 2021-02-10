using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    public class WeightDecayLayer : ILayer, IComponentWithMode
    {
        private int _batchSize;
        private BasicStandardLayer _underlyingLayer;
        private double _decayRate;

        public int LayerSize => UnderlyingLayer.LayerSize;

        public int InputSize => UnderlyingLayer.InputSize;

        public int BatchSize
        {
            get => _batchSize;
            set
            {
                _batchSize = value;
                _underlyingLayer.BatchSize = _batchSize;
            }
        }

        public Matrix<double> Activation => UnderlyingLayer.Activation;

        public Matrix<double> WeightedError => UnderlyingLayer.WeightedError;

        public BasicStandardLayer UnderlyingLayer => _underlyingLayer;

        public double DecayRate => _decayRate;

        public Mode Mode { get; set; }

        public WeightDecayLayer(BasicStandardLayer underlyingLayer, double decay, int batchSize)
        {
            _underlyingLayer = underlyingLayer;
            _decayRate = decay;
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
