using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    public class L2PenaltyLayer : ILayer, IComponentWithMode
    {
        private int _batchSize;
        private BasicStandardLayer _underlyingLayer;
        private double _penaltyCoefficient;

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

        public double PenaltyCoefficient => _penaltyCoefficient;

        public Mode Mode { get; set; }

        public L2PenaltyLayer(BasicStandardLayer underlyingLayer, double penalty, int batchSize)
        {
            _underlyingLayer = underlyingLayer;
            _penaltyCoefficient = penalty;
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
