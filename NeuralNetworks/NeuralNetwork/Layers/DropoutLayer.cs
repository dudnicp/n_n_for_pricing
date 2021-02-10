using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    public class DropoutLayer : ILayer, IComponentWithMode
    {
        private int _layerSize;
        private int _inputSize;
        private int _batchSize;
        private Matrix<double> _activation;
        private Matrix<double> _weightedError;
        private double _keepProbability;
        private Random _rng;
        private Vector<double> _mask;

        public int LayerSize => _layerSize;

        public int InputSize => _inputSize;

        public int BatchSize
        {
            get => _batchSize;
            set
            {
                _batchSize = value;
                _activation = Matrix<double>.Build.Dense(_layerSize, _batchSize);
                _weightedError = Matrix<double>.Build.Dense(_layerSize, _batchSize);
            }
        }

        public Matrix<double> Activation => _activation;

        public Matrix<double> WeightedError => _weightedError;

        public double KeepProbability => _keepProbability;

        public Random Rng => _rng;

        public Vector<double> Mask => _mask;

        public Mode Mode { get; set; }

        public DropoutLayer(int layerSize, double probability, int batchSize, Random rng)
        {
            _layerSize = layerSize;
            _inputSize = _layerSize;
            _keepProbability = probability;
            _rng = rng;
            _mask = Vector<double>.Build.Dense(layerSize);
            BatchSize = batchSize;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            for (int i = 0; i < upstreamWeightedErrors.RowCount; i++)
            {
                Mask.At(i, Bernoulli.Sample(Rng, KeepProbability));
                for (int j = 0; j < upstreamWeightedErrors.ColumnCount; j++)
                {
                    WeightedError.At(i, j, Mask.At(i) * upstreamWeightedErrors.At(i, j));
                }
            }
        }

        public void Propagate(Matrix<double> input)
        {
            if (Mode == Mode.Training)
            {
                for (int i = 0; i < input.RowCount; i++)
                {
                    Mask.At(i, Bernoulli.Sample(Rng, KeepProbability));
                    for (int j = 0; j < input.ColumnCount; j++)
                    {
                        Activation.At(i, j, Mask.At(i) * input.At(i, j));
                    }
                }
            }
            else
            {
                input.CopyTo(Activation);
            }
        }

        public void UpdateParameters()
        {
            // nothing to do
        }
    }
}
