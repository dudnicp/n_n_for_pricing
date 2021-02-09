using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    class DropoutLayer : ILayer
    {
        public int LayerSize { get; }

        public int InputSize { get; }

        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; }

        public Matrix<double> WeightedError { get; }

        public double KeepProbability { get; }

        public Random Rng { get; }

        public Vector<double> Mask { get; }
        
        public DropoutLayer(int layerSize, double probability, int batchSize, Random rng)
        {
            LayerSize = layerSize;
            BatchSize = batchSize;
            Activation = Matrix<double>.Build.Dense(layerSize, batchSize);
            WeightedError = Matrix<double>.Build.Dense(layerSize, batchSize);
            KeepProbability = probability;
            Rng = rng;
            Mask = Vector<double>.Build.Dense(layerSize);
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
            for (int i = 0; i < input.RowCount; i++)
            {
                Mask.At(i, Bernoulli.Sample(Rng, KeepProbability));
                for (int j = 0; j < input.ColumnCount; j++)
                {
                    Activation.At(i, j, Mask.At(i) * input.At(i, j));
                }
            }
        }

        public void UpdateParameters()
        {
            // nothing to do
        }
    }
}
