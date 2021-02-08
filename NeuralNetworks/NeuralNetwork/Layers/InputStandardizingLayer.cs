using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    class InputStandardizingLayer : ILayer
    {
        public int LayerSize { get; }

        public int InputSize { get; }

        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; }

        public Matrix<double> WeightedError { get; }

        public ILayer UnderlyingLayer { get; }

        public Vector<double> Mean { get; }

        public Vector<double> StdDev { get; }

        public InputStandardizingLayer(ILayer underlyingLayer, double[] mean, double[] stdDev)
        {
            UnderlyingLayer = underlyingLayer;
            Mean = Vector<double>.Build.DenseOfArray(mean);
            StdDev = Vector<double>.Build.DenseOfArray(stdDev);
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
