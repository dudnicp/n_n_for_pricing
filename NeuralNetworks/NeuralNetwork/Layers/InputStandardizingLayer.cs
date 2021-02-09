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

        public Matrix<double> Activation => UnderlyingLayer.Activation;

        public Matrix<double> WeightedError => UnderlyingLayer.WeightedError;

        public BasicStandardLayer UnderlyingLayer { get; }

        public Matrix<double> Mean { get; }

        public Matrix<double> StdDev { get; }

        public Matrix<double> StandardizedInput { get; }

        public InputStandardizingLayer(BasicStandardLayer underlyingLayer, double[] mean, double[] stdDev, int batchSize)
        {
            UnderlyingLayer = underlyingLayer;
            Mean = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { mean });
            StdDev = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { stdDev });
            BatchSize = batchSize;
            StandardizedInput = Matrix<double>.Build.Dense(Mean.RowCount, Mean.ColumnCount); 
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);
        }

        public void Propagate(Matrix<double> input)
        {
            input.Subtract(Mean, StandardizedInput);
            StandardizedInput.PointwiseDivide(StdDev, StandardizedInput);
            UnderlyingLayer.Propagate(StandardizedInput);
        }

        public void UpdateParameters()
        {
            UnderlyingLayer.UpdateParameters();
        }
    }
}
