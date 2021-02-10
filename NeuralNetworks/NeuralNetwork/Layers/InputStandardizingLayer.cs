using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    public class InputStandardizingLayer : ILayer, IComponentWithMode
    {
        private int _batchSize;
        private BasicStandardLayer _underlyingLayer;
        private Vector<double> _meanVector;
        private Vector<double> _stdDevVector;
        private Matrix<double> _meanMatrix;
        private Matrix<double> _stdDevMatrix;
        private Matrix<double> _standardizedInput;

        public int LayerSize => UnderlyingLayer.LayerSize;

        public int InputSize => UnderlyingLayer.InputSize;

        public int BatchSize
        {
            get => _batchSize;
            set
            {
                _batchSize = value;
                _meanMatrix = Matrix<double>.Build.Dense(InputSize, _batchSize);
                _stdDevMatrix = Matrix<double>.Build.Dense(InputSize, _batchSize);
                for (int i = 0; i < _batchSize; i++)
                {
                    _meanMatrix.SetColumn(i, _meanVector);
                    _stdDevMatrix.SetColumn(i, _stdDevVector);
                }
                _standardizedInput = Matrix<double>.Build.Dense(InputSize, _batchSize);
                _underlyingLayer.BatchSize = _batchSize;
            }
        }

        public Matrix<double> Activation => UnderlyingLayer.Activation;

        public Matrix<double> WeightedError => UnderlyingLayer.WeightedError;

        public BasicStandardLayer UnderlyingLayer => _underlyingLayer;

        public Matrix<double> Mean => _meanMatrix;

        public Matrix<double> StdDev => _stdDevMatrix;

        public Matrix<double> StandardizedInput => _standardizedInput;

        public Mode Mode { get; set; }

        public InputStandardizingLayer(BasicStandardLayer underlyingLayer, Vector<double> mean, Vector<double> stdDev, int batchSize)
        {
            _underlyingLayer = underlyingLayer;
            _meanVector = mean;
            _stdDevVector = stdDev;
            BatchSize = batchSize;
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
