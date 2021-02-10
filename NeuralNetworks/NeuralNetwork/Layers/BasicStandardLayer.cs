using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.GradientAdjustment;
using System;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Common;

namespace NeuralNetwork.Layers
{
    public class BasicStandardLayer : ILayer, IComponentWithMode
    {
        private int _layerSize;
        private int _inputSize;
        private int _batchSize;
        private Matrix<double> _activation;
        private Matrix<double> _previousActivation;
        private Matrix<double> _weightedError;
        private Matrix<double> _biasedError;
        private IActivator _activator;
        private Matrix<double> _netInput;
        private Matrix<double> _weights;
        private Matrix<double> _bias;
        private Matrix<double> _weightsGradient;
        private Vector<double> _biasGradient;
        private Vector<double> _onesM;
        private AbstractGradientAdjustmentStrategy _gradientAdjustmentStrategy;

        public int LayerSize => _layerSize;

        public int InputSize => _inputSize;

        public int BatchSize
        {
            get => _batchSize;
            set
            {
                _batchSize = value;
                _netInput = Matrix<double>.Build.Dense(LayerSize, BatchSize);
                _previousActivation = Matrix<double>.Build.Dense(InputSize, BatchSize);
                _activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);
                _weightedError = Matrix<double>.Build.Dense(InputSize, BatchSize);
                _biasedError = Matrix<double>.Build.Dense(LayerSize, BatchSize);
                _onesM = Vector<double>.Build.Dense(BatchSize, 1.0);
                _onesM.Divide(BatchSize, _onesM);
                Vector<double> temp = Vector<double>.Build.DenseOfVector(_bias.Column(0));
                _bias = Matrix<double>.Build.Dense(temp.Count, _batchSize);
                for (int i = 0; i < _batchSize; i++)
                {
                    _bias.SetColumn(i, temp);
                }
            }
        }

        public Matrix<double> Activation => _activation;

        public Matrix<double> PreviousActivation => _previousActivation;

        public Matrix<double> WeightedError => _weightedError;

        public Matrix<double> BiasedError => _biasedError;

        public IActivator Activator => _activator;

        public Matrix<double> NetInput => _netInput;

        public Matrix<double> Weights => _weights;

        public Matrix<double> Bias => _bias;

        public Matrix<double> WeightsGradient => _weightsGradient;

        public Vector<double> BiasGradient => _biasGradient;

        public Vector<double> OnesM => _onesM;

        public AbstractGradientAdjustmentStrategy GradientAdjustmentStrategy
        {
            get => _gradientAdjustmentStrategy;
            set => _gradientAdjustmentStrategy = value;
        }
        public Mode Mode { get; set; }

        public BasicStandardLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator, 
            IGradientAdjustmentParameters gradientAdjustmentParameters)
        {
            _inputSize = weights.RowCount;
            _layerSize = weights.ColumnCount;
            _weights = weights;
            _bias = bias;
            _weightsGradient = Matrix<double>.Build.Dense(InputSize, LayerSize);
            _biasGradient = Vector<double>.Build.Dense(LayerSize);
            _activator = activator;
            _gradientAdjustmentStrategy = GradientAdjustmentStrategyFactory.Build(gradientAdjustmentParameters);
            BatchSize = batchSize;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            GradientAdjustmentStrategy.BackPropagate(this, upstreamWeightedErrors);
        }

        public void Propagate(Matrix<double> input)
        {
            input.CopyTo(PreviousActivation);
            Weights.TransposeThisAndMultiply(input, NetInput);
            NetInput.Add(Bias, NetInput);
            NetInput.Map(Activator.Apply, Activation);
        }

        public void UpdateParameters()
        {
            GradientAdjustmentStrategy.UpdateParameters(this);
        }
    }
}