using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.GradientAdjustment;
using Prism.Mvvm;

namespace NeuralNetworkCreator.Model
{
    public class NetworkData : BindableBase
    {
        private int _batchSize;
        private int _inputSize;
        private ObservableCollection<LayerData> _layers;
        private GradientAdjustmentType _gradientAdjustmentType;
        private IGradientAdjustmentParameters _gradientAdjustmentParameters;
        private ActivatorType _activatorType;

        public int BatchSize 
        {
            get => _batchSize;
            set => SetProperty(ref _batchSize, value);
        }

        public int InputSize
        {
            get => _inputSize;
            set => SetProperty(ref _inputSize, value);
        }

        public ObservableCollection<LayerData> Layers
        {
            get => _layers;
            set => SetProperty(ref _layers, value);
        }

        public GradientAdjustmentType GradientAdjustmentType
        {
            get => _gradientAdjustmentType;
            set
            {
                SetProperty(ref _gradientAdjustmentType, value);
                GradientAdjustmentParameters = GradientAdjustmentParametersFactory.Build(GradientAdjustmentType);
            }
        }

        public IGradientAdjustmentParameters GradientAdjustmentParameters
        {
            get => _gradientAdjustmentParameters;
            set => SetProperty(ref _gradientAdjustmentParameters, value);
        }

        public ActivatorType ActivatorType
        {
            get => _activatorType;
            set => SetProperty(ref _activatorType, value);
        }

        public NetworkData()
        {
            BatchSize = 0;
            InputSize = 0;
            Layers = new ObservableCollection<LayerData>();
            GradientAdjustmentType = GradientAdjustmentType.FixedLearningRate;
            ActivatorType = ActivatorType.Identity;
        }

        public void AddLayer()
        {
            Layers.Add(new LayerData());
        }

        public void RemoveLayer(LayerData layer)
        {
            Layers.Remove(layer);
        }
    }
}
