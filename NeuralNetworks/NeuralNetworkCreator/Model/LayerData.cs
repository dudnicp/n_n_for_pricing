using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using Prism.Mvvm;
using System.Collections.ObjectModel;
using NeuralNetworkCreator.Services;
using NeuralNetwork.Activators;
using NeuralNetwork.GradientAdjustment;

namespace NeuralNetworkCreator.Model
{
    public class LayerData : BindableBase
    {
        private ActivatorType _activatorType;
        private GradientAdjustmentType _gradientAdjustmentType;
        private IGradientAdjustmentParameters _gradientAdjustmentParameters;
        private int _layerSize;
        private LayerType _layerType;

        public ActivatorType ActivatorType
        {
            get => _activatorType;
            set => SetProperty(ref _activatorType, value);
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

        public int LayerSize
        {
            get => _layerSize;
            set => SetProperty(ref _layerSize, value);
        }

        public LayerType LayerType
        {
            get => _layerType;
            set => SetProperty(ref _layerType, value);
        }

        public LayerData()
        {
            ActivatorType = ActivatorType.Identity;
            LayerSize = 1;
            LayerType = LayerType.Standard;
            GradientAdjustmentType = GradientAdjustmentType.FixedLearningRate;
            GradientAdjustmentParameters = new FixedLearningRateParameters(1.0);
        }
    }
}
