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
using NeuralNetwork.Layers.AdditionalLayerParameters;

namespace NeuralNetworkCreator.Model
{
    public class LayerData : BindableBase
    {
        private int _layerSize;
        private LayerType _layerType;
        private IAdditionalLayerParameters _additionalParameters;

        public int LayerSize
        {
            get => _layerSize;
            set => SetProperty(ref _layerSize, value);
        }

        public LayerType LayerType
        {
            get => _layerType;
            set
            {
                SetProperty(ref _layerType, value);
                AdditionalParameters = LayerParametersFactory.Build(LayerType);
            }
        }

        public IAdditionalLayerParameters AdditionalParameters
        {
            get => _additionalParameters;
            set => SetProperty(ref _additionalParameters, value);
        }

        public LayerData()
        {
            LayerSize = 1;
            LayerType = LayerType.Standard;
        }
    }
}
