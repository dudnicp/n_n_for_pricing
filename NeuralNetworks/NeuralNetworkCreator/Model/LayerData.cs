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

namespace NeuralNetworkCreator.Model
{
    public class LayerData : BindableBase
    {
        private IActivator _activator;
        private IGradientAdjustmentParameters _gradientAdjustmentParameters;
        private int _layerSize;
        private LayerType _layerType;

        public IActivator Activator
        {
            get => _activator;
            set => SetProperty(ref _activator, value);
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

        public ObservableCollection<IActivator> AviableActivators { get; }

        public ObservableCollection<LayerType> AviableLayerTypes { get; }

        public ObservableCollection<GradientAdjustmentType> AviableGradientAdjustmentTypes { get; }

        public LayerData()
        {
            AviableActivators = new ObservableCollection<IActivator>(AviableActivatorsService.Activators);
            AviableLayerTypes = new ObservableCollection<LayerType>(AviableLayerTypesService.Types);
            AviableGradientAdjustmentTypes = new ObservableCollection<GradientAdjustmentType>(AviableGradientAdjustmentTypesService.Types);
        }
    }
}
