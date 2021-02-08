using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetworkCreator.Model;
using NeuralNetworkCreator.Services;
using Prism.Commands;
using Prism.Mvvm;

namespace NeuralNetworkCreator.ViewModel
{
    public class MainWindowViewModel : BindableBase
    {
        private NetworkData _network;
        private LayerData _currentLayer;

        public NetworkData Network
        {
            get => _network;
            set => SetProperty(ref _network, value);
        }
        public LayerData CurrentLayer
        {
            get => _currentLayer;
            set
            {
                SetProperty(ref _currentLayer, value);
                RemoveLayerCommand.RaiseCanExecuteChanged();
            }
        }

        public ObservableCollection<ActivatorType> AviableActivators { get; }
        public ObservableCollection<LayerType> AviableLayerTypes { get; }
        public ObservableCollection<GradientAdjustmentType> AviableGradientAdjustmentTypes { get; }

        public DelegateCommand AddLayerCommand { get; }
        public DelegateCommand RemoveLayerCommand { get; }
        public DelegateCommand SaveNetworkCommand { get; }

        private bool LayerNotNull() => CurrentLayer != null;

        public MainWindowViewModel()
        {
            AviableActivators = new ObservableCollection<ActivatorType>(AviableActivatorsService.Activators);
            AviableLayerTypes = new ObservableCollection<LayerType>(AviableLayerTypesService.Types);
            AviableGradientAdjustmentTypes = new ObservableCollection<GradientAdjustmentType>(AviableGradientAdjustmentTypesService.Types);

            AddLayerCommand = new DelegateCommand(AddLayer);
            RemoveLayerCommand = new DelegateCommand(RemoveLayer, LayerNotNull);
            SaveNetworkCommand = new DelegateCommand(SaveNetwork);

            Network = new NetworkData();
        }

        public void AddLayer()
        {
            Network.AddLayer();
        }

        public void RemoveLayer()
        {
            Network.RemoveLayer(CurrentLayer);
            CurrentLayer = null;
        }

        public void SaveNetwork()
        {
            throw new NotImplementedException();
        }
    }
}
