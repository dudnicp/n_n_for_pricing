using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworkCreator.Model;
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
            set => SetProperty(ref _currentLayer, value);
        }

        public DelegateCommand AddLayerCommand { get; }
        public DelegateCommand RemoveLayerCommand { get; }
        public DelegateCommand SaveLayerCommand { get; }
        public DelegateCommand SaveNetworkCommand { get; }

        private bool LayerNotNull() => CurrentLayer != null;

        public MainWindowViewModel()
        {
            Network = new NetworkData();
            AddLayerCommand = new DelegateCommand(AddLayer);
            RemoveLayerCommand = new DelegateCommand(RemoveLayer, LayerNotNull);
            SaveLayerCommand = new DelegateCommand(SaveLayer, LayerNotNull);
            SaveNetworkCommand = new DelegateCommand(SaveNetwork);
        }

        public void AddLayer()
        {
            Network.AddLayer();
        }

        public void RemoveLayer()
        {
            throw new NotImplementedException();
        }

        public void SaveLayer()
        {
            throw new NotImplementedException();
        }

        public void SaveNetwork()
        {
            throw new NotImplementedException();
        }
    }
}
