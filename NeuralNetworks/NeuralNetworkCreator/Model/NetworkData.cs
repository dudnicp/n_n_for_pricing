using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Common.Layers;
using Prism.Mvvm;

namespace NeuralNetworkCreator.Model
{
    public class NetworkData : BindableBase
    {
        private int _batchSize;
        private int _inputSize;
        private ObservableCollection<LayerData> _layers;

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

        public NetworkData()
        {
            BatchSize = 0;
            InputSize = 0;
            Layers = new ObservableCollection<LayerData>();
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
