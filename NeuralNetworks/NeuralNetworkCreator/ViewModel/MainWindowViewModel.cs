using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Configuration;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetworkCreator.Model;
using NeuralNetworkCreator.Services;
using Newtonsoft.Json;
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
            string serializedContent;
            try
            {
                serializedContent = JsonConvert.SerializeObject(NetworkSerializer.Serialize(Network));
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message, "Erreur", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            using (var dialog = new SaveFileDialog())
            {
                dialog.InitialDirectory = ConfigurationManager.AppSettings["AbsoluteOutputPath"];
                var result = dialog.ShowDialog();
                if (result == DialogResult.OK)
                {
                    File.WriteAllText($"{dialog.FileName}.json", serializedContent);
                }
            }
            MessageBox.Show("Network saved successfully!", "", MessageBoxButtons.OK, MessageBoxIcon.Information);
            Network = new NetworkData();
        }
    }
}
