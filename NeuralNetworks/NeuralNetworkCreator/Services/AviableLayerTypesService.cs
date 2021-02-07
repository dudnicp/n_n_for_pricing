using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkCreator.Services
{
    public static class AviableLayerTypesService
    {
        private static List<LayerType> _types;

        public static List<LayerType> Types
        {
            get
            {
                if (_types == null)
                {
                    _types = new List<LayerType>();
                    foreach (LayerType type in (LayerType[])Enum.GetValues(typeof(LayerType)))
                    {
                        _types.Add(type);
                    }
                }
                return _types;
            }
        }
    }
}
