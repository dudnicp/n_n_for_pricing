using NeuralNetwork.Common.Activators;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using NeuralNetwork.Activators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkCreator.Services
{
    public static class AviableActivatorsService
    {
        private static List<ActivatorType> _activators;

        public static List<ActivatorType> Activators
        {
            get
            {
                if (_activators == null)
                {
                    _activators = new List<ActivatorType>();
                    foreach (ActivatorType type in (ActivatorType[]) Enum.GetValues(typeof(ActivatorType)))
                    {
                        _activators.Add(type);
                    }
                }
                return _activators;
            }
        }
    }
}
