using NeuralNetwork.Common.Activators;
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
        private static List<IActivator> _activators;

        public static List<IActivator> Activators
        {
            get
            {
                if (_activators == null)
                {
                    foreach (ActivatorType activatorType in (ActivatorType[]) Enum.GetValues(typeof(ActivatorType)))
                    {
                        _activators = new List<IActivator>();
                        _activators.Add(ActivatorFactory.Build(activatorType));
                    }
                }
                return _activators;
            }
        }
    }
}
