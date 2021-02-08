using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers.AdditionalLayerParameters
{
    public class StandardParameters : IAdditionalLayerParameters
    {
        public LayerType LayerType => LayerType.Standard;

        public StandardParameters()
        {
        }
    }
}
